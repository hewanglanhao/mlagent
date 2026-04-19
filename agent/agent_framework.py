from __future__ import annotations

import argparse
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from llm.openai_client import gpt_client

from .codegen import MicroBenchmarkGenerationModule
from .decision import ExperimentDecisionAndFeedbackModule
from .executor import CompilationAndExecutionModule
from .models import MetricEstimate, ProbeAttempt, ValidationResult
from .output_writer import OutputGenerationModule
from .parsers import parse_ncu_csv, parse_program_output
from .profiler import NCUProfilingModule
from .reasoning import ReasoningLogger
from .result_inference import DERIVED_ALIASES, ResultParsingAndInferenceModule
from .retry import ErrorHandlingAndRetryModule
from .spec_reader import TaskReadingModule
from .strategy import BenchmarkStrategySelectionModule
from .validation import CrossValidationModule


DEFAULT_WORKSPACE_ROOT = Path("/workspace")
EARLY_STOP_PRIMARY_CONFIDENCE = 0.90


class GPUProbeAgent:
    def __init__(
        self,
        *,
        workspace_root: Path,
        output_path: Path,
        target_spec_path: Path,
        max_rounds_per_plan: int | None = None,
    ):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.workspace_root = workspace_root
        self.output_path = output_path
        self.target_spec_path = target_spec_path
        self.run_dir = workspace_root / "mlagent_artifacts" / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.logger = ReasoningLogger(self.run_dir / "reasoning_log.jsonl")
        self.max_rounds_per_plan = max_rounds_per_plan

        self.task_reader = TaskReadingModule(self.logger)
        self.strategy_selector = BenchmarkStrategySelectionModule(self.logger)
        self.codegen = MicroBenchmarkGenerationModule(gpt_client, self.run_dir, self.logger)
        self.executor = CompilationAndExecutionModule(self.run_dir, self.logger)
        self.profiler = NCUProfilingModule(self.run_dir, self.logger)
        self.validator = CrossValidationModule(self.logger)
        self.retry_handler = ErrorHandlingAndRetryModule(self.logger)
        self.decision = ExperimentDecisionAndFeedbackModule(self.logger)
        self.result_inference = ResultParsingAndInferenceModule(self.logger)
        self.output_writer = OutputGenerationModule(gpt_client, self.logger)

    def run(self) -> Path:
        spec = self.task_reader.read(self.target_spec_path)
        plans = self.strategy_selector.build_plans(spec)
        if self.max_rounds_per_plan is not None:
            for plan in plans:
                plan.max_rounds = min(plan.max_rounds, self.max_rounds_per_plan)

        print(f"Loaded {len(spec.targets)} targets from {spec.source_path}")
        print(f"Built {len(plans)} benchmark plans")

        attempts: list[ProbeAttempt] = []
        accepted_primary_attempts: dict[str, ProbeAttempt] = {}
        for plan in plans:
            skip_reason = self._should_skip_plan_via_early_stop(plan, accepted_primary_attempts)
            if skip_reason:
                plan.skipped = True
                plan.skip_reason = skip_reason
                print(f"[plan] {plan.plan_id} skipped via early-stop")
                self.logger.log(
                    "agent_framework",
                    "skipped_plan_via_early_stop",
                    plan_id=plan.plan_id,
                    target=plan.primary_target or plan.targets,
                    probe_family=plan.probe_family,
                    probe_variant=plan.probe_variant,
                    plan_role=plan.plan_role,
                    skip_reason=skip_reason,
                )
                continue

            print(f"[plan] {plan.plan_id} -> {plan.probe_family} ({', '.join(plan.targets)})")
            plan_attempts = self._execute_plan(plan)
            attempts.extend(plan_attempts)
            accepted_attempt = self._find_accepted_attempt(plan_attempts)
            if self._should_register_target_early_stop(plan, accepted_attempt):
                accepted_primary_attempts[plan.primary_target] = accepted_attempt
                self.logger.log(
                    "agent_framework",
                    "registered_target_early_stop",
                    target=plan.primary_target,
                    plan_id=plan.plan_id,
                    round_index=accepted_attempt.round_index,
                    confidence=accepted_attempt.validation.confidence if accepted_attempt.validation else 0.0,
                )

        estimates = self.result_inference.infer(spec.targets, attempts, spec.target_to_probe)
        self._fill_unresolved_targets(estimates)
        output_path = self.output_writer.write(
            self.output_path,
            spec,
            plans,
            attempts,
            estimates,
        )
        print(f"Wrote final report to {output_path}")
        return output_path

    def _execute_plan(self, plan) -> list[ProbeAttempt]:
        plan_attempts: list[ProbeAttempt] = []
        feedback = ""

        for round_index in range(1, plan.max_rounds + 1):
            print(f"  [round {round_index}] generating benchmark")
            code, prompt_path, response_path = self.codegen.generate(plan, round_index, feedback)
            source_path = self.executor.write_source(plan.plan_id, round_index, code)
            binary_path, compile_result = self.executor.compile(source_path, plan.plan_id, round_index)

            attempt = ProbeAttempt(
                plan_id=plan.plan_id,
                probe_family=plan.probe_family,
                primary_target=plan.primary_target,
                probe_variant=plan.probe_variant,
                plan_role=plan.plan_role,
                round_index=round_index,
                generated_source_path=str(source_path),
                binary_path=str(binary_path),
                generation_feedback=feedback,
                llm_prompt_path=str(prompt_path),
                llm_response_path=str(response_path),
                compile_result=compile_result,
            )

            if not compile_result.ok:
                attempt.validation = ValidationResult(
                    credible=False,
                    confidence=0.0,
                    issues=["Benchmark compilation failed."],
                    cross_checks=[],
                )
                plan_attempts.append(attempt)
                feedback = self.retry_handler.build_retry(plan, "compile", compile_result).feedback
                continue

            print(f"  [round {round_index}] running benchmark")
            run_result = self.executor.run(binary_path, plan.plan_id, round_index, plan.program_args)
            attempt.run_result = run_result
            if not run_result.ok:
                attempt.validation = ValidationResult(
                    credible=False,
                    confidence=0.0,
                    issues=["Benchmark execution failed."],
                    cross_checks=[],
                )
                plan_attempts.append(attempt)
                feedback = self.retry_handler.build_retry(plan, "run", run_result).feedback
                continue

            attempt.benchmark_output = parse_program_output(run_result.stdout)
            if not attempt.benchmark_output:
                attempt.validation = ValidationResult(
                    credible=False,
                    confidence=0.0,
                    issues=["Benchmark output could not be parsed as JSON."],
                    cross_checks=[],
                )
                plan_attempts.append(attempt)
                feedback = self.retry_handler.build_retry(plan, "parse", run_result).feedback
                continue

            print(f"  [round {round_index}] profiling with ncu")
            kernel_name_filter = self._build_profile_kernel_filter(
                attempt.benchmark_output.get("kernel_name")
            )
            launch_skip = self._derive_profile_launch_skip(plan, attempt.benchmark_output)
            self.logger.log(
                "ncu_profiler",
                "resolved_profile_controls",
                plan_id=plan.plan_id,
                round_index=round_index,
                kernel_name_filter=kernel_name_filter,
                launch_skip=launch_skip,
                launch_count=plan.profile_launch_count,
                profile_env=plan.profile_env,
                profile_timeout_s=plan.profile_timeout_s,
            )
            profile_result = self.profiler.profile(
                binary_path=binary_path,
                plan_id=plan.plan_id,
                round_index=round_index,
                metrics=plan.ncu_metrics,
                program_args=plan.program_args,
                timeout_s=plan.profile_timeout_s,
                env_overrides=plan.profile_env,
                kernel_name_filter=kernel_name_filter,
                launch_count=plan.profile_launch_count,
                launch_skip=launch_skip,
            )
            attempt.profile_result = profile_result
            if profile_result.ok:
                metrics, rows, units = parse_ncu_csv(profile_result.stdout)
                attempt.ncu_metrics = metrics
                attempt.ncu_rows = rows
                attempt.ncu_metric_units = units

            attempt.validation = self.validator.validate(plan, attempt)
            plan_attempts.append(attempt)

            if attempt.validation.credible:
                print(
                    f"  [round {round_index}] accepted with confidence {attempt.validation.confidence:.2f}"
                )
                break

            directive = self.decision.decide(plan, attempt)
            if not directive.should_retry:
                break
            feedback = directive.feedback

        return plan_attempts

    def _fill_unresolved_targets(self, estimates: list[MetricEstimate]) -> None:
        for estimate in estimates:
            if estimate.value is None:
                estimate.reasoning += " The current output remains null to make later manual inspection easier."

    @staticmethod
    def _find_accepted_attempt(plan_attempts: list[ProbeAttempt]) -> ProbeAttempt | None:
        return next(
            (
                attempt
                for attempt in reversed(plan_attempts)
                if attempt.validation and attempt.validation.credible
            ),
            None,
        )

    @staticmethod
    def _is_usable_observation(value: object) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        return False

    @classmethod
    def _attempt_resolved_target(cls, target: str, attempt: ProbeAttempt | None) -> bool:
        if attempt is None:
            return False
        if cls._is_usable_observation(attempt.ncu_metrics.get(target)):
            return True
        derived = attempt.benchmark_output.get("derived_metrics", {})
        if not isinstance(derived, dict):
            return False
        return any(
            cls._is_usable_observation(derived.get(alias))
            for alias in DERIVED_ALIASES.get(target, [])
        )

    @classmethod
    def _should_register_target_early_stop(
        cls,
        plan,
        accepted_attempt: ProbeAttempt | None,
    ) -> bool:
        if plan.plan_role != "primary" or not plan.primary_target:
            return False
        if accepted_attempt is None or not accepted_attempt.validation:
            return False
        if accepted_attempt.validation.confidence < EARLY_STOP_PRIMARY_CONFIDENCE:
            return False
        return cls._attempt_resolved_target(plan.primary_target, accepted_attempt)

    @staticmethod
    def _should_skip_plan_via_early_stop(
        plan,
        accepted_primary_attempts: dict[str, ProbeAttempt],
    ) -> str:
        if plan.plan_role != "cross_check" or not plan.primary_target:
            return ""
        prior_attempt = accepted_primary_attempts.get(plan.primary_target)
        if prior_attempt is None or not prior_attempt.validation:
            return ""
        return (
            f"Primary plan `{prior_attempt.plan_id}` already resolved target "
            f"`{plan.primary_target}` with confidence "
            f"{prior_attempt.validation.confidence:.2f}, so this cross-check plan was skipped."
        )

    @staticmethod
    def _build_profile_kernel_filter(kernel_name_hint: object) -> str:
        if not isinstance(kernel_name_hint, str):
            return ""
        tokens = []
        for raw_token in kernel_name_hint.split("|"):
            token = raw_token.strip()
            if not token:
                continue
            token = token.split("(", 1)[0].strip()
            if token:
                tokens.append(re.escape(token))
        if not tokens:
            return ""
        if len(tokens) == 1:
            return f".*{tokens[0]}.*"
        return f".*({'|'.join(tokens)}).*"

    @staticmethod
    def _derive_profile_launch_skip(plan, benchmark_output: dict[str, object]) -> int:
        if not isinstance(benchmark_output, dict):
            return 0
        parameters = benchmark_output.get("parameters", {})
        if not isinstance(parameters, dict):
            return 0
        if "MLAGENT_PROFILE_MAX_WARMUP" in plan.profile_env:
            warmup = GPUProbeAgent._coerce_nonnegative_int(
                plan.profile_env.get("MLAGENT_PROFILE_MAX_WARMUP")
            )
        else:
            warmup = GPUProbeAgent._coerce_nonnegative_int(
                parameters.get("warmup")
                or parameters.get("warmup_iters")
                or parameters.get("warmups")
            )
        if warmup <= 0:
            return 0
        kernel_name_hint = benchmark_output.get("kernel_name")
        kernels_per_iteration = 1
        if isinstance(kernel_name_hint, str):
            kernels_per_iteration = max(
                1,
                len([part for part in kernel_name_hint.split("|") if part.strip()]),
            )
        if plan.probe_family == "bandwidth_probe":
            kernels_per_iteration = max(2, kernels_per_iteration)
        return warmup * kernels_per_iteration

    @staticmethod
    def _coerce_nonnegative_int(value: object) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, float):
            return max(0, int(value))
        if isinstance(value, str):
            try:
                return max(0, int(float(value)))
            except ValueError:
                return 0
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPT-5.4 GPU probing agent")
    parser.add_argument("--target-spec", default="/target/target_spec.json")
    parser.add_argument("--output", default="/workspace/output.md")
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    parser.add_argument("--max-rounds-per-plan", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    agent = GPUProbeAgent(
        workspace_root=Path(args.workspace_root),
        output_path=Path(args.output),
        target_spec_path=Path(args.target_spec),
        max_rounds_per_plan=args.max_rounds_per_plan,
    )
    agent.run()


if __name__ == "__main__":
    main()
