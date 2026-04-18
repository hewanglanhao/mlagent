from __future__ import annotations

import argparse
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
from .result_inference import ResultParsingAndInferenceModule
from .retry import ErrorHandlingAndRetryModule
from .spec_reader import TaskReadingModule
from .strategy import BenchmarkStrategySelectionModule
from .validation import CrossValidationModule


DEFAULT_WORKSPACE_ROOT = Path("/workspace")


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
        for plan in plans:
            print(f"[plan] {plan.plan_id} -> {plan.probe_family} ({', '.join(plan.targets)})")
            attempts.extend(self._execute_plan(plan))

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
                    issues=["benchmark 编译失败。"],
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
                    issues=["benchmark 运行失败。"],
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
                    issues=["benchmark 输出无法解析为 JSON。"],
                    cross_checks=[],
                )
                plan_attempts.append(attempt)
                feedback = self.retry_handler.build_retry(plan, "parse", run_result).feedback
                continue

            print(f"  [round {round_index}] profiling with ncu")
            profile_result = self.profiler.profile(
                binary_path=binary_path,
                plan_id=plan.plan_id,
                round_index=round_index,
                metrics=plan.ncu_metrics,
                program_args=plan.program_args,
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
                estimate.reasoning += " 当前输出保留为 null，方便后续人工检查。"


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
