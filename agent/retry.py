from __future__ import annotations

from dataclasses import dataclass

from .models import BenchmarkPlan, CommandResult
from .reasoning import ReasoningLogger


@dataclass
class RetryDirective:
    should_retry: bool
    feedback: str
    reason: str


class ErrorHandlingAndRetryModule:
    def __init__(self, logger: ReasoningLogger):
        self.logger = logger

    def build_retry(self, plan: BenchmarkPlan, stage: str, result: CommandResult | None) -> RetryDirective:
        if result is None:
            directive = RetryDirective(
                should_retry=True,
                feedback=(
                    f"The previous {plan.probe_family} attempt failed before the {stage} stage completed. "
                    "Regenerate the benchmark and ensure the JSON contract is preserved."
                ),
                reason=f"{stage}_missing",
            )
            self.logger.log("retry_handler", "built_retry_directive", plan_id=plan.plan_id, directive=directive)
            return directive

        error_summary = (result.stderr or result.stdout or "").strip()
        stage_hint = {
            "compile": (
                "The previous CUDA code failed to compile. Fix syntax errors, missing headers, undefined symbols, "
                "and host/device type mismatches while preserving the requested probe semantics."
            ),
            "run": (
                "The previous benchmark compiled but failed at runtime. Fix illegal memory accesses, launch "
                "configuration bugs, allocation sizes, or synchronization issues."
            ),
            "parse": (
                "The previous benchmark did not print a valid JSON object. Keep the benchmark logic but make stdout "
                "end with exactly one parseable JSON object that includes timings_ms and derived_metrics."
            ),
            "profile": (
                "The benchmark ran but ncu profiling failed. Keep the kernel simple, deterministic, and friendly to "
                "profiling. Support a lightweight profiling mode that reduces warmup, repeats, sweep breadth, and "
                "loop trip counts when `MLAGENT_PROFILE_MODE=1`, so ncu does not time out while still observing the "
                "same dominant kernel behavior."
            ),
        }.get(
            stage,
            "Regenerate the benchmark while keeping the probe objective unchanged.",
        )

        directive = RetryDirective(
            should_retry=True,
            feedback=f"{stage_hint}\n\nFailure stage: {stage}\nError output:\n{error_summary}",
            reason=f"{stage}_retry",
        )
        self.logger.log("retry_handler", "built_retry_directive", plan_id=plan.plan_id, directive=directive)
        return directive
