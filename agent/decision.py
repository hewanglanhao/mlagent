from __future__ import annotations

from .models import BenchmarkPlan, ProbeAttempt
from .reasoning import ReasoningLogger
from .retry import RetryDirective


class ExperimentDecisionAndFeedbackModule:
    def __init__(self, logger: ReasoningLogger):
        self.logger = logger

    def decide(self, plan: BenchmarkPlan, attempt: ProbeAttempt) -> RetryDirective:
        validation = attempt.validation
        if validation is None or validation.credible:
            directive = RetryDirective(
                should_retry=False,
                feedback="",
                reason="accepted",
            )
            self.logger.log("decision_module", "accepted_attempt", plan_id=plan.plan_id, directive=directive)
            return directive

        suggestions: list[str] = [
            "Keep the same JSON output contract and preserve the intended probe family."
        ]

        joined_issues = " ".join(validation.issues)
        if "variance" in joined_issues.lower():
            suggestions.append("Increase the number of repeated measurements and lengthen each measurement window.")
        if plan.probe_family == "bandwidth_probe":
            suggestions.append("Increase the DRAM working set and ensure the access pattern is fully coalesced.")
            suggestions.append("Use enough blocks and loop unrolling so the probe can saturate memory throughput.")
        elif plan.probe_family == "frequency_probe":
            suggestions.append("Increase arithmetic intensity and keep the kernel compute-bound rather than memory-bound.")
            suggestions.append("Launch enough blocks to occupy most SMs for a sustained interval.")
        elif plan.probe_family == "latency_probe":
            suggestions.append("Strengthen the dependent pointer-chasing chain and add more working-set sweep points.")
        elif plan.probe_family == "bank_conflict_probe":
            suggestions.append("Make the conflict-free and conflicted kernels more directly comparable.")
        elif plan.probe_family == "cache_capacity_probe":
            suggestions.append("Densify the working-set sweep around the suspected cache-capacity cliff.")

        suggestions.append("Address these validation issues explicitly:")
        suggestions.extend(validation.issues)

        directive = RetryDirective(
            should_retry=True,
            feedback="\n".join(suggestions),
            reason="validation_retry",
        )
        self.logger.log(
            "decision_module",
            "requested_new_round",
            plan_id=plan.plan_id,
            round_index=attempt.round_index,
            directive=directive,
        )
        return directive
