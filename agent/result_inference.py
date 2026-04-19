from __future__ import annotations

import math
from typing import Any

from .models import MetricEstimate, ProbeAttempt
from .reasoning import ReasoningLogger


DERIVED_ALIASES: dict[str, list[str]] = {
    "dram__bytes_read.sum.per_second": ["read_bytes_per_second"],
    "dram__bytes_write.sum.per_second": ["write_bytes_per_second"],
    "device__attribute_max_gpu_frequency_khz": ["sm_clock_khz_estimate"],
    "device__attribute_max_mem_frequency_khz": ["mem_clock_khz_estimate"],
    "dram_latency_cycles": ["dram_latency_cycles", "latency_cycles"],
    "l2_latency_cycles": ["l2_latency_cycles"],
    "l1_latency_cycles": ["l1_latency_cycles"],
    "bank_conflict_penalty_cycles": ["bank_conflict_penalty_cycles"],
    "l2_cache_capacity_bytes": ["l2_capacity_bytes", "cache_capacity_bytes"],
}


class ResultParsingAndInferenceModule:
    def __init__(self, logger: ReasoningLogger):
        self.logger = logger

    def infer(
        self,
        targets: list[str],
        attempts: list[ProbeAttempt],
        target_to_probe: dict[str, str] | None = None,
    ) -> list[MetricEstimate]:
        estimates = [
            self._infer_one(
                target,
                attempts,
                preferred_probe_family=(target_to_probe or {}).get(target),
            )
            for target in targets
        ]
        self.logger.log("result_inference", "built_metric_estimates", estimates=estimates)
        return estimates

    def _infer_one(
        self,
        target: str,
        attempts: list[ProbeAttempt],
        preferred_probe_family: str | None = None,
    ) -> MetricEstimate:
        candidates: list[dict[str, Any]] = []
        for attempt in attempts:
            ncu_value = attempt.ncu_metrics.get(target)
            if self._is_usable_candidate(ncu_value):
                candidates.append(
                    self._build_candidate(
                        target=target,
                        attempt=attempt,
                        value=ncu_value,
                        source_kind="ncu",
                        source=f"ncu:{attempt.plan_id}:round{attempt.round_index}",
                        preferred_probe_family=preferred_probe_family,
                    )
                )

            derived = attempt.benchmark_output.get("derived_metrics", {})
            for alias in DERIVED_ALIASES.get(target, []):
                derived_value = derived.get(alias)
                if self._is_usable_candidate(derived_value):
                    candidates.append(
                        self._build_candidate(
                            target=target,
                            attempt=attempt,
                            value=derived_value,
                            source_kind="benchmark",
                            source=f"benchmark:{attempt.plan_id}:round{attempt.round_index}:{alias}",
                            preferred_probe_family=preferred_probe_family,
                        )
                    )

        if not candidates:
            return MetricEstimate(
                target=target,
                value=None,
                confidence=0.0,
                source="unresolved",
                reasoning="No usable benchmark or ncu evidence was found, so the target could not be resolved.",
                selection_rule="No selection rule could be applied because no usable evidence was found.",
                evidence=[],
            )

        candidates.sort(key=self._candidate_sort_key)
        best = candidates[0]
        best_value = best["value"]
        source = best["source"]
        attempt = best["attempt"]
        normalized_value = self._normalize_value(target, best_value)

        evidence: list[str] = []
        if attempt.validation:
            evidence.extend(attempt.validation.cross_checks[:2])
            evidence.extend(attempt.validation.supporting_evidence[:2])
            evidence.extend(attempt.validation.issues[:1])
        reasoning = self._build_reasoning(
            target,
            normalized_value,
            source,
            attempt,
            selection_rule=best["selection_rule"],
        )
        return MetricEstimate(
            target=target,
            value=normalized_value,
            confidence=max(0.0, min(1.0, best["score"])),
            source=source,
            reasoning=reasoning,
            selection_rule=best["selection_rule"],
            evidence=[item for item in evidence if item],
        )

    def _build_candidate(
        self,
        *,
        target: str,
        attempt: ProbeAttempt,
        value: Any,
        source_kind: str,
        source: str,
        preferred_probe_family: str | None,
    ) -> dict[str, Any]:
        validation_confidence = attempt.validation.confidence if attempt.validation else 0.0
        target_specific = attempt.primary_target == target
        role_rank = 2 if attempt.plan_role == "primary" else 1 if attempt.plan_role == "cross_check" else 0
        source_bonus = 0.08 if source_kind == "ncu" else 0.03
        specificity_bonus = 0.05 if target_specific else 0.0
        tier = self._candidate_tier(
            target=target,
            attempt=attempt,
            source_kind=source_kind,
            preferred_probe_family=preferred_probe_family,
        )
        return {
            "tier": tier,
            "score": validation_confidence + source_bonus + specificity_bonus,
            "value": value,
            "source": source,
            "attempt": attempt,
            "selection_rule": self._selection_rule_text(tier),
            "sort_role_rank": role_rank,
            "sort_confidence": validation_confidence,
            "sort_round": attempt.round_index,
        }

    @staticmethod
    def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, ...]:
        return (
            candidate["tier"],
            -candidate["sort_role_rank"],
            -candidate["sort_confidence"],
            -candidate["sort_round"],
            -candidate["score"],
        )

    @staticmethod
    def _candidate_tier(
        *,
        target: str,
        attempt: ProbeAttempt,
        source_kind: str,
        preferred_probe_family: str | None,
    ) -> int:
        target_specific = attempt.primary_target == target
        accepted = bool(attempt.validation and attempt.validation.credible)
        preferred_family = preferred_probe_family is None or attempt.probe_family == preferred_probe_family
        direct_ncu = source_kind == "ncu"

        if target_specific and accepted and direct_ncu:
            return 0
        if target_specific and accepted and not direct_ncu:
            return 1
        if preferred_family and accepted and direct_ncu:
            return 2
        if preferred_family and accepted and not direct_ncu:
            return 3
        if target_specific and direct_ncu:
            return 4
        if target_specific and not direct_ncu:
            return 5
        if preferred_family and direct_ncu:
            return 6
        if preferred_family and not direct_ncu:
            return 7
        if accepted and direct_ncu:
            return 8
        if accepted and not direct_ncu:
            return 9
        if direct_ncu:
            return 10
        return 11

    @staticmethod
    def _selection_rule_text(tier: int) -> str:
        if tier == 0:
            return "Rule 1: use the target's own accepted direct ncu measurement."
        if tier == 1:
            return "Rule 2: use the target's own accepted benchmark-derived value when no accepted direct ncu value was selected."
        if tier == 2:
            return "Rule 3: fall back to an accepted shared probe in the same probe family because the target's own accepted rounds did not resolve this metric."
        if tier == 3:
            return "Rule 4: fall back to an accepted same-family benchmark-derived value because no accepted direct measurement resolved this metric."
        if tier in {4, 5, 6, 7}:
            return "Rule 5: no accepted round resolved this metric, so a lower-confidence same-family fallback was used."
        return "Rule 6: no same-family accepted evidence resolved this metric, so a last-resort fallback from other evidence was used."

    @staticmethod
    def _normalize_value(target: str, value: Any) -> Any:
        if not isinstance(value, (int, float)):
            return value
        if any(token in target for token in ("count", "khz", "width", "cycles", "capacity_bytes")):
            return int(round(value))
        if target.endswith("per_second"):
            return round(float(value), 3)
        return round(float(value), 4)

    @staticmethod
    def _build_reasoning(
        target: str,
        value: Any,
        source: str,
        attempt: ProbeAttempt,
        *,
        selection_rule: str,
    ) -> str:
        probe = f"{attempt.probe_family} round {attempt.round_index}"
        if source.startswith("ncu:"):
            return f"{target} used the direct ncu metric from {probe}; the selected final value is {value}. {selection_rule}"
        return f"{target} used a benchmark-derived metric from {probe}; the selected final value is {value}. {selection_rule}"

    @staticmethod
    def _is_usable_candidate(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        return False
