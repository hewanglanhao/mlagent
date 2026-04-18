from __future__ import annotations

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

    def infer(self, targets: list[str], attempts: list[ProbeAttempt]) -> list[MetricEstimate]:
        estimates = [self._infer_one(target, attempts) for target in targets]
        self.logger.log("result_inference", "built_metric_estimates", estimates=estimates)
        return estimates

    def _infer_one(self, target: str, attempts: list[ProbeAttempt]) -> MetricEstimate:
        candidates: list[tuple[float, Any, str, ProbeAttempt]] = []
        for attempt in attempts:
            validation_confidence = attempt.validation.confidence if attempt.validation else 0.0

            if target in attempt.ncu_metrics:
                candidates.append(
                    (
                        validation_confidence + 0.25,
                        attempt.ncu_metrics[target],
                        f"ncu:{attempt.plan_id}:round{attempt.round_index}",
                        attempt,
                    )
                )

            derived = attempt.benchmark_output.get("derived_metrics", {})
            for alias in DERIVED_ALIASES.get(target, []):
                if alias in derived:
                    candidates.append(
                        (
                            validation_confidence + 0.10,
                            derived[alias],
                            f"benchmark:{attempt.plan_id}:round{attempt.round_index}:{alias}",
                            attempt,
                        )
                    )

        if not candidates:
            return MetricEstimate(
                target=target,
                value=None,
                confidence=0.0,
                source="unresolved",
                reasoning="没有找到可用的 benchmark 或 ncu 证据，目标值未解析成功。",
                evidence=[],
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_value, source, attempt = candidates[0]
        normalized_value = self._normalize_value(target, best_value)

        evidence = []
        if attempt.validation:
            evidence.extend(attempt.validation.cross_checks[:2])
            evidence.extend(attempt.validation.issues[:1])
        reasoning = self._build_reasoning(target, normalized_value, source, attempt)
        return MetricEstimate(
            target=target,
            value=normalized_value,
            confidence=max(0.0, min(1.0, best_score)),
            source=source,
            reasoning=reasoning,
            evidence=[item for item in evidence if item],
        )

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
    def _build_reasoning(target: str, value: Any, source: str, attempt: ProbeAttempt) -> str:
        probe = f"{attempt.probe_family} 第 {attempt.round_index} 轮"
        if source.startswith("ncu:"):
            return f"{target} 采用 {probe} 的 ncu 直接指标，最终选取数值 {value}。"
        return f"{target} 采用 {probe} 的 benchmark 派生指标，最终选取数值 {value}。"
