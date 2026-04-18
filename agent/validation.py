from __future__ import annotations

import math
import statistics
from typing import Any

from .models import BenchmarkPlan, ProbeAttempt, ValidationResult
from .reasoning import ReasoningLogger


class CrossValidationModule:
    def __init__(self, logger: ReasoningLogger):
        self.logger = logger

    def validate(self, plan: BenchmarkPlan, attempt: ProbeAttempt) -> ValidationResult:
        issues: list[str] = []
        cross_checks: list[str] = []
        confidence = 0.0

        if not attempt.compile_result or not attempt.compile_result.ok:
            issues.append("benchmark 编译失败，无法建立可信度。")
            return self._finalize(plan, attempt, issues, cross_checks, confidence)

        if not attempt.run_result or not attempt.run_result.ok:
            issues.append("benchmark 运行失败，无法建立可信度。")
            return self._finalize(plan, attempt, issues, cross_checks, confidence)

        if attempt.benchmark_output:
            confidence += 0.2
        else:
            issues.append("benchmark stdout 中没有解析出 JSON 结果。")

        timings = self._extract_timings(attempt.benchmark_output)
        if timings:
            mean_value = statistics.fmean(timings)
            if mean_value > 0:
                confidence += 0.15
                if len(timings) >= 2:
                    variation = statistics.pstdev(timings) / mean_value
                    if variation <= 0.10:
                        confidence += 0.15
                        cross_checks.append(f"timings_ms 的变异系数约为 {variation:.3f}，重复性较好。")
                    else:
                        issues.append(f"timings_ms 波动较大，变异系数约为 {variation:.3f}。")
            else:
                issues.append("timings_ms 平均值不大于 0。")
        else:
            issues.append("benchmark JSON 未提供 timings_ms。")

        if attempt.profile_result and attempt.profile_result.ok and attempt.ncu_metrics:
            confidence += 0.2
        else:
            issues.append("ncu profiling 缺失、失败，或未产出可解析指标。")

        if plan.probe_family == "bandwidth_probe":
            confidence += self._validate_bandwidth(attempt, issues, cross_checks)
        elif plan.probe_family == "frequency_probe":
            confidence += self._validate_frequency(attempt, issues, cross_checks)
        elif plan.probe_family == "latency_probe":
            confidence += self._validate_latency(attempt, issues, cross_checks)
        elif plan.probe_family == "bank_conflict_probe":
            confidence += self._validate_bank_conflict(attempt, issues, cross_checks)
        elif plan.probe_family == "cache_capacity_probe":
            confidence += self._validate_cache_capacity(attempt, issues, cross_checks)

        return self._finalize(plan, attempt, issues, cross_checks, confidence)

    def _finalize(
        self,
        plan: BenchmarkPlan,
        attempt: ProbeAttempt,
        issues: list[str],
        cross_checks: list[str],
        confidence: float,
    ) -> ValidationResult:
        normalized_confidence = max(0.0, min(1.0, confidence))
        credible = normalized_confidence >= 0.6 and not any("失败" in issue for issue in issues)
        result = ValidationResult(
            credible=credible,
            confidence=normalized_confidence,
            issues=issues,
            cross_checks=cross_checks,
        )
        self.logger.log(
            "cross_validator",
            "validated_attempt",
            plan_id=plan.plan_id,
            round_index=attempt.round_index,
            validation=result,
        )
        return result

    @staticmethod
    def _extract_timings(benchmark_output: dict[str, Any]) -> list[float]:
        timings = benchmark_output.get("timings_ms", [])
        if not isinstance(timings, list):
            return []
        numeric: list[float] = []
        for item in timings:
            if isinstance(item, (int, float)) and math.isfinite(item):
                numeric.append(float(item))
        return numeric

    def _validate_bandwidth(
        self,
        attempt: ProbeAttempt,
        issues: list[str],
        cross_checks: list[str],
    ) -> float:
        confidence = 0.0
        derived = attempt.benchmark_output.get("derived_metrics", {})
        read_bench = self._coerce_float(derived.get("read_bytes_per_second"))
        write_bench = self._coerce_float(derived.get("write_bytes_per_second"))
        read_ncu = attempt.ncu_metrics.get("dram__bytes_read.sum.per_second")
        write_ncu = attempt.ncu_metrics.get("dram__bytes_write.sum.per_second")

        deltas: list[float] = []
        if read_bench and read_ncu:
            deltas.append(abs(read_bench - read_ncu) / max(abs(read_ncu), 1.0))
        if write_bench and write_ncu:
            deltas.append(abs(write_bench - write_ncu) / max(abs(write_ncu), 1.0))

        if deltas:
            worst_delta = max(deltas)
            if worst_delta <= 0.25:
                confidence += 0.2
                cross_checks.append(
                    f"benchmark 与 ncu 的读写带宽最大偏差约为 {worst_delta:.2%}，互相吻合。"
                )
            else:
                issues.append(
                    f"benchmark 与 ncu 的带宽偏差约为 {worst_delta:.2%}，需要进一步放大工作集或延长运行时间。"
                )
        else:
            issues.append("带宽 probe 缺少 benchmark 或 ncu 的读写带宽对照数据。")

        mem_throughput = attempt.ncu_metrics.get(
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
        )
        if mem_throughput is not None and mem_throughput >= 30:
            confidence += 0.1
            cross_checks.append(f"memory throughput 达到 {mem_throughput:.2f}% peak，probe 对 DRAM 有效施压。")
        elif mem_throughput is not None:
            issues.append(f"memory throughput 只有 {mem_throughput:.2f}% peak，probe 可能没有充分压满 DRAM。")
        return confidence

    def _validate_frequency(
        self,
        attempt: ProbeAttempt,
        issues: list[str],
        cross_checks: list[str],
    ) -> float:
        confidence = 0.0
        derived = attempt.benchmark_output.get("derived_metrics", {})
        freq_bench = self._coerce_float(derived.get("sm_clock_khz_estimate"))
        freq_ncu = attempt.ncu_metrics.get("device__attribute_max_gpu_frequency_khz")

        if freq_ncu:
            confidence += 0.15
            cross_checks.append(f"ncu 返回 device__attribute_max_gpu_frequency_khz={freq_ncu:.0f}。")

        if freq_bench and freq_ncu:
            delta = abs(freq_bench - freq_ncu) / max(abs(freq_ncu), 1.0)
            if delta <= 0.20:
                confidence += 0.15
                cross_checks.append(f"benchmark 估计频率与 ncu 的偏差约为 {delta:.2%}。")
            else:
                issues.append(f"benchmark 估计频率与 ncu 偏差约为 {delta:.2%}。")

        sm_throughput = attempt.ncu_metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")
        if sm_throughput is not None and sm_throughput >= 40:
            confidence += 0.10
            cross_checks.append(f"sm__throughput 达到 {sm_throughput:.2f}% peak，compute probe 足够激进。")
        elif sm_throughput is not None:
            issues.append(f"sm__throughput 只有 {sm_throughput:.2f}% peak，compute probe 可能仍受其他瓶颈影响。")
        return confidence

    def _validate_latency(
        self,
        attempt: ProbeAttempt,
        issues: list[str],
        cross_checks: list[str],
    ) -> float:
        derived = attempt.benchmark_output.get("derived_metrics", {})
        keys = [key for key in derived if key.endswith("latency_cycles")]
        if keys:
            cross_checks.append(f"benchmark 产出了延迟指标: {', '.join(sorted(keys))}。")
            return 0.2
        issues.append("latency probe 没有输出任何 *latency_cycles 指标。")
        return 0.0

    def _validate_bank_conflict(
        self,
        attempt: ProbeAttempt,
        issues: list[str],
        cross_checks: list[str],
    ) -> float:
        derived = attempt.benchmark_output.get("derived_metrics", {})
        penalty = self._coerce_float(derived.get("bank_conflict_penalty_cycles"))
        conflict_metric = attempt.ncu_metrics.get("l1tex__data_bank_conflicts_pipe_lsu.sum")
        confidence = 0.0
        if penalty and penalty > 0:
            confidence += 0.1
            cross_checks.append(f"bank_conflict_penalty_cycles={penalty:.2f}。")
        else:
            issues.append("bank conflict probe 没有观测到正的冲突惩罚。")
        if conflict_metric and conflict_metric > 0:
            confidence += 0.1
            cross_checks.append(f"ncu 观测到 bank conflicts，计数约为 {conflict_metric:.2f}。")
            return confidence
        issues.append("ncu 未观测到 bank conflict 指标。")
        return confidence

    def _validate_cache_capacity(
        self,
        attempt: ProbeAttempt,
        issues: list[str],
        cross_checks: list[str],
    ) -> float:
        derived = attempt.benchmark_output.get("derived_metrics", {})
        capacity = self._coerce_float(
            derived.get("l2_capacity_bytes") or derived.get("cache_capacity_bytes")
        )
        sweep_points = attempt.benchmark_output.get("sweep_points", [])
        confidence = 0.0
        if capacity and capacity > 0:
            confidence += 0.1
            cross_checks.append(f"cache capacity 估计值约为 {capacity:.0f} bytes。")
        else:
            issues.append("cache capacity probe 没有给出正的容量估计。")
        if isinstance(sweep_points, list) and len(sweep_points) >= 4:
            confidence += 0.1
            cross_checks.append(f"benchmark 输出了 {len(sweep_points)} 个 sweep 点。")
            return confidence
        issues.append("cache capacity probe 的 sweep 点数量不足。")
        return confidence

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
        return None
