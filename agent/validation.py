from __future__ import annotations

import math
import statistics
from typing import Any

from .consistency import (
    DRAM_READ_TARGET,
    DRAM_THROUGHPUT_TARGET,
    DRAM_WRITE_TARGET,
    FB_BUS_WIDTH_TARGET,
    GPU_MEMORY_THROUGHPUT_TARGET,
    MEMORY_FREQ_TARGET,
    assess_memory_target_observation,
    coerce_float,
    implied_dram_peak_pct,
    select_ncu_observation,
)
from .models import BenchmarkPlan, ProbeAttempt, ValidationResult
from .reasoning import ReasoningLogger


class CrossValidationModule:
    def __init__(self, logger: ReasoningLogger):
        self.logger = logger

    def validate(self, plan: BenchmarkPlan, attempt: ProbeAttempt) -> ValidationResult:
        issues: list[str] = []
        cross_checks: list[str] = []
        supporting_evidence: list[str] = []
        confidence = 0.0

        if not attempt.compile_result or not attempt.compile_result.ok:
            issues.append("Benchmark compilation failed, so the result cannot be considered credible.")
            return self._finalize(plan, attempt, issues, cross_checks, supporting_evidence, confidence)

        if not attempt.run_result or not attempt.run_result.ok:
            issues.append("Benchmark execution failed, so the result cannot be considered credible.")
            return self._finalize(plan, attempt, issues, cross_checks, supporting_evidence, confidence)

        if attempt.benchmark_output:
            confidence += 0.2
        else:
            issues.append("No parseable JSON result was found in benchmark stdout.")

        timings = self._extract_timings(attempt.benchmark_output)
        if timings:
            mean_value = statistics.fmean(timings)
            if mean_value > 0:
                confidence += 0.15
                if len(timings) >= 2:
                    variation = statistics.pstdev(timings) / mean_value
                    if variation <= 0.10:
                        confidence += 0.15
                        supporting_evidence.append(
                            f"The coefficient of variation for timings_ms is about {variation:.3f}, indicating good repeatability."
                        )
                    else:
                        issues.append(
                            f"timings_ms shows high variance, with a coefficient of variation of about {variation:.3f}."
                        )
            else:
                issues.append("The mean of timings_ms is not greater than 0.")
        else:
            issues.append("The benchmark JSON did not provide timings_ms.")

        if attempt.profile_result and attempt.profile_result.ok and attempt.ncu_metrics:
            confidence += 0.2
        else:
            issues.append("ncu profiling was missing, failed, or did not produce parseable metrics.")

        if plan.probe_family == "bandwidth_probe":
            confidence += self._validate_bandwidth(attempt, issues, cross_checks, supporting_evidence)
        elif plan.probe_family == "frequency_probe":
            confidence += self._validate_frequency(attempt, issues, cross_checks, supporting_evidence)
        elif plan.probe_family == "latency_probe":
            confidence += self._validate_latency(attempt, issues, supporting_evidence)
        elif plan.probe_family == "bank_conflict_probe":
            confidence += self._validate_bank_conflict(attempt, issues, cross_checks, supporting_evidence)
        elif plan.probe_family == "cache_capacity_probe":
            confidence += self._validate_cache_capacity(attempt, issues, supporting_evidence)

        return self._finalize(plan, attempt, issues, cross_checks, supporting_evidence, confidence)

    def _finalize(
        self,
        plan: BenchmarkPlan,
        attempt: ProbeAttempt,
        issues: list[str],
        cross_checks: list[str],
        supporting_evidence: list[str],
        confidence: float,
    ) -> ValidationResult:
        normalized_confidence = max(0.0, min(1.0, confidence))
        blocking_markers = ("failed", "not suitable", "physical consistency check")
        credible = normalized_confidence >= 0.6 and not any(
            marker in issue.lower() for issue in issues for marker in blocking_markers
        )
        result = ValidationResult(
            credible=credible,
            confidence=normalized_confidence,
            issues=issues,
            cross_checks=cross_checks,
            supporting_evidence=supporting_evidence,
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
        supporting_evidence: list[str],
    ) -> float:
        confidence = 0.0
        target = attempt.primary_target
        observation = select_ncu_observation(attempt, target)
        derived = attempt.benchmark_output.get("derived_metrics", {})
        read_bench = coerce_float(derived.get("read_bytes_per_second"))
        write_bench = coerce_float(derived.get("write_bytes_per_second"))
        read_ncu = coerce_float(observation.get(DRAM_READ_TARGET))
        write_ncu = coerce_float(observation.get(DRAM_WRITE_TARGET))

        deltas: list[float] = []
        if target == DRAM_READ_TARGET and read_bench and read_ncu:
            deltas.append(abs(read_bench - read_ncu) / max(abs(read_ncu), 1.0))
        elif target == DRAM_WRITE_TARGET and write_bench and write_ncu:
            deltas.append(abs(write_bench - write_ncu) / max(abs(write_ncu), 1.0))
        else:
            if read_bench and read_ncu:
                deltas.append(abs(read_bench - read_ncu) / max(abs(read_ncu), 1.0))
            if write_bench and write_ncu:
                deltas.append(abs(write_bench - write_ncu) / max(abs(write_ncu), 1.0))

        if deltas:
            worst_delta = max(deltas)
            if worst_delta <= 0.25:
                confidence += 0.2
                cross_checks.append(
                    f"The maximum benchmark-versus-ncu read/write bandwidth gap is about {worst_delta:.2%}, so bandwidth cross-validation passed."
                )
            else:
                issues.append(
                    f"The benchmark-versus-ncu bandwidth gap is about {worst_delta:.2%}; cross-validation was not established, so this can only be used as mismatch diagnostics."
                )
        else:
            issues.append("The bandwidth probe is missing benchmark-side or ncu-side read/write bandwidth observations for comparison.")

        suitable, suitability_issues = assess_memory_target_observation(target, attempt, observation)
        issues.extend(suitability_issues)

        dram_throughput = coerce_float(observation.get(DRAM_THROUGHPUT_TARGET))
        gpu_mem_throughput = coerce_float(observation.get(GPU_MEMORY_THROUGHPUT_TARGET))
        implied_pct = implied_dram_peak_pct(observation)

        if target in {
            DRAM_READ_TARGET,
            DRAM_WRITE_TARGET,
            MEMORY_FREQ_TARGET,
            FB_BUS_WIDTH_TARGET,
        }:
            if dram_throughput is not None and suitable:
                confidence += 0.1
                supporting_evidence.append(
                    f"DRAM throughput reached {dram_throughput:.2f}% of peak in the selected row, indicating that the probe applied real DRAM pressure."
                )
            elif dram_throughput is not None:
                issues.append(
                    f"DRAM throughput reached only {dram_throughput:.2f}% of peak in the selected row, so this probe did not establish a trustworthy DRAM-bound condition for the target."
                )
        elif target == GPU_MEMORY_THROUGHPUT_TARGET:
            if gpu_mem_throughput is not None and gpu_mem_throughput >= 30:
                confidence += 0.1
                supporting_evidence.append(
                    f"Compute-memory throughput reached {gpu_mem_throughput:.2f}% of peak in the selected row."
                )
            elif gpu_mem_throughput is not None:
                issues.append(
                    f"Compute-memory throughput reached only {gpu_mem_throughput:.2f}% of peak, so the probe may not have been sufficiently memory-bound for this target."
                )

        if implied_pct is not None and dram_throughput is not None:
            supporting_evidence.append(
                f"The selected row's DRAM byte rate implies about {implied_pct:.2f}% of peak, versus `dram__throughput` at {dram_throughput:.2f}%."
            )
        return confidence

    def _validate_frequency(
        self,
        attempt: ProbeAttempt,
        issues: list[str],
        cross_checks: list[str],
        supporting_evidence: list[str],
    ) -> float:
        confidence = 0.0
        derived = attempt.benchmark_output.get("derived_metrics", {})
        freq_bench = coerce_float(derived.get("sm_clock_khz_estimate"))
        freq_ncu = attempt.ncu_metrics.get("device__attribute_max_gpu_frequency_khz")

        if freq_ncu:
            confidence += 0.15
            supporting_evidence.append(
                f"ncu returned device__attribute_max_gpu_frequency_khz={freq_ncu:.0f}."
            )

        if freq_bench and freq_ncu:
            delta = abs(freq_bench - freq_ncu) / max(abs(freq_ncu), 1.0)
            if delta <= 0.20:
                confidence += 0.15
                cross_checks.append(
                    f"The benchmark-estimated frequency differs from ncu by about {delta:.2%}, so frequency cross-validation passed."
                )
            else:
                issues.append(
                    f"The benchmark-estimated frequency differs from ncu by about {delta:.2%}, so cross-validation was not established."
                )

        sm_throughput = attempt.ncu_metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")
        if sm_throughput is not None and sm_throughput >= 40:
            confidence += 0.10
            supporting_evidence.append(
                f"sm__throughput reached {sm_throughput:.2f}% of peak, indicating that the compute probe was sufficiently aggressive."
            )
        elif sm_throughput is not None:
            issues.append(
                f"sm__throughput reached only {sm_throughput:.2f}% of peak, so the compute probe may still be limited by another bottleneck."
            )
        return confidence

    def _validate_latency(
        self,
        attempt: ProbeAttempt,
        issues: list[str],
        supporting_evidence: list[str],
    ) -> float:
        derived = attempt.benchmark_output.get("derived_metrics", {})
        keys = [key for key in derived if key.endswith("latency_cycles")]
        if keys:
            supporting_evidence.append(
                f"The benchmark produced latency metrics: {', '.join(sorted(keys))}."
            )
            return 0.2
        issues.append("The latency probe did not output any *latency_cycles metrics.")
        return 0.0

    def _validate_bank_conflict(
        self,
        attempt: ProbeAttempt,
        issues: list[str],
        cross_checks: list[str],
        supporting_evidence: list[str],
    ) -> float:
        derived = attempt.benchmark_output.get("derived_metrics", {})
        penalty = coerce_float(derived.get("bank_conflict_penalty_cycles"))
        conflict_metric = attempt.ncu_metrics.get("l1tex__data_bank_conflicts_pipe_lsu.sum")
        confidence = 0.0
        if penalty and penalty > 0:
            confidence += 0.1
            supporting_evidence.append(f"bank_conflict_penalty_cycles={penalty:.2f}.")
        else:
            issues.append("The bank-conflict probe did not observe a positive conflict penalty.")
        if conflict_metric and conflict_metric > 0:
            confidence += 0.1
            if penalty and penalty > 0:
                cross_checks.append(
                    "Both a benchmark-side conflict penalty and an ncu bank-conflict count were observed, so shared-memory conflict cross-validation passed."
                )
            else:
                supporting_evidence.append(
                    f"ncu observed bank conflicts with a count of about {conflict_metric:.2f}."
                )
            return confidence
        issues.append("ncu did not observe a bank-conflict metric.")
        return confidence

    def _validate_cache_capacity(
        self,
        attempt: ProbeAttempt,
        issues: list[str],
        supporting_evidence: list[str],
    ) -> float:
        derived = attempt.benchmark_output.get("derived_metrics", {})
        capacity = coerce_float(
            derived.get("l2_capacity_bytes") or derived.get("cache_capacity_bytes")
        )
        sweep_points = attempt.benchmark_output.get("sweep_points", [])
        confidence = 0.0
        if capacity and capacity > 0:
            confidence += 0.1
            supporting_evidence.append(f"The estimated cache capacity is about {capacity:.0f} bytes.")
        else:
            issues.append("The cache-capacity probe did not produce a positive capacity estimate.")
        if isinstance(sweep_points, list) and len(sweep_points) >= 4:
            confidence += 0.1
            supporting_evidence.append(f"The benchmark emitted {len(sweep_points)} sweep points.")
            return confidence
        issues.append("The cache-capacity probe did not provide enough sweep points.")
        return confidence
