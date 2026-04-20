from __future__ import annotations

import math
from typing import Any

from .models import ProbeAttempt


DRAM_READ_TARGET = "dram__bytes_read.sum.per_second"
DRAM_WRITE_TARGET = "dram__bytes_write.sum.per_second"
DRAM_THROUGHPUT_TARGET = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
GPU_MEMORY_THROUGHPUT_TARGET = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
MEMORY_FREQ_TARGET = "device__attribute_max_mem_frequency_khz"
FB_BUS_WIDTH_TARGET = "device__attribute_fb_bus_width"

DRAM_DIRECTIONAL_TARGETS = {DRAM_READ_TARGET, DRAM_WRITE_TARGET}
MEMORY_SIDE_TARGETS = {
    DRAM_READ_TARGET,
    DRAM_WRITE_TARGET,
    MEMORY_FREQ_TARGET,
    FB_BUS_WIDTH_TARGET,
    GPU_MEMORY_THROUGHPUT_TARGET,
}


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def select_ncu_observation(attempt: ProbeAttempt, target: str) -> dict[str, Any]:
    rows = [row for row in attempt.ncu_rows if isinstance(row, dict)]
    if not rows:
        return _summary_observation(attempt)

    ranked_rows: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for row in rows:
        value = coerce_float(row.get(target))
        if value is None:
            continue
        ranked_rows.append((_observation_sort_key(target, row), row))

    if not ranked_rows:
        return _summary_observation(attempt)
    ranked_rows.sort(key=lambda item: item[0], reverse=True)
    return ranked_rows[0][1]


def theoretical_peak_dram_bytes_per_second(metrics: dict[str, Any]) -> float | None:
    mem_frequency_khz = coerce_float(metrics.get(MEMORY_FREQ_TARGET))
    bus_width_bits = coerce_float(metrics.get(FB_BUS_WIDTH_TARGET))
    if not mem_frequency_khz or not bus_width_bits:
        return None
    if mem_frequency_khz <= 0 or bus_width_bits <= 0:
        return None
    # CUDA/NCU expose memory clock in kHz. DRAM links are effectively double-pumped,
    # so the theoretical peak uses a 2x data-rate multiplier.
    return mem_frequency_khz * 1000.0 * (bus_width_bits / 8.0) * 2.0


def implied_dram_peak_pct(metrics: dict[str, Any]) -> float | None:
    peak_bytes_per_second = theoretical_peak_dram_bytes_per_second(metrics)
    if not peak_bytes_per_second:
        return None
    read_bytes = coerce_float(metrics.get(DRAM_READ_TARGET)) or 0.0
    write_bytes = coerce_float(metrics.get(DRAM_WRITE_TARGET)) or 0.0
    total_bytes = read_bytes + write_bytes
    if total_bytes <= 0:
        return None
    return total_bytes / peak_bytes_per_second * 100.0


def assess_memory_target_observation(
    target: str,
    attempt: ProbeAttempt,
    observation: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    metrics = observation or select_ncu_observation(attempt, target)
    reasons: list[str] = []

    dram_pct = coerce_float(metrics.get(DRAM_THROUGHPUT_TARGET))
    gpu_mem_pct = coerce_float(metrics.get(GPU_MEMORY_THROUGHPUT_TARGET))
    implied_pct = implied_dram_peak_pct(metrics)
    kernel_name = str(
        metrics.get("kernel_name")
        or attempt.benchmark_output.get("kernel_name")
        or ""
    ).lower()
    read_bytes = coerce_float(metrics.get(DRAM_READ_TARGET)) or 0.0
    write_bytes = coerce_float(metrics.get(DRAM_WRITE_TARGET)) or 0.0

    if implied_pct is not None and dram_pct is not None:
        rel_gap = abs(implied_pct - dram_pct) / max(abs(dram_pct), 1.0)
        if rel_gap > 0.25:
            reasons.append(
                "Physical consistency check failed: the observed DRAM byte rate does not agree with the same row's DRAM-throughput percentage and the row's memory-frequency/bus-width peak."
            )

    if target == DRAM_READ_TARGET:
        if dram_pct is None:
            reasons.append(
                "The selected read-bandwidth row does not include `dram__throughput.avg.pct_of_peak_sustained_elapsed`, so DRAM saturation cannot be verified."
            )
        elif dram_pct < 10.0:
            reasons.append(
                f"The selected read-bandwidth row reached only {dram_pct:.2f}% of peak DRAM throughput, so it is not suitable for resolving sustained DRAM read bandwidth."
            )
        if read_bytes <= max(write_bytes * 2.0, 1.0):
            reasons.append(
                "The selected row is not cleanly read-dominant, so incidental write traffic could be polluting the read-bandwidth target."
            )
        if kernel_name and "write" in kernel_name and "read" not in kernel_name:
            reasons.append(
                "The selected row comes from a write-oriented kernel name, so it is not suitable for the read-bandwidth target."
            )
    elif target == DRAM_WRITE_TARGET:
        if dram_pct is None:
            reasons.append(
                "The selected write-bandwidth row does not include `dram__throughput.avg.pct_of_peak_sustained_elapsed`, so DRAM saturation cannot be verified."
            )
        elif dram_pct < 10.0:
            reasons.append(
                f"The selected write-bandwidth row reached only {dram_pct:.2f}% of peak DRAM throughput, so it is not suitable for resolving sustained DRAM write bandwidth."
            )
        if write_bytes <= max(read_bytes * 2.0, 1.0):
            reasons.append(
                "The selected row is not cleanly write-dominant, so incidental read traffic could be polluting the write-bandwidth target."
            )
        if kernel_name and "read" in kernel_name and "write" not in kernel_name:
            reasons.append(
                "The selected row comes from a read-oriented kernel name, so it is not suitable for the write-bandwidth target."
            )
    elif target in {MEMORY_FREQ_TARGET, FB_BUS_WIDTH_TARGET}:
        if dram_pct is None:
            reasons.append(
                "The selected memory-side observation does not include `dram__throughput.avg.pct_of_peak_sustained_elapsed`, so the memory-bound condition cannot be verified."
            )
        elif dram_pct < 10.0:
            reasons.append(
                f"The selected memory-side observation reached only {dram_pct:.2f}% of peak DRAM throughput, so it is not suitable for observing this memory-side target under real DRAM pressure."
            )
    elif target == GPU_MEMORY_THROUGHPUT_TARGET:
        if gpu_mem_pct is None:
            reasons.append(
                "The selected row does not include `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`."
            )
        elif gpu_mem_pct < 30.0:
            reasons.append(
                f"The selected row reached only {gpu_mem_pct:.2f}% of peak compute-memory throughput, so it is not suitable for this target."
            )

    return (not reasons), reasons


def benchmark_value_is_physically_plausible(
    target: str,
    attempt: ProbeAttempt,
    value: Any,
) -> tuple[bool, str]:
    numeric_value = coerce_float(value)
    if numeric_value is None:
        return False, "The benchmark-derived value is not a finite numeric value."
    if target not in DRAM_DIRECTIONAL_TARGETS:
        return True, ""

    observation = select_ncu_observation(attempt, target)
    peak_bytes_per_second = theoretical_peak_dram_bytes_per_second(observation)
    if peak_bytes_per_second and numeric_value > peak_bytes_per_second * 1.15:
        return (
            False,
            "The benchmark-derived bandwidth exceeds the theoretical DRAM peak implied by the same run's memory frequency and bus width.",
        )
    return True, ""


def _summary_observation(attempt: ProbeAttempt) -> dict[str, Any]:
    summary = dict(attempt.ncu_metrics)
    kernel_name = attempt.benchmark_output.get("kernel_name")
    if kernel_name and "kernel_name" not in summary:
        summary["kernel_name"] = kernel_name
    return summary


def _observation_sort_key(target: str, row: dict[str, Any]) -> tuple[float, ...]:
    read_bytes = coerce_float(row.get(DRAM_READ_TARGET)) or 0.0
    write_bytes = coerce_float(row.get(DRAM_WRITE_TARGET)) or 0.0
    dram_pct = coerce_float(row.get(DRAM_THROUGHPUT_TARGET)) or 0.0
    gpu_mem_pct = coerce_float(row.get(GPU_MEMORY_THROUGHPUT_TARGET)) or 0.0
    target_value = coerce_float(row.get(target)) or 0.0
    kernel_name = str(row.get("kernel_name") or "").lower()

    if target == DRAM_READ_TARGET:
        read_named = 1.0 if "read" in kernel_name else 0.0
        return (
            read_named,
            read_bytes / max(write_bytes, 1.0),
            dram_pct,
            target_value,
        )
    if target == DRAM_WRITE_TARGET:
        write_named = 1.0 if "write" in kernel_name else 0.0
        return (
            write_named,
            write_bytes / max(read_bytes, 1.0),
            dram_pct,
            target_value,
        )
    if target in {MEMORY_FREQ_TARGET, FB_BUS_WIDTH_TARGET}:
        return (dram_pct, gpu_mem_pct, target_value)
    if target == GPU_MEMORY_THROUGHPUT_TARGET:
        return (gpu_mem_pct, dram_pct, target_value)
    return (target_value,)
