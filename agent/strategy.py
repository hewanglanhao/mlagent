from __future__ import annotations

from collections import defaultdict

from .models import BenchmarkPlan, TargetSpec
from .reasoning import ReasoningLogger


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


BLUEPRINTS: dict[str, dict[str, object]] = {
    "latency_probe": {
        "benchmark_objective": (
            "Generate a pointer-chasing latency micro-benchmark that measures the memory hierarchy with dependent loads and a working-set sweep."
        ),
        "benchmark_requirements": [
            "Use a dependent pointer-chasing access pattern so the hardware cannot overlap or prefetch the loads away.",
            "Sweep at least three working-set regions so L1/L2/DRAM latency cliffs can be observed.",
            "Report latency in cycles and include the raw sweep points in the JSON output.",
        ],
        "ncu_metrics": [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ],
        "success_criteria": [
            "timings_ms has repeated measurements with low variance",
            "derived_metrics contains at least one latency_cycles metric",
            "working-set sweep is visible in the benchmark JSON output",
        ],
        "parser_expectations": [
            "timings_ms",
            "derived_metrics.l1_latency_cycles",
            "derived_metrics.l2_latency_cycles",
            "derived_metrics.dram_latency_cycles",
            "sweep_points",
        ],
        "max_rounds": 3,
    },
    "bandwidth_probe": {
        "benchmark_objective": (
            "Generate a DRAM streaming bandwidth benchmark with separate read and write stress kernels and enough work to sustain peak memory throughput."
        ),
        "benchmark_requirements": [
            "Use a working set that clearly exceeds the expected L2 cache size.",
            "Include separate read-dominant and write-dominant kernels or phases.",
            "Report read_bytes_per_second and write_bytes_per_second in derived_metrics.",
            "Keep the kernels simple and long enough for ncu to measure stable throughput.",
        ],
        "ncu_metrics": [
            "dram__bytes_read.sum.per_second",
            "dram__bytes_write.sum.per_second",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "device__attribute_max_mem_frequency_khz",
            "device__attribute_fb_bus_width",
            "launch__sm_count",
        ],
        "success_criteria": [
            "benchmark and ncu bandwidth numbers are within roughly 25%",
            "memory throughput is high enough that the probe is not clearly under-driving DRAM",
            "timings_ms shows stable repeated measurements",
        ],
        "parser_expectations": [
            "timings_ms",
            "derived_metrics.read_bytes_per_second",
            "derived_metrics.write_bytes_per_second",
        ],
        "max_rounds": 3,
    },
    "bank_conflict_probe": {
        "benchmark_objective": (
            "Generate a shared-memory bank-conflict benchmark that compares a conflict-free access pattern against one or more intentionally conflicted patterns."
        ),
        "benchmark_requirements": [
            "Use the same kernel structure for both conflict-free and conflicted cases so the delta is attributable to bank conflicts.",
            "Report bank_conflict_penalty_cycles in derived_metrics.",
            "Emit the per-case timing results in the JSON output.",
        ],
        "ncu_metrics": [
            "l1tex__data_bank_conflicts_pipe_lsu.sum",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ],
        "success_criteria": [
            "conflicted access is measurably slower than the conflict-free baseline",
            "ncu reports bank conflicts for the conflicted case",
        ],
        "parser_expectations": [
            "timings_ms",
            "derived_metrics.bank_conflict_penalty_cycles",
        ],
        "max_rounds": 2,
    },
    "cache_capacity_probe": {
        "benchmark_objective": (
            "Generate a cache-capacity benchmark that sweeps the working-set size and identifies the latency cliff for the target cache level."
        ),
        "benchmark_requirements": [
            "Use enough sweep points to detect a clear knee in the latency curve.",
            "Include the full sweep series in the JSON output.",
            "Report l2_capacity_bytes or cache_capacity_bytes in derived_metrics when a cliff is found.",
        ],
        "ncu_metrics": [
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "lts__t_sectors_srcunit_tex_op_read.sum",
        ],
        "success_criteria": [
            "latency curve shows a stable cliff rather than random noise",
            "derived_metrics includes an inferred capacity",
        ],
        "parser_expectations": [
            "timings_ms",
            "sweep_points",
            "derived_metrics.l2_capacity_bytes",
        ],
        "max_rounds": 3,
    },
    "frequency_probe": {
        "benchmark_objective": (
            "Generate a compute-heavy frequency probe that drives sustained SM activity with a long-running FMA-dominant kernel and exposes direct ncu metrics."
        ),
        "benchmark_requirements": [
            "Use a compute-heavy kernel with enough arithmetic intensity to avoid being DRAM-bound.",
            "Keep the runtime long enough for ncu to capture a stable active frequency window.",
            "Estimate sm_clock_khz_estimate in derived_metrics if possible from benchmark timing.",
            "Make sure the kernel launch is large enough to occupy most SMs.",
        ],
        "ncu_metrics": [
            "launch__sm_count",
            "device__attribute_max_gpu_frequency_khz",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
        ],
        "success_criteria": [
            "SM throughput is high enough that the probe is meaningfully compute-bound",
            "timings_ms is stable across repeated measurements",
            "ncu returns the requested device and launch metrics",
        ],
        "parser_expectations": [
            "timings_ms",
            "derived_metrics.sm_clock_khz_estimate",
        ],
        "max_rounds": 3,
    },
    "exploratory_probe": {
        "benchmark_objective": (
            "Generate a generic exploratory GPU micro-benchmark that surfaces the requested metric through a simple but measurable kernel."
        ),
        "benchmark_requirements": [
            "Favor a self-contained benchmark with a clear measurement loop and JSON output.",
            "If the target metric is direct from ncu, make the kernel simple and stable enough for profiling.",
        ],
        "ncu_metrics": [],
        "success_criteria": [
            "benchmark produces parseable JSON output",
            "at least one requested target receives a usable estimate",
        ],
        "parser_expectations": [
            "timings_ms",
            "derived_metrics",
        ],
        "max_rounds": 2,
    },
}


class BenchmarkStrategySelectionModule:
    def __init__(self, logger: ReasoningLogger):
        self.logger = logger

    def build_plans(self, spec: TargetSpec) -> list[BenchmarkPlan]:
        grouped_targets: dict[str, list[str]] = defaultdict(list)
        for target in spec.targets:
            grouped_targets[spec.target_to_probe[target]].append(target)

        plans: list[BenchmarkPlan] = []
        for index, (probe_family, targets) in enumerate(grouped_targets.items(), start=1):
            blueprint = BLUEPRINTS[probe_family]
            direct_ncu_targets = [target for target in targets if "__" in target]
            ncu_metrics = _unique(list(blueprint["ncu_metrics"]) + direct_ncu_targets)
            plan = BenchmarkPlan(
                plan_id=f"{index:02d}_{probe_family}",
                probe_family=probe_family,
                targets=targets,
                benchmark_objective=str(blueprint["benchmark_objective"]),
                benchmark_requirements=list(blueprint["benchmark_requirements"]),
                ncu_metrics=ncu_metrics,
                success_criteria=list(blueprint["success_criteria"]),
                parser_expectations=list(blueprint["parser_expectations"]),
                max_rounds=int(blueprint["max_rounds"]),
                tags=[probe_family, *targets],
            )
            plans.append(plan)

        self.logger.log("strategy_selector", "built_benchmark_plans", plans=plans)
        return plans
