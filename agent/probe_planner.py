from __future__ import annotations

import re

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


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "target"


FAMILY_BLUEPRINTS: dict[str, dict[str, object]] = {
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


class TargetProbePlanningModule:
    def __init__(self, logger: ReasoningLogger):
        self.logger = logger

    def build_target_plans(self, spec: TargetSpec) -> list[BenchmarkPlan]:
        plans: list[BenchmarkPlan] = []
        for plan_index, target in enumerate(spec.targets, start=1):
            probe_family = spec.target_to_probe[target]
            variants = self._variants_for_target(target, probe_family)
            for variant_index, variant in enumerate(variants, start=1):
                plans.append(
                    self._build_plan(
                        target=target,
                        probe_family=probe_family,
                        variant=variant,
                        plan_index=plan_index,
                        variant_index=variant_index,
                    )
                )

        self.logger.log("target_probe_planner", "built_target_probe_plans", plans=plans)
        return plans

    def _build_plan(
        self,
        *,
        target: str,
        probe_family: str,
        variant: dict[str, object],
        plan_index: int,
        variant_index: int,
    ) -> BenchmarkPlan:
        blueprint = FAMILY_BLUEPRINTS[probe_family]
        variant_name = str(variant["probe_variant"])
        role = str(variant["plan_role"])
        objective = f"{blueprint['benchmark_objective']} {variant['objective_suffix']}"
        requirements = _unique(
            list(blueprint["benchmark_requirements"]) + list(variant.get("requirements", []))
        )
        success_criteria = _unique(
            list(blueprint["success_criteria"]) + list(variant.get("success_criteria", []))
        )
        parser_expectations = _unique(
            list(blueprint["parser_expectations"]) + list(variant.get("parser_expectations", []))
        )
        ncu_metrics = _unique(
            list(blueprint["ncu_metrics"])
            + list(variant.get("ncu_metrics", []))
            + ([target] if "__" in target else [])
        )
        max_rounds = int(variant.get("max_rounds", blueprint["max_rounds"]))
        profile_env = self._profile_env_for(probe_family, role, variant_name)
        profile_timeout_s = self._profile_timeout_for(probe_family, role)
        profile_launch_count = self._profile_launch_count_for(probe_family)

        target_slug = _slugify(target)
        variant_slug = _slugify(variant_name)
        plan_id = f"{plan_index:02d}_{variant_index:02d}_{probe_family}_{target_slug}_{variant_slug}"

        return BenchmarkPlan(
            plan_id=plan_id,
            probe_family=probe_family,
            targets=[target],
            benchmark_objective=objective,
            benchmark_requirements=requirements,
            ncu_metrics=ncu_metrics,
            success_criteria=success_criteria,
            parser_expectations=parser_expectations,
            primary_target=target,
            probe_variant=variant_name,
            plan_role=role,
            max_rounds=max_rounds,
            tags=[probe_family, target, variant_name, role],
            profile_timeout_s=profile_timeout_s,
            profile_launch_count=profile_launch_count,
            profile_env=profile_env,
        )

    @staticmethod
    def _profile_env_for(probe_family: str, plan_role: str, probe_variant: str) -> dict[str, str]:
        del probe_variant
        family_scale = {
            "frequency_probe": "0.0625",
            "bandwidth_probe": "0.125",
            "latency_probe": "0.25",
            "bank_conflict_probe": "0.25",
            "cache_capacity_probe": "0.25",
            "exploratory_probe": "0.25",
        }.get(probe_family, "0.25")
        target_kernel_ms = {
            "frequency_probe": "400",
            "bandwidth_probe": "500",
            "latency_probe": "250",
            "bank_conflict_probe": "250",
            "cache_capacity_probe": "300",
            "exploratory_probe": "300",
        }.get(probe_family, "300")
        sweep_limit = "2" if plan_role == "cross_check" else "3"
        return {
            "MLAGENT_PROFILE_MODE": "1",
            "MLAGENT_PROFILE_SCALE": family_scale,
            "MLAGENT_PROFILE_MAX_WARMUP": "1",
            "MLAGENT_PROFILE_MAX_REPEATS": "2",
            "MLAGENT_PROFILE_SWEEP_LIMIT": sweep_limit,
            "MLAGENT_PROFILE_TARGET_KERNEL_MS": target_kernel_ms,
        }

    @staticmethod
    def _profile_timeout_for(probe_family: str, plan_role: str) -> int:
        if probe_family in {"frequency_probe", "bandwidth_probe"} and plan_role == "cross_check":
            return 480
        if probe_family in {"frequency_probe", "bandwidth_probe"}:
            return 420
        return 300

    @staticmethod
    def _profile_launch_count_for(probe_family: str) -> int:
        if probe_family == "bandwidth_probe":
            return 4
        if probe_family == "frequency_probe":
            return 2
        return 1

    def _variants_for_target(self, target: str, probe_family: str) -> list[dict[str, object]]:
        target_variants = {
            "launch__sm_count": [
                self._variant(
                    probe_variant="occupancy_saturate",
                    plan_role="primary",
                    objective_suffix=(
                        "Focus this probe on revealing the active SM count with a single dominant saturated kernel."
                    ),
                    requirements=[
                        "Use a very large grid and keep per-block resource usage modest so all SMs can participate.",
                        "Favor one dominant kernel launch and avoid helper kernels that would clutter launch-level profiling.",
                        "Report launch geometry in JSON so launch-shape stability can be audited.",
                    ],
                    ncu_metrics=[
                        "launch__sm_count",
                        "launch__grid_size",
                        "launch__block_size",
                        "launch__waves_per_multiprocessor",
                    ],
                    success_criteria=[
                        "launch__sm_count is stable across repeated runs",
                        "the kernel launch shape is large enough to occupy all SMs",
                    ],
                    parser_expectations=["parameters.blocks", "parameters.threads_per_block"],
                ),
                self._variant(
                    probe_variant="launch_shape_sweep",
                    plan_role="cross_check",
                    objective_suffix=(
                        "Use a launch-shape sweep to confirm the inferred SM count remains stable under multiple large-grid configurations."
                    ),
                    requirements=[
                        "Sweep at least two large launch shapes that should all exceed the SM count comfortably.",
                        "Emit the tested launch shapes in JSON for later inspection.",
                    ],
                    ncu_metrics=["launch__sm_count", "launch__waves_per_multiprocessor"],
                    success_criteria=[
                        "launch__sm_count remains consistent across the launch-shape sweep"
                    ],
                    parser_expectations=["sweep_points"],
                ),
            ],
            "device__attribute_max_gpu_frequency_khz": [
                self._variant(
                    probe_variant="sustained_fma_frequency",
                    plan_role="primary",
                    objective_suffix=(
                        "Focus this probe on holding a stable compute-bound FMA-heavy interval so the GPU frequency can be observed cleanly."
                    ),
                    requirements=[
                        "Use a long-running FMA-dominant kernel with minimal DRAM traffic.",
                        "Emit any benchmark-side frequency estimate only as cross-check evidence, not as the main output.",
                    ],
                    ncu_metrics=[
                        "device__attribute_max_gpu_frequency_khz",
                        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
                    ],
                    success_criteria=[
                        "frequency readings are stable across repeated measurements",
                        "the probe is clearly compute-bound rather than memory-bound",
                    ],
                ),
                self._variant(
                    probe_variant="intensity_sweep_frequency",
                    plan_role="cross_check",
                    objective_suffix=(
                        "Sweep arithmetic intensity to confirm the observed GPU frequency stays credible under nearby compute-heavy settings."
                    ),
                    requirements=[
                        "Vary arithmetic intensity or inner loop depth while keeping the probe compute-bound.",
                        "Record the tested intensity points in JSON.",
                    ],
                    ncu_metrics=[
                        "device__attribute_max_gpu_frequency_khz",
                        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                    ],
                    success_criteria=[
                        "the frequency remains stable across neighboring compute-heavy sweep points"
                    ],
                    parser_expectations=["sweep_points"],
                ),
            ],
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": [
                self._variant(
                    probe_variant="compute_saturation",
                    plan_role="primary",
                    objective_suffix=(
                        "Focus this probe on measuring sustained SM throughput under a clearly compute-bound kernel."
                    ),
                    requirements=[
                        "Keep arithmetic intensity high and memory traffic low.",
                        "Favor one dominant kernel and avoid fragmented helper work.",
                    ],
                    ncu_metrics=[
                        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
                    ],
                    success_criteria=[
                        "sm__throughput reaches a high sustained level with low timing variance"
                    ],
                ),
                self._variant(
                    probe_variant="occupancy_compute_sweep",
                    plan_role="cross_check",
                    objective_suffix=(
                        "Sweep launch occupancy and loop depth to confirm the SM throughput estimate is not an artifact of one configuration."
                    ),
                    requirements=[
                        "Test multiple occupancy-friendly launch shapes and report them in JSON.",
                    ],
                    ncu_metrics=[
                        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                        "launch__waves_per_multiprocessor",
                    ],
                    success_criteria=[
                        "high SM throughput remains reproducible across nearby configurations"
                    ],
                    parser_expectations=["sweep_points"],
                ),
            ],
            "dram__bytes_read.sum.per_second": [
                self._variant(
                    probe_variant="read_stream_primary",
                    plan_role="primary",
                    objective_suffix=(
                        "Focus this probe on a clean read-dominant DRAM streaming kernel to measure read bandwidth directly."
                    ),
                    requirements=[
                        "Make the dominant phase overwhelmingly read-heavy with minimal writeback traffic.",
                        "Keep the working set well beyond cache residency.",
                    ],
                    ncu_metrics=[
                        "dram__bytes_read.sum.per_second",
                        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                    ],
                    success_criteria=[
                        "ncu read bandwidth and benchmark-side read bandwidth agree closely"
                    ],
                ),
                self._variant(
                    probe_variant="read_working_set_sweep",
                    plan_role="cross_check",
                    objective_suffix=(
                        "Sweep the read working-set size to confirm the final read-bandwidth value comes from a truly DRAM-bound regime."
                    ),
                    requirements=[
                        "Emit the tested working-set sizes and measured bandwidths in JSON.",
                    ],
                    ncu_metrics=["dram__bytes_read.sum.per_second", "dram__throughput.avg.pct_of_peak_sustained_elapsed"],
                    success_criteria=[
                        "read bandwidth stabilizes only after the working set exceeds cache capacity"
                    ],
                    parser_expectations=["sweep_points"],
                ),
            ],
            "dram__bytes_write.sum.per_second": [
                self._variant(
                    probe_variant="write_stream_primary",
                    plan_role="primary",
                    objective_suffix=(
                        "Focus this probe on a clean write-dominant DRAM streaming kernel to measure write bandwidth directly."
                    ),
                    requirements=[
                        "Make the dominant phase overwhelmingly write-heavy with minimal read-modify-write overhead.",
                        "Keep the working set well beyond cache residency.",
                    ],
                    ncu_metrics=[
                        "dram__bytes_write.sum.per_second",
                        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                    ],
                    success_criteria=[
                        "ncu write bandwidth and benchmark-side write bandwidth agree closely"
                    ],
                ),
                self._variant(
                    probe_variant="write_working_set_sweep",
                    plan_role="cross_check",
                    objective_suffix=(
                        "Sweep the write working-set size to confirm the final write-bandwidth value comes from a truly DRAM-bound regime."
                    ),
                    requirements=[
                        "Emit the tested working-set sizes and measured bandwidths in JSON.",
                    ],
                    ncu_metrics=["dram__bytes_write.sum.per_second", "dram__throughput.avg.pct_of_peak_sustained_elapsed"],
                    success_criteria=[
                        "write bandwidth stabilizes only after the working set exceeds cache capacity"
                    ],
                    parser_expectations=["sweep_points"],
                ),
            ],
            "device__attribute_max_mem_frequency_khz": [
                self._variant(
                    probe_variant="memory_clock_under_saturation",
                    plan_role="primary",
                    objective_suffix=(
                        "Focus this probe on a sustained DRAM-saturated interval so the memory frequency attribute is observed under a trustworthy memory-bound condition."
                    ),
                    requirements=[
                        "Use a strong DRAM streaming workload rather than a formula-based estimate.",
                        "Keep the kernel long enough for stable memory-side profiling.",
                    ],
                    ncu_metrics=[
                        "device__attribute_max_mem_frequency_khz",
                        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                    ],
                    success_criteria=[
                        "memory throughput is high enough that the memory frequency reading is trustworthy"
                    ],
                ),
                self._variant(
                    probe_variant="memory_clock_intensity_sweep",
                    plan_role="cross_check",
                    objective_suffix=(
                        "Sweep memory intensity to confirm the observed memory frequency remains stable across nearby DRAM-heavy settings."
                    ),
                    requirements=[
                        "Vary the amount of concurrent memory traffic while staying memory-bound.",
                        "Report the tested intensity settings in JSON.",
                    ],
                    ncu_metrics=[
                        "device__attribute_max_mem_frequency_khz",
                        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                    ],
                    success_criteria=[
                        "memory frequency remains stable across nearby memory-heavy sweep points"
                    ],
                    parser_expectations=["sweep_points"],
                ),
            ],
            "device__attribute_fb_bus_width": [
                self._variant(
                    probe_variant="bus_width_streaming_observation",
                    plan_role="primary",
                    objective_suffix=(
                        "Focus this probe on stable full-width streaming traffic so the frame-buffer bus-width attribute is observed under realistic memory pressure."
                    ),
                    requirements=[
                        "Use wide, coalesced accesses and keep the workload memory-bound.",
                        "Do not hardcode any bus-width value in the benchmark logic.",
                    ],
                    ncu_metrics=[
                        "device__attribute_fb_bus_width",
                        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                    ],
                    success_criteria=[
                        "the bus-width observation is collected while DRAM throughput is near saturation"
                    ],
                ),
                self._variant(
                    probe_variant="transaction_width_sweep",
                    plan_role="cross_check",
                    objective_suffix=(
                        "Sweep transaction width and vector width to confirm the bus-width observation is robust under multiple coalesced streaming shapes."
                    ),
                    requirements=[
                        "Test at least two transaction or vector-width settings and report them in JSON.",
                    ],
                    ncu_metrics=["device__attribute_fb_bus_width", "dram__throughput.avg.pct_of_peak_sustained_elapsed"],
                    success_criteria=[
                        "the bus-width observation remains stable across nearby coalesced streaming shapes"
                    ],
                    parser_expectations=["sweep_points"],
                ),
            ],
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": [
                self._variant(
                    probe_variant="memory_throughput_primary",
                    plan_role="primary",
                    objective_suffix=(
                        "Focus this probe on pushing sustained memory throughput close to peak with a clearly memory-bound workload."
                    ),
                    requirements=[
                        "Keep arithmetic intensity very low and maximize coalesced DRAM traffic.",
                        "Separate read-heavy and write-heavy phases if both are present.",
                    ],
                    ncu_metrics=[
                        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                    ],
                    success_criteria=[
                        "memory throughput reaches a high sustained percentage of peak"
                    ],
                ),
                self._variant(
                    probe_variant="coalescing_saturation_sweep",
                    plan_role="cross_check",
                    objective_suffix=(
                        "Sweep coalescing and working-set settings to confirm the selected memory-throughput value comes from the best saturated regime."
                    ),
                    requirements=[
                        "Emit the tested coalescing or stride settings in JSON.",
                    ],
                    ncu_metrics=[
                        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                    ],
                    success_criteria=[
                        "the best memory-throughput point is clearly separated from under-driven sweep points"
                    ],
                    parser_expectations=["sweep_points"],
                ),
            ],
        }
        if target in target_variants:
            return target_variants[target]
        return self._generic_variants_for_family(probe_family)

    def _generic_variants_for_family(self, probe_family: str) -> list[dict[str, object]]:
        if probe_family == "latency_probe":
            return [
                self._variant(
                    probe_variant="latency_primary",
                    plan_role="primary",
                    objective_suffix="Focus this probe on isolating the target latency signal with a dependent access chain.",
                    requirements=["Keep the access chain fully dependent and emit the raw sweep series."],
                    parser_expectations=["sweep_points"],
                ),
                self._variant(
                    probe_variant="latency_sweep_cross_check",
                    plan_role="cross_check",
                    objective_suffix="Add a denser working-set sweep to confirm the observed latency cliffs.",
                    requirements=["Zoom in around the suspected latency cliff region."],
                    parser_expectations=["sweep_points"],
                ),
            ]
        if probe_family == "bank_conflict_probe":
            return [
                self._variant(
                    probe_variant="bank_conflict_primary",
                    plan_role="primary",
                    objective_suffix="Focus this probe on a clean conflict-free versus conflicted comparison.",
                    requirements=["Keep the baseline and conflicted kernels structurally comparable."],
                ),
                self._variant(
                    probe_variant="bank_conflict_stride_sweep",
                    plan_role="cross_check",
                    objective_suffix="Sweep shared-memory strides to locate the strongest and weakest conflict patterns.",
                    requirements=["Emit stride sweep results in JSON."],
                    parser_expectations=["sweep_points"],
                ),
            ]
        if probe_family == "cache_capacity_probe":
            return [
                self._variant(
                    probe_variant="capacity_primary",
                    plan_role="primary",
                    objective_suffix="Focus this probe on locating the main cache-capacity cliff with a broad sweep.",
                    requirements=["Cover a wide enough working-set range to bracket the capacity cliff."],
                    parser_expectations=["sweep_points"],
                ),
                self._variant(
                    probe_variant="capacity_zoom_in",
                    plan_role="cross_check",
                    objective_suffix="Zoom in near the suspected cliff to refine the capacity estimate.",
                    requirements=["Use a denser sweep near the suspected capacity boundary."],
                    parser_expectations=["sweep_points"],
                ),
            ]
        return [
            self._variant(
                probe_variant="exploratory_primary",
                plan_role="primary",
                objective_suffix="Focus this probe on a stable target-specific measurement path for the requested metric.",
                requirements=["Emit enough JSON context for later traceability."],
            ),
            self._variant(
                probe_variant="exploratory_cross_check",
                plan_role="cross_check",
                objective_suffix="Add a nearby stress or sweep variation to cross-check the primary exploratory probe.",
                requirements=["Emit the cross-check sweep points in JSON."],
                parser_expectations=["sweep_points"],
            ),
        ]

    @staticmethod
    def _variant(
        *,
        probe_variant: str,
        plan_role: str,
        objective_suffix: str,
        requirements: list[str] | None = None,
        ncu_metrics: list[str] | None = None,
        success_criteria: list[str] | None = None,
        parser_expectations: list[str] | None = None,
        max_rounds: int | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "probe_variant": probe_variant,
            "plan_role": plan_role,
            "objective_suffix": objective_suffix,
            "requirements": requirements or [],
            "ncu_metrics": ncu_metrics or [],
            "success_criteria": success_criteria or [],
            "parser_expectations": parser_expectations or [],
        }
        if max_rounds is not None:
            payload["max_rounds"] = max_rounds
        return payload
