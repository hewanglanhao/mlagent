from __future__ import annotations

import json
from pathlib import Path

from .models import TargetSpec
from .reasoning import ReasoningLogger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_SPEC_PATH = Path("/target/target_spec.json")
SAMPLE_TARGET_SPEC_PATH = PROJECT_ROOT / "target_spec_sample.json"


class TaskReadingModule:
    def __init__(self, logger: ReasoningLogger):
        self.logger = logger

    def read(self, target_spec_path: Path | None = None) -> TargetSpec:
        source_path = target_spec_path or DEFAULT_TARGET_SPEC_PATH
        used_fallback = False

        if not source_path.exists():
            source_path = SAMPLE_TARGET_SPEC_PATH
            used_fallback = True

        with source_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        targets = data.get("targets") or data.get("metrics") or []
        if not isinstance(targets, list) or not all(isinstance(item, str) and item.strip() for item in targets):
            raise ValueError(
                f"Invalid target spec format in {source_path}. Expected a JSON object with a non-empty string list under 'targets' or 'metrics'."
            )

        normalized_targets = [item.strip() for item in targets]
        target_to_probe = {
            target: self._resolve_probe_family(target) for target in normalized_targets
        }

        spec = TargetSpec(
            source_path=str(source_path),
            targets=normalized_targets,
            target_to_probe=target_to_probe,
            used_fallback=used_fallback,
        )
        self.logger.log("task_reader", "loaded_target_spec", spec=spec)
        return spec

    @staticmethod
    def _resolve_probe_family(target: str) -> str:
        lowered = target.lower()
        if "bank" in lowered and "conflict" in lowered:
            return "bank_conflict_probe"
        if "cache" in lowered and ("capacity" in lowered or "size" in lowered):
            return "cache_capacity_probe"
        if "latency" in lowered:
            return "latency_probe"
        if target in {
            "dram__bytes_read.sum.per_second",
            "dram__bytes_write.sum.per_second",
            "device__attribute_max_mem_frequency_khz",
            "device__attribute_fb_bus_width",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        }:
            return "bandwidth_probe"
        if target in {
            "launch__sm_count",
            "device__attribute_max_gpu_frequency_khz",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        }:
            return "frequency_probe"
        if "occupancy" in lowered or "launch__" in lowered:
            return "frequency_probe"
        return "exploratory_probe"
