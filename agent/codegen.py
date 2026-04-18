from __future__ import annotations

import json
from pathlib import Path

from llm.openai_client import GPTClient

from .models import BenchmarkPlan
from .reasoning import ReasoningLogger


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT_FILE = PROMPT_DIR / "generate_probe_system.txt"
USER_PROMPT_FILE = PROMPT_DIR / "generate_probe.txt"


class MicroBenchmarkGenerationModule:
    def __init__(self, llm: GPTClient, run_dir: Path, logger: ReasoningLogger):
        self.llm = llm
        self.logger = logger
        self.raw_llm_dir = run_dir / "llm"
        self.raw_llm_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, plan: BenchmarkPlan, round_index: int, feedback: str = "") -> tuple[str, Path, Path]:
        system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
        prompt_template = USER_PROMPT_FILE.read_text(encoding="utf-8")
        prompt = prompt_template.format(
            plan_json=json.dumps(plan.to_prompt_context(), ensure_ascii=False, indent=2),
            round_index=round_index,
            feedback=feedback.strip() or "No previous feedback.",
            target_specific_guidance=self._build_target_specific_guidance(plan),
        )
        prompt_path = self.raw_llm_dir / f"{plan.plan_id}_round{round_index}.prompt.txt"
        response_path = self.raw_llm_dir / f"{plan.plan_id}_round{round_index}.response.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        response = self.llm.complete_text(
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_output_tokens=5000,
        )

        response_path.write_text(response.text, encoding="utf-8")
        code = self._strip_code_fence(response.text)
        self.logger.log(
            "micro_benchmark_generator",
            "generated_benchmark_source",
            plan_id=plan.plan_id,
            round_index=round_index,
            api_mode=response.api_mode,
            prompt_path=prompt_path,
            response_path=response_path,
        )
        return code, prompt_path, response_path

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return stripped

    @staticmethod
    def _build_target_specific_guidance(plan: BenchmarkPlan) -> str:
        target_guidance = {
            "launch__sm_count": (
                "- `launch__sm_count` is a direct `ncu` target. The benchmark should not guess the SM count by itself; "
                "its job is to provide one stable, sufficiently large dominant kernel so `ncu` can observe launch and SM-related metrics cleanly."
            ),
            "dram__bytes_read.sum.per_second": (
                "- `dram__bytes_read.sum.per_second` needs a clearly read-dominant DRAM streaming phase. "
                "The working set must be well beyond L2 capacity and the access pattern should be as coalesced as possible."
            ),
            "dram__bytes_write.sum.per_second": (
                "- `dram__bytes_write.sum.per_second` needs a clearly write-dominant DRAM streaming phase. "
                "Do not mix heavy read-modify-write behavior into the same dominant phase, or `ncu` will have trouble separating read and write throughput."
            ),
            "device__attribute_max_gpu_frequency_khz": (
                "- `device__attribute_max_gpu_frequency_khz` is a direct `ncu` metric. Generate a sufficiently long-lived, clearly compute-bound, FMA-heavy kernel so the frequency and `sm__throughput` readings are trustworthy."
            ),
            "device__attribute_max_mem_frequency_khz": (
                "- `device__attribute_max_mem_frequency_khz` is a direct `ncu` metric. The benchmark should create a stable DRAM-intensive condition rather than trying to infer memory frequency from formulas alone."
            ),
            "device__attribute_fb_bus_width": (
                "- `device__attribute_fb_bus_width` should also be treated as a direct `ncu` or device-side observation. Do not hardcode bus width values and do not rely on spec-sheet knowledge."
            ),
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": (
                "- `sm__throughput.avg.pct_of_peak_sustained_elapsed` needs high arithmetic intensity, low DRAM interference, and enough blocks so the compute utilization observed by `ncu` is meaningful."
            ),
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": (
                "- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` needs a truly memory-bound kernel. "
                "Prefer a large working set, low arithmetic intensity, and sustained streaming accesses."
            ),
        }

        family_guidance = {
            "frequency_probe": [
                "- For the real frequency and compute targets in this project, the main goal is to create a sustained compute workload with one dominant kernel.",
                "- This workload should minimize interference from helper kernels, initialization kernels, or very short-lived kernels that clutter `ncu` output.",
                "- If you emit `sm_clock_khz_estimate`, treat it as a cross-check rather than the primary source of truth.",
            ],
            "bandwidth_probe": [
                "- For the real bandwidth and memory targets in this project, the main goal is to create a stable DRAM streaming workload.",
                "- Prefer two clean read-heavy and write-heavy phases or two separate kernels, and report separate cross-check bandwidth values in JSON.",
                "- For `device__attribute_max_mem_frequency_khz` and `device__attribute_fb_bus_width`, the benchmark only needs to create a trustworthy profiling condition.",
            ],
        }

        lines = [
            "Real target guidance for this plan:",
            *family_guidance.get(plan.probe_family, []),
        ]
        for target in plan.targets:
            guidance = target_guidance.get(target)
            if guidance:
                lines.append(guidance)
        return "\n".join(lines)
