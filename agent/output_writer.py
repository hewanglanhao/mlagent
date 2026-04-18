from __future__ import annotations

import json
from pathlib import Path

from llm.openai_client import GPTClient

from .models import BenchmarkPlan, MetricEstimate, ProbeAttempt, TargetSpec
from .reasoning import ReasoningLogger


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT_FILE = PROMPT_DIR / "summarize_output_system.txt"
USER_PROMPT_FILE = PROMPT_DIR / "summarize_output.txt"


class OutputGenerationModule:
    def __init__(self, llm: GPTClient, logger: ReasoningLogger):
        self.llm = llm
        self.logger = logger

    def write(
        self,
        output_path: Path,
        spec: TargetSpec,
        plans: list[BenchmarkPlan],
        attempts: list[ProbeAttempt],
        estimates: list[MetricEstimate],
    ) -> Path:
        result_json = {estimate.target: estimate.value for estimate in estimates}
        analysis_lines = [
            f"- `{estimate.target}`: `{self._render_value(estimate.value)}`。{estimate.reasoning}"
            + (f" 证据: {'; '.join(estimate.evidence)}。" if estimate.evidence else "")
            for estimate in estimates
        ]
        trace_digest = self._build_trace_digest(spec, plans, attempts)
        llm_summary = self._build_llm_summary(spec, estimates, trace_digest)

        content = "\n".join(
            [
                "# Results",
                "```json",
                json.dumps(result_json, ensure_ascii=False, indent=2),
                "```",
                "",
                "# Metric Analysis",
                *analysis_lines,
                "",
                "# Inference Trace",
                llm_summary,
                "",
                "# Raw Trace Digest",
                trace_digest,
            ]
        ).strip() + "\n"

        output_path.write_text(content, encoding="utf-8")
        self.logger.log("output_writer", "wrote_output", output_path=output_path)
        return output_path

    def _build_llm_summary(
        self,
        spec: TargetSpec,
        estimates: list[MetricEstimate],
        trace_digest: str,
    ) -> str:
        system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
        prompt_template = USER_PROMPT_FILE.read_text(encoding="utf-8")
        prompt = prompt_template.format(
            targets=json.dumps(spec.targets, ensure_ascii=False),
            metric_analyses=json.dumps(
                [
                    {
                        "target": estimate.target,
                        "value": estimate.value,
                        "confidence": estimate.confidence,
                        "source": estimate.source,
                        "reasoning": estimate.reasoning,
                        "evidence": estimate.evidence,
                    }
                    for estimate in estimates
                ],
                ensure_ascii=False,
                indent=2,
            ),
            trace_digest=trace_digest,
            target_summary_guidance=self._build_target_summary_guidance(spec),
        )

        try:
            response = self.llm.complete_text(
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_output_tokens=1500,
            )
            return response.text.strip()
        except Exception as exc:
            return (
                "LLM 总结生成失败，因此使用程序化摘要替代。"
                f" 目标共 {len(spec.targets)} 个，当前输出了 {len(estimates)} 个指标槽位。"
                f" 失败原因: {exc}"
            )

    @staticmethod
    def _build_trace_digest(
        spec: TargetSpec,
        plans: list[BenchmarkPlan],
        attempts: list[ProbeAttempt],
    ) -> str:
        lines = [
            f"Target spec: {', '.join(spec.targets)}",
            f"Target spec source: {spec.source_path}",
            "Plans:",
        ]
        for plan in plans:
            lines.append(
                f"- {plan.plan_id}: {plan.probe_family}, targets={', '.join(plan.targets)}, max_rounds={plan.max_rounds}"
            )

        lines.append("Attempts:")
        for attempt in attempts:
            validation = attempt.validation
            validation_text = (
                f"credible={validation.credible}, confidence={validation.confidence:.2f}"
                if validation
                else "credible=unknown"
            )
            lines.append(
                f"- {attempt.plan_id} round {attempt.round_index}: compile={attempt.compile_result.returncode if attempt.compile_result else 'n/a'}, "
                f"run={attempt.run_result.returncode if attempt.run_result else 'n/a'}, "
                f"ncu={attempt.profile_result.returncode if attempt.profile_result else 'n/a'}, {validation_text}"
            )
            if validation and validation.issues:
                lines.append(f"  issues: {' | '.join(validation.issues)}")
        return "\n".join(lines)

    @staticmethod
    def _render_value(value: object) -> str:
        if value is None:
            return "null"
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _build_target_summary_guidance(spec: TargetSpec) -> str:
        direct_ncu_targets = {
            "launch__sm_count",
            "dram__bytes_read.sum.per_second",
            "dram__bytes_write.sum.per_second",
            "device__attribute_max_gpu_frequency_khz",
            "device__attribute_max_mem_frequency_khz",
            "device__attribute_fb_bus_width",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        }
        lines = [
            "Use the following interpretation rules for the real target set in this project:",
            "- Most of these metrics are direct `ncu` metrics. The benchmark's main role is to create a trustworthy profiling condition and provide cross-check evidence.",
            "- Do not claim that the benchmark directly measured `launch__sm_count`, `device__attribute_max_gpu_frequency_khz`, `device__attribute_max_mem_frequency_khz`, or `device__attribute_fb_bus_width` unless the evidence explicitly shows a benchmark-side derivation.",
            "- The purpose of `frequency_probe` is to create a compute-bound condition so the `ncu` values for `launch__sm_count`, `device__attribute_max_gpu_frequency_khz`, and `sm__throughput.avg.pct_of_peak_sustained_elapsed` are credible.",
            "- The purpose of `bandwidth_probe` is to create a memory-bound condition so the `ncu` values for `dram__bytes_read.sum.per_second`, `dram__bytes_write.sum.per_second`, `device__attribute_max_mem_frequency_khz`, `device__attribute_fb_bus_width`, and `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` are credible.",
            "- If a final metric value mainly comes from `ncu`, say explicitly that it was taken directly from `ncu`. If it only comes from benchmark cross-checking, describe it as cross-validation evidence.",
            "- When a round is rejected as untrustworthy, prefer explanations such as kernel too short, working set too small, compute or memory not saturated, read and write interference, or high timing variance.",
            "Target list for this run:",
        ]
        for target in spec.targets:
            source_hint = "direct ncu" if target in direct_ncu_targets else "benchmark/derived"
            lines.append(f"- {target}: {source_hint}")
        return "\n".join(lines)
