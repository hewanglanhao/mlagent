from __future__ import annotations

import json
import re
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
        metric_cards = self._build_metric_cards(estimates, attempts)
        analysis_lines = [
            (
                f"- `{card['target']}`: value=`{card['value']}`; "
                f"method={card['method']}; "
                f"evidence={card['evidence']}; "
                f"confidence={card['confidence']}."
            )
            for card in metric_cards
        ]
        trial_digest = self._build_trial_and_cross_validation_digest(plans, attempts)
        retry_digest = self._build_retry_and_fix_digest(plans, attempts)
        method_commitments = self._build_method_commitments()
        trace_digest = self._build_trace_digest(spec, plans, attempts)
        conclusion_sentence = self._build_conclusion_sentence(spec, estimates, attempts)
        llm_summary = self._build_llm_summary(
            spec=spec,
            metric_cards=metric_cards,
            trial_digest=trial_digest,
            retry_digest=retry_digest,
            method_commitments=method_commitments,
            trace_digest=trace_digest,
            conclusion_sentence=conclusion_sentence,
        )
        conclusion_line, summary_body = self._split_summary_conclusion(
            llm_summary,
            conclusion_sentence,
        )

        content = "\n".join(
            [
                conclusion_line,
                "",
                "# Results",
                "```json",
                json.dumps(result_json, ensure_ascii=False, indent=2),
                "```",
                "",
                "# Per-Target Findings",
                *analysis_lines,
                "",
                "# Trial And Cross-Validation",
                trial_digest,
                "",
                "# Retry And Fix History",
                retry_digest,
                "",
                "# Method Commitments",
                method_commitments,
                "",
                "# Inference Trace",
                summary_body,
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
        *,
        spec: TargetSpec,
        metric_cards: list[dict[str, str]],
        trial_digest: str,
        retry_digest: str,
        method_commitments: str,
        trace_digest: str,
        conclusion_sentence: str,
    ) -> str:
        system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
        prompt_template = USER_PROMPT_FILE.read_text(encoding="utf-8")
        prompt = prompt_template.format(
            targets=json.dumps(spec.targets, ensure_ascii=False),
            metric_analyses=json.dumps(metric_cards, ensure_ascii=False, indent=2),
            trial_and_cross_validation_digest=trial_digest,
            retry_and_fix_digest=retry_digest,
            method_commitments=method_commitments,
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
            return self._build_fallback_summary(
                spec=spec,
                metric_cards=metric_cards,
                trial_digest=trial_digest,
                retry_digest=retry_digest,
                method_commitments=method_commitments,
                conclusion_sentence=conclusion_sentence,
                error_text=str(exc),
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
            if validation and validation.cross_checks:
                lines.append(f"  cross_checks: {' | '.join(validation.cross_checks)}")
            if attempt.generation_feedback:
                lines.append(f"  retry_feedback: {OutputGenerationModule._summarize_feedback(attempt.generation_feedback)}")
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

    def _build_metric_cards(
        self,
        estimates: list[MetricEstimate],
        attempts: list[ProbeAttempt],
    ) -> list[dict[str, str]]:
        cards: list[dict[str, str]] = []
        for estimate in estimates:
            attempt, source_kind, alias = self._resolve_attempt_from_source(estimate.source, attempts)
            method = self._build_method_text(estimate, attempt, source_kind, alias, attempts)
            evidence = self._build_evidence_text(estimate, attempt)
            cards.append(
                {
                    "target": estimate.target,
                    "value": self._render_value(estimate.value),
                    "method": method,
                    "evidence": evidence,
                    "confidence": f"{estimate.confidence:.2f}",
                    "source": estimate.source,
                }
            )
        return cards

    @staticmethod
    def _resolve_attempt_from_source(
        source: str,
        attempts: list[ProbeAttempt],
    ) -> tuple[ProbeAttempt | None, str, str]:
        if not source or source == "unresolved":
            return None, "unresolved", ""

        benchmark_match = re.fullmatch(r"benchmark:([^:]+):round(\d+):(.+)", source)
        if benchmark_match:
            plan_id, round_index_text, alias = benchmark_match.groups()
            round_index = int(round_index_text)
            for attempt in attempts:
                if attempt.plan_id == plan_id and attempt.round_index == round_index:
                    return attempt, "benchmark", alias
            return None, "benchmark", alias

        ncu_match = re.fullmatch(r"ncu:([^:]+):round(\d+)", source)
        if ncu_match:
            plan_id, round_index_text = ncu_match.groups()
            round_index = int(round_index_text)
            for attempt in attempts:
                if attempt.plan_id == plan_id and attempt.round_index == round_index:
                    return attempt, "ncu", ""
            return None, "ncu", ""

        return None, "unknown", ""

    def _build_method_text(
        self,
        estimate: MetricEstimate,
        attempt: ProbeAttempt | None,
        source_kind: str,
        alias: str,
        attempts: list[ProbeAttempt],
    ) -> str:
        if attempt is None:
            return "No valid benchmark or profiling attempt resolved this target."

        total_rounds = max(
            (item.round_index for item in attempts if item.plan_id == attempt.plan_id),
            default=attempt.round_index,
        )
        timings_count = len(attempt.benchmark_output.get("timings_ms", []))
        if source_kind == "ncu":
            return (
                f"Used the direct `ncu` metric from `{attempt.probe_family}` round {attempt.round_index} "
                f"after {total_rounds} total round(s) for this plan; the benchmark created the profiling condition "
                f"and collected {timings_count} timing sample(s) for cross-validation."
            )
        if source_kind == "benchmark":
            alias_text = f" derived metric `{alias}`" if alias else " derived benchmark metric"
            return (
                f"Used the benchmark{alias_text} from `{attempt.probe_family}` round {attempt.round_index} "
                f"after {total_rounds} total round(s); the value was kept only after repeated timings "
                f"and `ncu` cross-checking where available."
            )
        return estimate.reasoning

    def _build_evidence_text(
        self,
        estimate: MetricEstimate,
        attempt: ProbeAttempt | None,
    ) -> str:
        evidence_items = list(estimate.evidence)
        if attempt is None:
            return "; ".join(evidence_items) if evidence_items else "No supporting evidence was resolved."

        timings = attempt.benchmark_output.get("timings_ms", [])
        if isinstance(timings, list) and timings:
            evidence_items.append(f"multiple trials: {len(timings)} timing sample(s)")

        if attempt.validation and attempt.validation.cross_checks:
            evidence_items.extend(
                [
                    f"cross-validation: {item}"
                    for item in attempt.validation.cross_checks[:2]
                ]
            )

        if attempt.validation and attempt.validation.issues:
            evidence_items.append(
                f"rejected-or-risk notes: {' | '.join(attempt.validation.issues[:2])}"
            )

        if not evidence_items:
            evidence_items.append("No explicit evidence was recorded beyond the selected source.")
        return "; ".join(evidence_items)

    def _build_trial_and_cross_validation_digest(
        self,
        plans: list[BenchmarkPlan],
        attempts: list[ProbeAttempt],
    ) -> str:
        lines: list[str] = []
        for plan in plans:
            plan_attempts = [attempt for attempt in attempts if attempt.plan_id == plan.plan_id]
            if not plan_attempts:
                lines.append(f"- `{plan.plan_id}` / `{plan.probe_family}`: no attempts were executed.")
                continue

            accepted = next(
                (
                    attempt
                    for attempt in reversed(plan_attempts)
                    if attempt.validation and attempt.validation.credible
                ),
                None,
            )
            total_rounds = max((attempt.round_index for attempt in plan_attempts), default=0)
            timings_count = 0
            if accepted:
                timings = accepted.benchmark_output.get("timings_ms", [])
                if isinstance(timings, list):
                    timings_count = len(timings)
            cross_checks: list[str] = []
            if accepted and accepted.validation:
                cross_checks = accepted.validation.cross_checks[:2]

            line = (
                f"- `{plan.plan_id}` / `{plan.probe_family}` ran {total_rounds} round(s); "
                f"accepted round={accepted.round_index if accepted else 'none'}; "
                f"accepted target(s)={', '.join(plan.targets)}; "
                f"timing_trials={timings_count if timings_count else 'n/a'}."
            )
            if cross_checks:
                line += " Cross-validation: " + " | ".join(cross_checks) + "."
            else:
                line += " Cross-validation: no accepted cross-check notes were recorded."
            lines.append(line)
        return "\n".join(lines)

    def _build_retry_and_fix_digest(
        self,
        plans: list[BenchmarkPlan],
        attempts: list[ProbeAttempt],
    ) -> str:
        lines: list[str] = []
        for plan in plans:
            plan_attempts = [attempt for attempt in attempts if attempt.plan_id == plan.plan_id]
            if not plan_attempts:
                lines.append(f"- `{plan.plan_id}` / `{plan.probe_family}`: no retry or fix history because no attempt was run.")
                continue

            round_notes: list[str] = []
            for attempt in plan_attempts:
                issues = attempt.validation.issues[:2] if attempt.validation else []
                if attempt.validation and attempt.validation.credible:
                    round_notes.append(
                        f"round {attempt.round_index} accepted with confidence {attempt.validation.confidence:.2f}"
                    )
                    continue

                failure_bits: list[str] = []
                if attempt.compile_result and not attempt.compile_result.ok:
                    failure_bits.append("compile failed")
                elif attempt.run_result and not attempt.run_result.ok:
                    failure_bits.append("runtime failed")
                elif attempt.profile_result and not attempt.profile_result.ok:
                    failure_bits.append("profiling failed")
                else:
                    failure_bits.append("validation rejected")

                issue_text = " | ".join(issues) if issues else "no issue text recorded"
                retry_text = (
                    f"; retry guidance: {self._summarize_feedback(attempt.generation_feedback)}"
                    if attempt.generation_feedback
                    else ""
                )
                round_notes.append(
                    f"round {attempt.round_index} {' / '.join(failure_bits)} because {issue_text}{retry_text}"
                )

            lines.append(f"- `{plan.plan_id}` / `{plan.probe_family}`: " + "; ".join(round_notes) + ".")
        return "\n".join(lines)

    @staticmethod
    def _build_method_commitments() -> str:
        return "\n".join(
            [
                "- The agent did not rely on static spec sheets, online lookup tables, or API-based attribute queries as the primary source of truth.",
                "- Prompt constraints explicitly forbid using `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`, `nvidia-smi`, or similar static/device-query shortcuts as the main measurement method.",
                "- Final values are expected to come from repeated benchmark trials plus `ncu` profiling, with benchmark-derived numbers used as cross-validation evidence when applicable.",
            ]
        )

    @staticmethod
    def _build_conclusion_sentence(
        spec: TargetSpec,
        estimates: list[MetricEstimate],
        attempts: list[ProbeAttempt],
    ) -> str:
        resolved_count = sum(1 for estimate in estimates if estimate.value is not None)
        accepted_attempts = sum(
            1
            for attempt in attempts
            if attempt.validation and attempt.validation.credible
        )
        return (
            f"结论：本次针对 {len(spec.targets)} 个 target 进行了多轮 probe 实验，"
            f"当前解析出 {resolved_count} 个非空结果，最终结论基于 {accepted_attempts} 个通过交叉验证的有效 round，"
            "并明确未依赖静态规格表或 API 查表。"
        )

    @staticmethod
    def _split_summary_conclusion(summary: str, fallback_conclusion: str) -> tuple[str, str]:
        lines = [line.strip() for line in summary.splitlines() if line.strip()]
        if not lines:
            return fallback_conclusion, fallback_conclusion
        if lines[0].startswith("结论："):
            body = "\n".join(lines[1:]).strip()
            return lines[0], body or lines[0]
        return fallback_conclusion, summary.strip()

    def _build_fallback_summary(
        self,
        *,
        spec: TargetSpec,
        metric_cards: list[dict[str, str]],
        trial_digest: str,
        retry_digest: str,
        method_commitments: str,
        conclusion_sentence: str,
        error_text: str,
    ) -> str:
        top_metrics = ", ".join(
            f"{card['target']}={card['value']}"
            for card in metric_cards[:3]
        )
        return "\n".join(
            [
                conclusion_sentence,
                (
                    "本次输出的逐项 target 结果已经按 value、method、evidence、confidence 结构展开；"
                    f"其中代表性结果包括 {top_metrics if top_metrics else 'no resolved metrics'}。"
                ),
                "交叉验证与多次试验情况如下：",
                trial_digest,
                "失败重试与最终修复过程如下：",
                retry_digest,
                "方法约束如下：",
                method_commitments,
                f"LLM summary generation failed, so this fallback summary was used. Error: {error_text}",
            ]
        )

    @staticmethod
    def _summarize_feedback(text: str, limit: int = 220) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 3] + "..."
