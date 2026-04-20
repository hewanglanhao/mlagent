from __future__ import annotations

import json
import re
from pathlib import Path

from llm.openai_client import GPTClient

from .models import BenchmarkPlan, MetricEstimate, ProbeAttempt, TargetSpec
from .reasoning import ReasoningLogger
from .result_inference import DERIVED_ALIASES


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT_FILE = PROMPT_DIR / "summarize_output_system.txt"
USER_PROMPT_FILE = PROMPT_DIR / "summarize_output.txt"
PER_TARGET_SYSTEM_PROMPT_FILE = PROMPT_DIR / "summarize_per_target_system.txt"
PER_TARGET_USER_PROMPT_FILE = PROMPT_DIR / "summarize_per_target.txt"


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
        selection_rules = self._build_final_value_selection_rules()
        metric_cards = self._build_metric_cards(estimates, attempts)
        target_probe_digests = self._build_target_probe_digests(spec.targets, estimates, attempts)
        analysis_body = self._build_llm_per_target_findings(target_probe_digests, estimates, attempts)
        trial_digest = self._build_trial_and_cross_validation_digest(plans, attempts)
        retry_digest = self._build_retry_and_fix_digest(plans, attempts)
        method_commitments = self._build_method_commitments()
        trace_digest = self._build_trace_digest(spec, plans, attempts)
        conclusion_sentence = self._build_conclusion_sentence(spec, estimates, attempts)
        llm_summary = self._build_llm_summary(
            spec=spec,
            selection_rules=selection_rules,
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
                "# Final Value Selection Rules",
                selection_rules,
                "",
                "# Results",
                "```json",
                json.dumps(result_json, ensure_ascii=False, indent=2),
                "```",
                "",
                "# Per-Target Findings",
                analysis_body,
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

    def _build_llm_per_target_findings(
        self,
        target_probe_digests: list[dict[str, object]],
        estimates: list[MetricEstimate],
        attempts: list[ProbeAttempt],
    ) -> str:
        system_prompt = PER_TARGET_SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
        prompt_template = PER_TARGET_USER_PROMPT_FILE.read_text(encoding="utf-8")
        prompt = prompt_template.format(
            target_probe_digests=json.dumps(target_probe_digests, ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.complete_text(
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_output_tokens=2200,
            )
            text = response.text.strip()
            if text:
                return text
        except Exception:
            pass

        return "\n".join(self._build_per_target_findings(estimates, attempts))

    def _build_llm_summary(
        self,
        *,
        spec: TargetSpec,
        selection_rules: str,
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
            final_value_selection_rules=selection_rules,
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
                selection_rules=selection_rules,
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
            plan_line = (
                f"- {plan.plan_id}: {plan.probe_family}, target={plan.primary_target or ', '.join(plan.targets)}, "
                f"variant={plan.probe_variant}, role={plan.plan_role}, max_rounds={plan.max_rounds}"
            )
            if plan.skipped:
                plan_line += ", skipped=True"
            lines.append(plan_line)
            if plan.skipped and plan.skip_reason:
                lines.append(f"  skip_reason: {plan.skip_reason}")

        lines.append("Attempts:")
        for attempt in attempts:
            validation = attempt.validation
            validation_text = (
                f"credible={validation.credible}, confidence={validation.confidence:.2f}"
                if validation
                else "credible=unknown"
            )
            lines.append(
                f"- {attempt.plan_id} round {attempt.round_index}: target={attempt.primary_target}, variant={attempt.probe_variant}, "
                f"role={attempt.plan_role}, compile={attempt.compile_result.returncode if attempt.compile_result else 'n/a'}, "
                f"run={attempt.run_result.returncode if attempt.run_result else 'n/a'}, "
                f"ncu={attempt.profile_result.returncode if attempt.profile_result else 'n/a'}, {validation_text}"
            )
            if validation and validation.issues:
                lines.append(f"  issues: {' | '.join(validation.issues)}")
            if validation and validation.cross_checks:
                lines.append(f"  cross_checks: {' | '.join(validation.cross_checks)}")
            if validation and validation.supporting_evidence:
                lines.append(
                    f"  supporting_evidence: {' | '.join(validation.supporting_evidence)}"
                )
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
            "- For DRAM read/write targets, the final value must come from a directionally clean profiling row of the same direction; incidental opposite-direction traffic from another row must not be promoted into a standalone target value.",
            "- For DRAM-side targets, the selected row must be physically consistent with `dram__throughput.avg.pct_of_peak_sustained_elapsed` and with the memory-frequency/bus-width-implied peak before it can be treated as a final-value candidate.",
            "- Final-value selection must follow the predeclared rule order: prefer the target's own accepted round, and only fall back to shared probes when the target's own accepted rounds did not resolve the metric.",
            "- Only call something `cross-validation` when benchmark-side and `ncu`-side measurements agree within the configured threshold. Stable timings, high throughput, or a returned `ncu` metric are supporting evidence, not cross-validation by themselves.",
            "- If a final metric value mainly comes from `ncu`, say explicitly that it was taken directly from `ncu`. If it only comes from benchmark cross-checking, describe it as cross-validation evidence. If benchmark and `ncu` disagree materially, call it a mismatch or failed consistency check instead.",
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
                    "selection_rule": estimate.selection_rule,
                    "method": method,
                    "evidence": evidence,
                    "confidence": f"{estimate.confidence:.2f}",
                    "source": estimate.source,
                }
            )
        return cards

    def _build_target_probe_digests(
        self,
        targets: list[str],
        estimates: list[MetricEstimate],
        attempts: list[ProbeAttempt],
    ) -> list[dict[str, object]]:
        estimate_by_target = {estimate.target: estimate for estimate in estimates}
        digests: list[dict[str, object]] = []
        for target in targets:
            estimate = estimate_by_target[target]
            selected_attempt, selected_source_kind, selected_alias = self._resolve_attempt_from_source(
                estimate.source,
                attempts,
            )
            related_attempts = [attempt for attempt in attempts if attempt.primary_target == target]
            plan_ids = []
            for attempt in related_attempts:
                if attempt.plan_id not in plan_ids:
                    plan_ids.append(attempt.plan_id)

            probe_runs = [
                self._build_probe_run_digest(
                    target=target,
                    plan_id=plan_id,
                    plan_attempts=[attempt for attempt in related_attempts if attempt.plan_id == plan_id],
                    selected_attempt=selected_attempt,
                )
                for plan_id in plan_ids
            ]
            digests.append(
                {
                    "target": target,
                    "final_value": self._render_value(estimate.value),
                    "final_confidence": f"{estimate.confidence:.2f}",
                    "final_source": estimate.source,
                    "final_selection_rule": estimate.selection_rule,
                    "final_reasoning": estimate.reasoning,
                    "selected_source_kind": selected_source_kind,
                    "selected_alias": selected_alias,
                    "selected_plan_id": selected_attempt.plan_id if selected_attempt else "",
                    "probe_runs": probe_runs,
                }
            )
        return digests

    def _build_probe_run_digest(
        self,
        *,
        target: str,
        plan_id: str,
        plan_attempts: list[ProbeAttempt],
        selected_attempt: ProbeAttempt | None,
    ) -> dict[str, object]:
        reference_attempt = plan_attempts[0]
        accepted_attempt = next(
            (
                attempt
                for attempt in reversed(plan_attempts)
                if attempt.validation and attempt.validation.credible
            ),
            None,
        )
        latest_attempt = plan_attempts[-1]
        best_attempt = accepted_attempt or latest_attempt
        observed = self._extract_target_observation(target, best_attempt)

        return {
            "plan_id": plan_id,
            "probe_family": reference_attempt.probe_family,
            "probe_variant": reference_attempt.probe_variant,
            "plan_role": reference_attempt.plan_role,
            "rounds_run": len(plan_attempts),
            "accepted_round": accepted_attempt.round_index if accepted_attempt else None,
            "selected_for_final_value": bool(selected_attempt and selected_attempt.plan_id == plan_id),
            "observed_value_kind": observed["kind"],
            "observed_value": observed["value"],
            "observed_value_source": observed["source"],
            "attempt_summaries": [
                {
                    "round_index": attempt.round_index,
                    "compile_returncode": attempt.compile_result.returncode if attempt.compile_result else None,
                    "run_returncode": attempt.run_result.returncode if attempt.run_result else None,
                    "ncu_returncode": attempt.profile_result.returncode if attempt.profile_result else None,
                    "credible": attempt.validation.credible if attempt.validation else False,
                    "confidence": (
                        f"{attempt.validation.confidence:.2f}" if attempt.validation else "0.00"
                    ),
                    "issues": attempt.validation.issues if attempt.validation else [],
                    "cross_checks": attempt.validation.cross_checks if attempt.validation else [],
                    "supporting_evidence": (
                        attempt.validation.supporting_evidence if attempt.validation else []
                    ),
                    "timing_trials": (
                        len(attempt.benchmark_output.get("timings_ms", []))
                        if isinstance(attempt.benchmark_output.get("timings_ms", []), list)
                        else 0
                    ),
                }
                for attempt in plan_attempts
            ],
        }

    @staticmethod
    def _extract_target_observation(target: str, attempt: ProbeAttempt | None) -> dict[str, str]:
        if attempt is None:
            return {"kind": "none", "value": "null", "source": ""}
        if target in attempt.ncu_metrics:
            return {
                "kind": "ncu",
                "value": OutputGenerationModule._render_value(attempt.ncu_metrics[target]),
                "source": target,
            }
        derived = attempt.benchmark_output.get("derived_metrics", {})
        for alias in DERIVED_ALIASES.get(target, []):
            if alias in derived:
                return {
                    "kind": "benchmark",
                    "value": OutputGenerationModule._render_value(derived[alias]),
                    "source": alias,
                }
        return {"kind": "none", "value": "null", "source": ""}

    def _build_per_target_findings(
        self,
        estimates: list[MetricEstimate],
        attempts: list[ProbeAttempt],
    ) -> list[str]:
        lines: list[str] = []
        for estimate in estimates:
            attempt, source_kind, alias = self._resolve_attempt_from_source(estimate.source, attempts)
            lines.append(
                self._build_per_target_finding_text(
                    estimate=estimate,
                    attempt=attempt,
                    source_kind=source_kind,
                    alias=alias,
                    attempts=attempts,
                )
            )
        return lines

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
                f"Used the direct `ncu` metric from `{attempt.probe_family}` variant `{attempt.probe_variant}` "
                f"({attempt.plan_role}) in round {attempt.round_index} after {total_rounds} total round(s) for this plan; "
                f"the benchmark created the profiling condition "
                f"and collected {timings_count} timing sample(s) for repeatability/supporting evidence. "
                f"{estimate.selection_rule}"
            )
        if source_kind == "benchmark":
            alias_text = f" derived metric `{alias}`" if alias else " derived benchmark metric"
            return (
                f"Used the benchmark{alias_text} from `{attempt.probe_family}` variant `{attempt.probe_variant}` "
                f"({attempt.plan_role}) in round {attempt.round_index} after {total_rounds} total round(s); "
                f"the value was kept only after repeated timings "
                f"and `ncu` cross-checking where the agreement threshold was satisfied. "
                f"{estimate.selection_rule}"
            )
        return estimate.reasoning

    def _build_per_target_finding_text(
        self,
        *,
        estimate: MetricEstimate,
        attempt: ProbeAttempt | None,
        source_kind: str,
        alias: str,
        attempts: list[ProbeAttempt],
    ) -> str:
        if attempt is None:
            evidence = self._build_evidence_text(estimate, attempt)
            return (
                f"- `{estimate.target}` remains unresolved. {estimate.reasoning} "
                f"The current value is `{self._render_value(estimate.value)}` with confidence `{estimate.confidence:.2f}`. "
                f"Available evidence: {evidence}"
            )

        total_rounds = max(
            (item.round_index for item in attempts if item.plan_id == attempt.plan_id),
            default=attempt.round_index,
        )
        timings = attempt.benchmark_output.get("timings_ms", [])
        timings_count = len(timings) if isinstance(timings, list) else 0
        value_text = self._render_value(estimate.value)
        evidence = self._build_evidence_text(estimate, attempt)

        if source_kind == "ncu":
            origin = (
                f"The final value is `{value_text}`, taken directly from `ncu` in "
                f"`{attempt.probe_family}` variant `{attempt.probe_variant}` ({attempt.plan_role}) round {attempt.round_index}"
            )
        elif source_kind == "benchmark":
            alias_text = f" derived metric `{alias}`" if alias else " a benchmark-side derived metric"
            origin = (
                f"The current value is `{value_text}`, inferred from{alias_text} in "
                f"`{attempt.probe_family}` variant `{attempt.probe_variant}` ({attempt.plan_role}) round {attempt.round_index}"
            )
        else:
            origin = f"The current value is `{value_text}`, selected from `{estimate.source}`"

        timing_clause = (
            f" The accepted run also recorded {timings_count} timing sample(s) for repeated-trial checks."
            if timings_count
            else ""
        )
        return (
            f"- `{estimate.target}`: {origin} after {total_rounds} round(s) for this plan."
            f"{timing_clause} Confidence is `{estimate.confidence:.2f}`. "
            f"Selection rule: {estimate.selection_rule} "
            f"Supporting evidence includes: {evidence}"
        )

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
                    f"cross-validation passed: {item}"
                    for item in attempt.validation.cross_checks[:2]
                ]
            )

        if attempt.validation and attempt.validation.supporting_evidence:
            evidence_items.extend(
                [
                    f"supporting evidence: {item}"
                    for item in attempt.validation.supporting_evidence[:2]
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
                if plan.skipped:
                    lines.append(
                        f"- `{plan.plan_id}` / `{plan.probe_family}` / target=`{plan.primary_target or ', '.join(plan.targets)}` / "
                        f"variant=`{plan.probe_variant}` / role=`{plan.plan_role}` skipped before execution by early-stop. "
                        f"Reason: {plan.skip_reason or 'no skip reason recorded'}."
                    )
                else:
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
            supporting_evidence: list[str] = []
            if accepted and accepted.validation:
                cross_checks = accepted.validation.cross_checks[:2]
                supporting_evidence = accepted.validation.supporting_evidence[:2]

            line = (
                f"- `{plan.plan_id}` / `{plan.probe_family}` / target=`{plan.primary_target or ', '.join(plan.targets)}` / "
                f"variant=`{plan.probe_variant}` / role=`{plan.plan_role}` ran {total_rounds} round(s); "
                f"accepted round={accepted.round_index if accepted else 'none'}; "
                f"accepted target(s)={', '.join(plan.targets)}; "
                f"timing_trials={timings_count if timings_count else 'n/a'}."
            )
            if cross_checks:
                line += " Cross-validation: " + " | ".join(cross_checks) + "."
            elif supporting_evidence:
                line += " Supporting evidence: " + " | ".join(supporting_evidence) + "."
            else:
                line += " Cross-validation: no accepted benchmark-vs-ncu agreement was recorded."
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
                if plan.skipped:
                    lines.append(
                        f"- `{plan.plan_id}` / `{plan.probe_family}` / target=`{plan.primary_target or ', '.join(plan.targets)}` / "
                        f"variant=`{plan.probe_variant}` / role=`{plan.plan_role}`: no retry or fix history because this plan was skipped by early-stop. "
                        f"Reason: {plan.skip_reason or 'no skip reason recorded'}."
                    )
                else:
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

            lines.append(
                f"- `{plan.plan_id}` / `{plan.probe_family}` / target=`{plan.primary_target or ', '.join(plan.targets)}` / "
                f"variant=`{plan.probe_variant}` / role=`{plan.plan_role}`: " + "; ".join(round_notes) + "."
            )
        return "\n".join(lines)

    @staticmethod
    def _build_method_commitments() -> str:
        return "\n".join(
            [
                "- The agent did not rely on static spec sheets, online lookup tables, or API-based attribute queries as the primary source of truth.",
                "- Prompt constraints explicitly forbid using `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`, `nvidia-smi`, or similar static/device-query shortcuts as the main measurement method.",
                "- Final values follow a predeclared rule order: prefer the target's own accepted round first, and only fall back to shared probes if the target's own accepted rounds do not resolve the metric.",
                "- For DRAM-side targets, the agent rejects candidates that are not physically consistent with the same row's `dram__throughput` and the row's memory-frequency/bus-width-implied peak, and it does not promote incidental opposite-direction traffic into a final read/write target value.",
                "- Benchmark-derived numbers are only called cross-validation evidence when benchmark-side and `ncu`-side values agree within threshold; otherwise they are treated as mismatch diagnostics or supporting context only.",
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
            f"Conclusion: this run executed multiple probe rounds for {len(spec.targets)} targets, "
            f"resolved {resolved_count} non-null results, and based the final report on {accepted_attempts} validated rounds "
            "without relying on static spec sheets or API lookup as the primary source of truth."
        )

    @staticmethod
    def _split_summary_conclusion(summary: str, fallback_conclusion: str) -> tuple[str, str]:
        lines = [line.strip() for line in summary.splitlines() if line.strip()]
        if not lines:
            return fallback_conclusion, fallback_conclusion
        if lines[0].startswith("Conclusion:"):
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
        selection_rules: str,
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
                "The final-value selection rules are:",
                selection_rules,
                (
                    "The per-target outcomes are already documented in the `Per-Target Findings` section; "
                    f"representative resolved metrics include {top_metrics if top_metrics else 'no resolved metrics'}."
                ),
                "Repeated-trial and cross-validation details:",
                trial_digest,
                "Failure, retry, and repair history:",
                retry_digest,
                "Method commitments:",
                method_commitments,
                f"LLM summary generation failed, so this fallback summary was used. Error: {error_text}",
            ]
        )

    @staticmethod
    def _build_final_value_selection_rules() -> str:
        return "\n".join(
            [
                "1. Prefer the target's own accepted direct `ncu` measurement.",
                "2. If that is unavailable, use the target's own accepted benchmark-derived value.",
                "3. If the target's own accepted rounds do not resolve the metric, fall back to an accepted shared probe from the same probe family.",
                "4. Only if no accepted same-family evidence resolves the metric may the system use lower-confidence or cross-family fallback evidence, and the report must say so explicitly.",
                "5. For DRAM-side targets, reject candidate values that are not physically consistent with the same row's `dram__throughput` and memory-frequency/bus-width-implied peak, and reject incidental opposite-direction traffic as a final read/write value.",
                "6. Within the same rule tier, prefer `primary` over `cross_check`, then prefer higher validation confidence and later repaired rounds.",
                "7. `Cross-validation` means benchmark-side and `ncu`-side values agree within threshold; stable timing, high throughput, or returned `ncu` metrics alone are only supporting evidence.",
            ]
        )

    @staticmethod
    def _summarize_feedback(text: str, limit: int = 220) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 3] + "..."
