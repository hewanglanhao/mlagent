from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TargetSpec:
    source_path: str
    targets: list[str]
    target_to_probe: dict[str, str]
    used_fallback: bool = False


@dataclass
class BenchmarkPlan:
    plan_id: str
    probe_family: str
    targets: list[str]
    benchmark_objective: str
    benchmark_requirements: list[str]
    ncu_metrics: list[str]
    success_criteria: list[str]
    parser_expectations: list[str]
    max_rounds: int = 3
    tags: list[str] = field(default_factory=list)
    program_args: list[str] = field(default_factory=list)

    def to_prompt_context(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "probe_family": self.probe_family,
            "targets": self.targets,
            "benchmark_objective": self.benchmark_objective,
            "benchmark_requirements": self.benchmark_requirements,
            "ncu_metrics": self.ncu_metrics,
            "success_criteria": self.success_criteria,
            "parser_expectations": self.parser_expectations,
            "max_rounds": self.max_rounds,
            "tags": self.tags,
            "program_args": self.program_args,
        }


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    stdout_path: str = ""
    stderr_path: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class ValidationResult:
    credible: bool
    confidence: float
    issues: list[str] = field(default_factory=list)
    cross_checks: list[str] = field(default_factory=list)


@dataclass
class ProbeAttempt:
    plan_id: str
    probe_family: str
    round_index: int
    generated_source_path: str
    binary_path: str
    generation_feedback: str = ""
    llm_prompt_path: str = ""
    llm_response_path: str = ""
    compile_result: CommandResult | None = None
    run_result: CommandResult | None = None
    profile_result: CommandResult | None = None
    benchmark_output: dict[str, Any] = field(default_factory=dict)
    ncu_metrics: dict[str, float] = field(default_factory=dict)
    ncu_metric_units: dict[str, str] = field(default_factory=dict)
    ncu_rows: list[dict[str, Any]] = field(default_factory=list)
    validation: ValidationResult | None = None


@dataclass
class MetricEstimate:
    target: str
    value: Any
    confidence: float
    source: str
    reasoning: str
    evidence: list[str] = field(default_factory=list)
