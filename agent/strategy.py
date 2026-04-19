from __future__ import annotations

from .models import BenchmarkPlan, TargetSpec
from .probe_planner import TargetProbePlanningModule
from .reasoning import ReasoningLogger


class BenchmarkStrategySelectionModule:
    def __init__(self, logger: ReasoningLogger):
        self.logger = logger
        self.target_probe_planner = TargetProbePlanningModule(logger)

    def build_plans(self, spec: TargetSpec) -> list[BenchmarkPlan]:
        plans = self.target_probe_planner.build_target_plans(spec)
        self.logger.log("strategy_selector", "built_benchmark_plans", plans=plans)
        return plans
