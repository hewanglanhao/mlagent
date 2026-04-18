from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .models import CommandResult
from .reasoning import ReasoningLogger


class NCUProfilingModule:
    def __init__(self, run_dir: Path, logger: ReasoningLogger):
        self.logger = logger
        self.log_dir = run_dir / "raw"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def profile(
        self,
        binary_path: Path,
        plan_id: str,
        round_index: int,
        metrics: list[str],
        program_args: list[str],
    ) -> CommandResult:
        ncu = shutil.which("ncu")
        if ncu is None:
            result = CommandResult(
                command=["ncu"],
                returncode=127,
                stdout="",
                stderr="ncu not found in PATH.",
                duration_s=0.0,
            )
            return self._persist_result(result, f"{plan_id}_round{round_index}.ncu")

        command = [
            ncu,
            "-f",
            "--target-processes",
            "all",
            "--page",
            "raw",
            "--csv",
            "--metrics",
            ",".join(metrics),
            str(binary_path),
            *program_args,
        ]

        start = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
                timeout=300,
            )
            duration_s = time.perf_counter() - start
            result = CommandResult(
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_s=duration_s,
            )
        except subprocess.TimeoutExpired as exc:
            duration_s = time.perf_counter() - start
            result = CommandResult(
                command=command,
                returncode=124,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + "\nNCU profiling timed out after 300 seconds.",
                duration_s=duration_s,
            )

        persisted = self._persist_result(result, f"{plan_id}_round{round_index}.ncu")
        self.logger.log(
            "ncu_profiler",
            "profiling_finished",
            plan_id=plan_id,
            round_index=round_index,
            command=command,
            returncode=persisted.returncode,
        )
        return persisted

    def _persist_result(self, result: CommandResult, stem: str) -> CommandResult:
        stdout_path = self.log_dir / f"{stem}.stdout.txt"
        stderr_path = self.log_dir / f"{stem}.stderr.txt"
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")
        result.stdout_path = str(stdout_path)
        result.stderr_path = str(stderr_path)
        return result
