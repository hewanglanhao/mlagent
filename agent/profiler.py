from __future__ import annotations

import os
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
        timeout_s: int = 300,
        env_overrides: dict[str, str] | None = None,
        kernel_name_filter: str = "",
        launch_count: int = 0,
        launch_skip: int = 0,
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
        ]
        if kernel_name_filter:
            command.extend(["--kernel-name-base", "demangled", "-k", f"regex:{kernel_name_filter}"])
        if launch_skip > 0:
            command.extend(["-s", str(launch_skip)])
        if launch_count > 0:
            command.extend(["-c", str(launch_count), "--kill", "1"])
        command.extend(
            [
            str(binary_path),
            *program_args,
            ]
        )
        env = os.environ.copy()
        env.update(env_overrides or {})

        start = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_s,
                env=env,
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
            stdout_text = _coerce_text(exc.stdout)
            stderr_text = _coerce_text(exc.stderr)
            result = CommandResult(
                command=command,
                returncode=124,
                stdout=stdout_text,
                stderr=stderr_text + f"\nNCU profiling timed out after {timeout_s} seconds.",
                duration_s=duration_s,
            )

        persisted = self._persist_result(result, f"{plan_id}_round{round_index}.ncu")
        self.logger.log(
            "ncu_profiler",
            "profiling_finished",
            plan_id=plan_id,
            round_index=round_index,
            command=command,
            timeout_s=timeout_s,
            env_overrides=env_overrides or {},
            kernel_name_filter=kernel_name_filter,
            launch_count=launch_count,
            launch_skip=launch_skip,
            returncode=persisted.returncode,
        )
        return persisted

    def _persist_result(self, result: CommandResult, stem: str) -> CommandResult:
        stdout_path = self.log_dir / f"{stem}.stdout.txt"
        stderr_path = self.log_dir / f"{stem}.stderr.txt"
        stdout_path.write_text(_coerce_text(result.stdout), encoding="utf-8")
        stderr_path.write_text(_coerce_text(result.stderr), encoding="utf-8")
        result.stdout_path = str(stdout_path)
        result.stderr_path = str(stderr_path)
        return result


def _coerce_text(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data
