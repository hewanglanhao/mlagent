from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .models import CommandResult
from .reasoning import ReasoningLogger


class CompilationAndExecutionModule:
    def __init__(self, run_dir: Path, logger: ReasoningLogger):
        self.logger = logger
        self.generated_dir = run_dir / "generated"
        self.binary_dir = run_dir / "bin"
        self.log_dir = run_dir / "raw"
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.binary_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def write_source(self, plan_id: str, round_index: int, code: str) -> Path:
        path = self.generated_dir / f"{plan_id}_round{round_index}.cu"
        path.write_text(code, encoding="utf-8")
        self.logger.log(
            "compile_execute",
            "wrote_generated_source",
            plan_id=plan_id,
            round_index=round_index,
            source_path=path,
        )
        return path

    def compile(self, source_path: Path, plan_id: str, round_index: int) -> tuple[Path, CommandResult]:
        binary_path = self.binary_dir / f"{plan_id}_round{round_index}"
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            result = CommandResult(
                command=["nvcc"],
                returncode=127,
                stdout="",
                stderr="nvcc not found in PATH.",
                duration_s=0.0,
            )
            return binary_path, self._persist_result(result, f"{plan_id}_round{round_index}.compile")

        command = [
            nvcc,
            str(source_path),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-o",
            str(binary_path),
        ]
        result = self._run(command)
        persisted = self._persist_result(result, f"{plan_id}_round{round_index}.compile")
        self.logger.log(
            "compile_execute",
            "compile_finished",
            plan_id=plan_id,
            round_index=round_index,
            command=command,
            returncode=persisted.returncode,
        )
        return binary_path, persisted

    def run(self, binary_path: Path, plan_id: str, round_index: int, program_args: list[str]) -> CommandResult:
        command = [str(binary_path), *program_args]
        result = self._run(command)
        persisted = self._persist_result(result, f"{plan_id}_round{round_index}.run")
        self.logger.log(
            "compile_execute",
            "benchmark_finished",
            plan_id=plan_id,
            round_index=round_index,
            command=command,
            returncode=persisted.returncode,
        )
        return persisted

    def _run(self, command: list[str], timeout_s: int = 300) -> CommandResult:
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_s,
            )
            duration_s = time.perf_counter() - start
            return CommandResult(
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
            return CommandResult(
                command=command,
                returncode=124,
                stdout=stdout_text,
                stderr=stderr_text + f"\nCommand timed out after {timeout_s} seconds.",
                duration_s=duration_s,
            )

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
