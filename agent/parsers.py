from __future__ import annotations

import csv
import io
import json
from statistics import fmean
from typing import Any


def extract_last_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}

    candidates: list[str] = []
    depth = 0
    start_index: int | None = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start_index is not None:
                candidates.append(text[start_index : index + 1])
                start_index = None

    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {}


def parse_program_output(stdout: str) -> dict[str, Any]:
    parsed = extract_last_json_object(stdout)
    if not isinstance(parsed, dict):
        return {}
    return parsed


def parse_ncu_csv(stdout: str) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, str]]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    header_index = -1
    for index, line in enumerate(lines):
        if "Metric Name" in line and "Metric Value" in line:
            header_index = index
            break

    if header_index < 0:
        return {}, [], {}

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    values_by_metric: dict[str, list[float]] = {}
    metric_units: dict[str, str] = {}
    raw_rows: list[dict[str, Any]] = []

    for row in reader:
        metric_name = (row.get("Metric Name") or "").strip()
        metric_value = (row.get("Metric Value") or "").strip()
        metric_unit = (row.get("Metric Unit") or "").strip()
        kernel_name = (row.get("Kernel Name") or "").strip()

        if not metric_name or not metric_value:
            continue

        numeric_value = _parse_numeric(metric_value)
        if numeric_value is None:
            continue

        values_by_metric.setdefault(metric_name, []).append(numeric_value)
        metric_units.setdefault(metric_name, metric_unit)
        raw_rows.append(
            {
                "metric_name": metric_name,
                "metric_value": numeric_value,
                "metric_unit": metric_unit,
                "kernel_name": kernel_name,
            }
        )

    summarized = {metric: fmean(values) for metric, values in values_by_metric.items()}
    return summarized, raw_rows, metric_units


def _parse_numeric(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("%", "").replace("\"", "").strip()
    if cleaned.lower() in {"", "nan", "n/a"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None
