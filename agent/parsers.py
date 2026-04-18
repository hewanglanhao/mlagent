from __future__ import annotations

import csv
import io
import json
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
            return json.loads(_sanitize_json_candidate(candidate))
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
    legacy_header_index = -1
    wide_header_index = -1
    for index, line in enumerate(lines):
        if "Metric Name" in line and "Metric Value" in line:
            legacy_header_index = index
            break
        if wide_header_index < 0 and _looks_like_wide_ncu_header(line):
            wide_header_index = index

    if legacy_header_index >= 0:
        return _parse_legacy_ncu_csv(lines[legacy_header_index:])
    if wide_header_index >= 0:
        return _parse_wide_ncu_csv(lines[wide_header_index:])
    return {}, [], {}


def _parse_legacy_ncu_csv(lines: list[str]) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, str]]:
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
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

    summarized = {
        metric: _summarize_metric_values(metric, values)
        for metric, values in values_by_metric.items()
        if values
    }
    return summarized, raw_rows, metric_units


def _parse_wide_ncu_csv(lines: list[str]) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, str]]:
    reader = csv.reader(io.StringIO("\n".join(lines)))
    rows = list(reader)
    if len(rows) < 3:
        return {}, [], {}

    header = rows[0]
    units_row = rows[1]
    metric_units: dict[str, str] = {}
    raw_rows: list[dict[str, Any]] = []
    values_by_metric: dict[str, list[float]] = {}

    for row in rows[2:]:
        if not any(cell.strip() for cell in row):
            continue
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))

        row_record: dict[str, Any] = {}
        kernel_name = row[_safe_index(header, "Kernel Name")] if "Kernel Name" in header else ""
        if kernel_name:
            row_record["kernel_name"] = kernel_name

        row_id = row[_safe_index(header, "ID")] if "ID" in header else ""
        if row_id:
            row_record["id"] = row_id

        for index, column_name in enumerate(header):
            if not column_name:
                continue
            numeric_value = _parse_numeric(row[index])
            if numeric_value is None:
                continue
            values_by_metric.setdefault(column_name, []).append(numeric_value)
            if index < len(units_row) and units_row[index].strip():
                metric_units.setdefault(column_name, units_row[index].strip())
            row_record[column_name] = numeric_value

        if row_record:
            raw_rows.append(row_record)

    summarized = {
        metric: _summarize_metric_values(metric, values)
        for metric, values in values_by_metric.items()
        if values
    }
    return summarized, raw_rows, metric_units


def _looks_like_wide_ncu_header(line: str) -> bool:
    try:
        row = next(csv.reader([line]))
    except csv.Error:
        return False
    return "ID" in row and "Kernel Name" in row


def _safe_index(items: list[str], value: str) -> int:
    try:
        return items.index(value)
    except ValueError:
        return -1


def _summarize_metric_values(metric_name: str, values: list[float]) -> float:
    if not values:
        raise ValueError(f"No values available for metric {metric_name}.")
    # Raw ncu output may contain one row per kernel launch; for per-target inference we
    # keep the strongest observed value so read/write kernels do not dilute each other.
    return max(values)


def _parse_numeric(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("%", "").replace("\"", "").strip()
    if cleaned.lower() in {"", "nan", "n/a"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _sanitize_json_candidate(candidate: str) -> str:
    replacements = {
        "inf": "null",
        "+inf": "null",
        "-inf": "null",
        "nan": "null",
        "+nan": "null",
        "-nan": "null",
    }
    normalized: list[str] = []
    in_string = False
    escape = False
    index = 0

    while index < len(candidate):
        char = candidate[index]
        if in_string:
            normalized.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "\"":
                in_string = False
            index += 1
            continue

        if char == "\"":
            in_string = True
            normalized.append(char)
            index += 1
            continue

        matched = False
        for token, replacement in replacements.items():
            end_index = index + len(token)
            if (
                candidate.startswith(token, index)
                and _is_json_token_boundary(candidate, index - 1)
                and _is_json_token_boundary(candidate, end_index)
            ):
                normalized.append(replacement)
                index = end_index
                matched = True
                break
        if matched:
            continue

        normalized.append(char)
        index += 1

    return "".join(normalized)


def _is_json_token_boundary(text: str, index: int) -> bool:
    if index < 0 or index >= len(text):
        return True
    return text[index] in " \t\r\n,:[]{}"
