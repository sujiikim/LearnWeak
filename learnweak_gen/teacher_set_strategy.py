#!/usr/bin/env python3
"""Compare teacher/student verify results and summarize student weaknesses.

This script:
1) Loads teacher and student verify_results JSON files.
2) Finds tasks where teacher passed but student failed.
3) Collects instruction + student agent_failure_analysis.
4) Calls GPT-5-mini to generate a high-level capability gap summary.
5) Saves fail pairs + model analysis into a single JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class TaskRecord:
    task_id: str
    instruction: str
    agent_pass: Optional[bool]
    agent_failure_analysis: str
    raw: Dict[str, Any]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _extract_agent_failure_analysis(item: Dict[str, Any]) -> str:
    result = item.get("result")
    if isinstance(result, dict):
        value = result.get("agent_failure_analysis", "")
        if isinstance(value, str):
            return value.strip()
    return ""


def _extract_task_records(data: Any) -> List[TaskRecord]:
    if not isinstance(data, list):
        raise ValueError("verify_results JSON must be a list of task result objects.")

    records: List[TaskRecord] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        task_id = item.get("task_id")
        instruction = item.get("instruction")
        if not isinstance(task_id, str) or not task_id.strip():
            continue
        if not isinstance(instruction, str):
            instruction = ""

        records.append(
            TaskRecord(
                task_id=task_id.strip(),
                instruction=instruction.strip(),
                agent_pass=_safe_bool(item.get("agent_pass")),
                agent_failure_analysis=_extract_agent_failure_analysis(item),
                raw=item,
            )
        )
    return records


def _index_by_task_id(records: Sequence[TaskRecord]) -> Dict[str, TaskRecord]:
    index: Dict[str, TaskRecord] = {}
    for r in records:
        # Keep first occurrence to avoid accidental overwrite in duplicated entries.
        if r.task_id not in index:
            index[r.task_id] = r
    return index


def collect_teacher_pass_student_fail(
    teacher_records: Sequence[TaskRecord], student_records: Sequence[TaskRecord]
) -> List[Dict[str, str]]:
    teacher_map = _index_by_task_id(teacher_records)
    selected: List[Dict[str, str]] = []

    for student in student_records:
        teacher = teacher_map.get(student.task_id)
        if teacher is None:
            continue

        if teacher.agent_pass is True and student.agent_pass is False:
            selected.append(
                {
                    "task_id": student.task_id,
                    "instruction": student.instruction,
                    "student_agent_failure_analysis": student.agent_failure_analysis,
                }
            )
    return selected


def build_gpt_prompt(items: Sequence[Dict[str, str]], max_items: Optional[int] = None) -> str:
    if max_items is not None and max_items > 0:
        items = items[:max_items]

    header = (
        "You are analyzing failure patterns of a student UI agent.\n"
        "Input cases are tasks where TEACHER passed but STUDENT failed.\n\n"
        "For each case, you receive:\n"
        "- instruction\n"
        "- student_agent_failure_analysis (judge explanation)\n\n"
        "Please produce a concise, high-level report in JSON with this schema:\n"
        "{\n"
        '  "overall_summary": "string",\n'
        '  "failure_categories": [\n'
        "    {\n"
        '      "category": "string",\n'
        '      "what_student_cannot_do": "string",\n'
        '      "likely_failed_features_or_operations": ["string"],\n'
        "    }\n"
        "  ],\n"
        "}\n\n"
        "Requirements:\n"
        "1) Focus on sub-tasks the agent cannot do reliably.\n"
        "2) Identify concrete operations the agent misuses or fails to execute.\n"
        "3) Categories should be notably different from each other. Do not include similar categories.\n"
        "4) Group repeated failures into reusable categories.\n"
        "5) Do not include markdown, return JSON only.\n"
    )

    body_lines = ["\nCases:\n"]
    for i, item in enumerate(items, start=1):
        instruction = item.get("instruction", "").strip()
        failure = item.get("student_agent_failure_analysis", "").strip()
        task_id = item.get("task_id", "").strip()
        body_lines.append(f"[Case {i}] task_id={task_id}")
        body_lines.append(f"instruction: {instruction}")
        body_lines.append(
            "student_agent_failure_analysis: "
            + (failure if failure else "(empty; no analysis text provided)")
        )
        body_lines.append("")

    return header + "\n".join(body_lines)


def _maybe_import_openai():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OpenAI SDK is not installed. Install it with: pip install openai"
        ) from e
    return OpenAI


def run_gpt5_mini(prompt: str, model: str = "gpt-5-mini") -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    OpenAI = _maybe_import_openai()
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=prompt,
        text={"format": {"type": "text"}},
    )
    return (response.output_text or "").strip()


def _parse_gpt_json_text(text: str) -> Any:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()
            if raw.lower().startswith("json\n"):
                raw = raw[5:].strip()
    return json.loads(raw)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _resolve_output_path(raw_path: str, output_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return output_dir / path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare teacher/student verify_results and build GPT-5-mini analysis "
            "for teacher-pass/student-fail tasks."
        )
    )
    parser.add_argument("--teacher", required=True, help="Path to teacher verify_results.json")
    parser.add_argument("--student", required=True, help="Path to student verify_results.json")
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Base directory for relative output paths. "
            "Default: parent directory of --student file."
        ),
    )
    parser.add_argument(
        "--report-out",
        default="teacher_pass_student_fail_report.json",
        help="Output filename/path for combined report JSON.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Limit number of cases included in prompt (0 means all).",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Model name for OpenAI Responses API.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    teacher_path = Path(args.teacher).expanduser()
    student_path = Path(args.student).expanduser()
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else student_path.parent
    )
    report_out_path = _resolve_output_path(args.report_out, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_data = _read_json(teacher_path)
    student_data = _read_json(student_path)

    teacher_records = _extract_task_records(teacher_data)
    student_records = _extract_task_records(student_data)
    assert len(teacher_records) == len(student_records)
    pairs = collect_teacher_pass_student_fail(teacher_records, student_records)

    max_items = args.max_items if args.max_items > 0 else None
    prompt = build_gpt_prompt(pairs, max_items=max_items)
    analysis_text = run_gpt5_mini(prompt=prompt, model=args.model)
    analysis_json = _parse_gpt_json_text(analysis_text)

    report = {
        "meta": {
            "teacher_file": str(teacher_path),
            "student_file": str(student_path),
            "model": args.model,
            "teacher_record_count": len(teacher_records),
            "student_record_count": len(student_records),
            "teacher_pass_student_fail_count": len(pairs),
        },
        "teacher_pass_student_fail_pairs": pairs,
        "gpt_analysis": analysis_json,
    }
    _write_json(report_out_path, report)

    print(f"Teacher records: {len(teacher_records)}")
    print(f"Student records: {len(student_records)}")
    print(f"Teacher-pass/student-fail pairs: {len(pairs)}")
    print(f"Output dir: {output_dir}")
    print(f"Saved combined report: {report_out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


'''
        "You are analyzing failure patterns of a student UI agent.\n"
        "Input cases are tasks where TEACHER passed but STUDENT failed.\n\n"
        "For each case, you receive:\n"
        "- instruction\n"
        "- student_agent_failure_analysis (judge explanation)\n\n"
        "Please produce a concise, high-level report in JSON with this schema:\n"
        "{\n"
        '  "overall_summary": "string",\n'
        '  "failure_categories": [\n'
        "    {\n"
        '      "category": "string",\n'
        '      "what_student_cannot_do": "string",\n'
        '      "likely_failed_features_or_operations": ["string"],\n'
        '      "training_needs": ["string"],\n'
        '      "supporting_task_ids": ["string"],\n'
        '      "example_instructions": ["string"]\n'
        "    }\n"
        "  ],\n"
        '  "data_generation_recommendations": ["string"]\n'
        "}\n\n"
        "Requirements:\n"
        "1) Focus on sub-tasks the agent cannot do reliably.\n"
        "2) Identify concrete VS Code/UI operations the agent misuses or fails to execute.\n"
        "3) Suggest what additional training data should be generated.\n"
        "4) Group repeated failures into reusable categories.\n"
        "5) Do not include markdown, return JSON only.\n"
'''
