#!/usr/bin/env python3
"""Run a student EvoCUA agent on screenshots from teacher trajectories.

The output is one JSON file per task. Each step contains the original teacher
step plus the student's response for the same input screenshot. The script uses
OSWorld's EvoCUA agent implementation through OSWORLD_ROOT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_teacher_steps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        step_num = int(row.get("step_num", 0))
        by_step.setdefault(step_num, []).append(row)

    grouped: list[dict[str, Any]] = []
    for step_num in sorted(by_step):
        items = by_step[step_num]
        merged = dict(items[0])
        merged["action"] = [item.get("action") for item in items]
        merged["screenshot_file"] = items[-1].get("screenshot_file")
        grouped.append(merged)
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osworld-root", default=os.environ.get("OSWORLD_ROOT", ""))
    parser.add_argument("--domain", default="gimp")
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--task-config-dir", required=True)
    parser.add_argument("--teacher-result-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--student-model", default="vllm_evocua-8b")
    parser.add_argument("--vllm-base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:7703"))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "4")))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-history-turns", type=int, default=4)
    return parser.parse_args()


def build_agent(args: argparse.Namespace):
    os.environ["VLLM_BASE_URL"] = args.vllm_base_url
    sys.path.insert(0, str(Path(args.osworld_root).resolve()))
    from mm_agents.evocua.evocua_agent import EvoCUAAgent

    return EvoCUAAgent(
        model=args.student_model,
        temperature=args.temperature,
        observation_type="screenshot",
        max_steps=args.max_steps,
        prompt_style="S2",
        max_history_turns=args.max_history_turns,
        coordinate_type="relative",
        resize_factor=32,
        api_backend="vllm",
    )


def process_task(task_id: str, args: argparse.Namespace) -> str:
    agent = build_agent(args)

    task_config = load_json(Path(args.task_config_dir) / args.domain / f"{task_id}.json")
    instruction = task_config["instruction"]

    task_teacher_dir = Path(args.teacher_result_dir) / args.domain / task_id
    teacher_rows = load_jsonl(task_teacher_dir / "traj.jsonl")
    teacher_steps = group_teacher_steps(teacher_rows)

    results: list[dict[str, Any]] = []
    for idx, step in enumerate(teacher_steps):
        screenshot_name = teacher_steps[max(idx - 1, 0)].get("screenshot_file")
        screenshot_path = task_teacher_dir / str(screenshot_name)
        screenshot_bytes = screenshot_path.read_bytes()

        student_response, student_pyautogui_code = agent.predict(
            instruction, {"screenshot": screenshot_bytes}
        )

        results.append(
            {
                "step_num": step.get("step_num"),
                "input_screenshot": str(screenshot_path),
                "teacher_cot": {
                    "response": step.get("response", ""),
                    "action": step.get("action", []),
                },
                "student_cot": {
                    "response": student_response,
                    "action": student_pyautogui_code,
                },
            }
        )

        if agent.actions:
            agent.actions[-1] = step.get("action", [])
        if agent.responses:
            agent.responses[-1] = step.get("response", "")

    save_path = Path(args.save_dir) / f"{task_id}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return task_id


def main() -> None:
    args = parse_args()
    if not args.osworld_root:
        raise SystemExit("Set --osworld-root or OSWORLD_ROOT.")

    task_payload = load_json(Path(args.task_file))
    task_ids = task_payload[args.domain]

    workers = max(1, args.num_workers)
    if workers == 1:
        for task_id in tqdm(task_ids, desc="pair inference"):
            process_task(task_id, args)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_task, task_id, args): task_id for task_id in task_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="pair inference"):
            future.result()


if __name__ == "__main__":
    main()
