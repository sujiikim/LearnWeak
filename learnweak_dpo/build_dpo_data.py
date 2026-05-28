#!/usr/bin/env python3
"""Build LearnWeak DPO data from teacher/student trajectory pairs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from learnweak_dpo.evocua_output_parser import parse_response, parse_toolcall_json
from learnweak_dpo.evocua_prompt import gen_SYSTEM_PROMPT


SYSTEM_PROMPT = gen_SYSTEM_PROMPT()
USER_TEMPLATE = """Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {instruction}

Previous actions:
{previous_actions}"""

COORD_THRESHOLD = 20
ACTIONS_WITH_COORDINATE = {
    "left_click",
    "double_click",
    "right_click",
    "mouse_move",
    "triple_click",
    "left_click_drag",
}
ACTIONS_IGNORE_PARAMS = {"wait", "terminate", "scroll"}
ACTIONS_COMPARE_FIELD = {"key": "keys", "type": "text"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def verify_to_dict(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_json(path)
    return {row["task_id"]: row for row in rows if isinstance(row, dict) and "task_id" in row}


def load_tasks(task_file: Path, examples_root: Path, domain: str) -> list[tuple[str, str]]:
    payload = load_json(task_file)
    tasks = []
    for task_id in payload[domain]:
        cfg = load_json(examples_root / domain / f"{task_id}.json")
        tasks.append((task_id, str(cfg.get("instruction", ""))))
    return tasks


def params_same(student: dict[str, Any], teacher: dict[str, Any]) -> bool:
    action = student.get("action")
    if action in ACTIONS_IGNORE_PARAMS:
        return True
    if action in ACTIONS_COMPARE_FIELD:
        field = ACTIONS_COMPARE_FIELD[action]
        return student.get(field) == teacher.get(field)
    if action in ACTIONS_WITH_COORDINATE:
        sc, tc = student.get("coordinate"), teacher.get("coordinate")
        if sc is not None and tc is not None:
            return abs(sc[0] - tc[0]) <= COORD_THRESHOLD and abs(sc[1] - tc[1]) <= COORD_THRESHOLD
        return sc is None and tc is None
    return True


def classify_actions(student_actions: list[dict[str, Any]], teacher_actions: list[dict[str, Any]]) -> str:
    student_args = [a.get("arguments") for a in student_actions if a.get("name") == "computer_use"]
    teacher_args = [a.get("arguments") for a in teacher_actions if a.get("name") == "computer_use"]
    student_args = [a for a in student_args if isinstance(a, dict) and a.get("action") != "wait"]
    teacher_args = [a for a in teacher_args if isinstance(a, dict) and a.get("action") != "wait"]

    if not student_args or not teacher_args:
        return "action_wait"
    if len(student_args) != len(teacher_args):
        return "diff_len"

    labels = []
    for student, teacher in zip(student_args, teacher_args):
        if student.get("action") != teacher.get("action"):
            labels.append("diff_action")
        elif params_same(student, teacher):
            labels.append("exact_match")
        else:
            labels.append("diff_param")

    if all(label == "exact_match" for label in labels):
        return "exact_match"
    if "diff_action" in labels:
        return "diff_action"
    if "diff_param" in labels:
        return "diff_param"
    return "unknown"


def compare_step(teacher_cot: dict[str, Any], student_cot: dict[str, Any]) -> tuple[bool, str | None]:
    teacher_tool = teacher_cot.get("tool")
    student_tool = student_cot.get("tool")
    if teacher_tool is None or student_tool is None:
        return False, None

    label = classify_actions(parse_toolcall_json(student_tool), parse_toolcall_json(teacher_tool))
    if label == "diff_param":
        return True, "learn_param"
    if label in {"diff_action", "diff_len"}:
        return True, "learn_action"
    return False, None


def assistant_content(cot: dict[str, Any]) -> str:
    tools = cot.get("tool") or []
    tool_blocks = "\n".join(f"<tool_call>\n{tool}\n</tool_call>" for tool in tools)
    return f"<think>{cot.get('thought') or ''}\n</think>\n\nAction: {cot.get('action') or ''}\n{tool_blocks}"


def make_sample(
    instruction: str,
    image_path: Path,
    teacher_cot: dict[str, Any],
    student_cot: dict[str, Any],
    learn_type: str,
) -> dict[str, Any]:
    user_text = USER_TEMPLATE.format(instruction=instruction, previous_actions="None")
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\n{user_text}"},
        ],
        "chosen": {"role": "assistant", "content": assistant_content(teacher_cot)},
        "rejected": {"role": "assistant", "content": assistant_content(student_cot)},
        "learn_type": learn_type,
        "images": [str(image_path)],
    }


def iter_step_names(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def step_num(step_name: str) -> int:
    return 1 if step_name == "seed" else int(step_name.replace("iter", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="gimp")
    parser.add_argument("--steps", default="seed,iter2,iter3,iter4,iter5")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "learnweak_gen/data"))
    parser.add_argument("--generated-data-root", default=str(REPO_ROOT / "learnweak_gen/data/synthetic_evocua"))
    parser.add_argument("--result-root", default=str(REPO_ROOT / "learnweak_gen/rollouts"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "learnweak_dpo/data/llamafactory"))
    parser.add_argument("--image-dir", default=str(REPO_ROOT / "learnweak_dpo/data/dpo_images"))
    args = parser.parse_args()

    dataset: list[dict[str, Any]] = []
    image_dir = Path(args.image_dir) / args.domain
    image_dir.mkdir(parents=True, exist_ok=True)

    for name in iter_step_names(args.steps):
        n = step_num(name)
        if name == "seed":
            task_file = Path(args.data_root) / "seed" / f"test_{args.domain}.json"
            examples_root = Path(args.data_root) / "seed" / "examples"
        else:
            task_file = Path(args.generated_data_root) / name / f"test_{args.domain}.json"
            examples_root = Path(args.generated_data_root) / name / "examples"

        teacher_dir = Path(args.result_root) / "dataset_generation" / args.domain / f"evocua-32b_synthetic_step{n}" / "pyautogui/screenshot/meituan/EvoCUA-32B-20260105"
        student_dir = Path(args.result_root) / "dataset_generation" / args.domain / f"evocua-8b_synthetic_step{n}" / "pyautogui/screenshot/vllm_evocua-8b"
        pair_dir = Path(args.result_root) / "dataset_generation" / args.domain / f"evocua-32b_synthetic_step{n}_student"

        teacher_verify = verify_to_dict(teacher_dir / "verify_results.json")
        student_verify = verify_to_dict(student_dir / "verify_results.json")

        usable = 0
        for task_id, instruction in load_tasks(task_file, examples_root, args.domain):
            task_key = f"{args.domain}/{task_id}"
            teacher_success = (teacher_verify.get(task_key) or teacher_verify.get(task_id) or {}).get("agent_pass")
            student_success = (student_verify.get(task_key) or student_verify.get(task_id) or {}).get("agent_pass")
            teacher_pass_student_fail = teacher_success is True and student_success is not True
            if not teacher_pass_student_fail:
                continue

            pair_path = pair_dir / f"{task_id}.json"
            if not pair_path.exists():
                print(f"Warning: missing pair trajectory: {pair_path}")
                continue
            for step in load_json(pair_path):
                teacher_cot = parse_response(step.get("teacher_cot", {}).get("response", ""))
                student_cot = parse_response(step.get("student_cot", {}).get("response", ""))
                use_step, learn_type = compare_step(teacher_cot, student_cot)
                if not use_step or learn_type is None:
                    continue

                src_image = Path(step["input_screenshot"])
                dst_image = image_dir / f"{name}_{task_id}_step{step.get('step_num')}_input.png"
                shutil.copyfile(src_image, dst_image)
                dataset.append(make_sample(instruction, dst_image, teacher_cot, student_cot, learn_type))
                usable += 1
        print(f"{name}: {usable} samples")

    dataset_name = f"evocua_synthetic_dpo_{args.domain}"
    out_path = Path(args.out_dir) / f"{dataset_name}.json"
    dump_json(out_path, dataset)
    dataset_info_path = Path(args.out_dir) / "dataset_info.json"
    dataset_info = load_json(dataset_info_path) if dataset_info_path.exists() else {}
    dataset_info[dataset_name] = {
        "file_name": out_path.name,
        "ranking": True,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
            "images": "images",
        },
    }
    dump_json(dataset_info_path, dataset_info)
    print(f"Saved {len(dataset)} samples to {out_path}")


if __name__ == "__main__":
    main()
