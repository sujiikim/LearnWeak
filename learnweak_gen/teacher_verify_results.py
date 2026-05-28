"""
Verifier for GUI trajectories from a single agent run folder.
Given instruction + trajectory (+ screenshots), judge task pass/fail.
If fail, return a concise failure reason.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

STAGE = "seed"
DOMAIN = "libreoffice_impress"
# Defaults — override with CLI or env (OSWORLD_TRAJ_DIR).
# DEFAULT_TRAJ_DIR = Path(f"/c2/kangsan/OSWorld/results/dataset_generation/{DOMAIN}/opencua-7b_synthetic_step1/pyautogui/screenshot/vllm_opencua/{DOMAIN}")
DEFAULT_TRAJ_DIR = Path(f"/c2/kangsan/OSWorld/results/dataset_generation/{DOMAIN}/evocua-32b_synthetic_step1/pyautogui/screenshot/meituan/EvoCUA-32B-20260105/{DOMAIN}")
# Verification results live next to the run directory.
DEFAULT_OUT = DEFAULT_TRAJ_DIR.parent / "verify_results.json"
# Task instructions: per-example JSON under synthetic seed {DOMAIN} (dataset_generation / teacher_check).
DEFAULT_EVALUATION_EXAMPLES_ROOT = Path(
    os.environ.get("OSWORLD_EVALUATION_EXAMPLES", "/c2/kangsan/OSWorld/evaluation_examples")
)
DEFAULT_INSTRUCTION_DIR = (
    DEFAULT_EVALUATION_EXAMPLES_ROOT / "synthetic" / STAGE / "examples" / DOMAIN
)
MODEL = "gpt-5-mini"
MAX_IMAGE_PIXELS = 750_000


def read_traj(traj_path: Path) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    with traj_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                steps.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return steps


def encode_image(path: Path) -> str | None:
    if not path.exists():
        print(f"Image not found: {path}")
        return None
    ext = path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"

    try:
        with Image.open(path) as img:
            w, h = img.size
            total_pixels = w * h
            if total_pixels > MAX_IMAGE_PIXELS:
                scale = (MAX_IMAGE_PIXELS / float(total_pixels)) ** 0.5
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            fmt = "PNG" if mime == "image/png" else "JPEG"
            if fmt == "JPEG" and img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(buf, format=fmt, quality=90, optimize=True)
            data = base64.b64encode(buf.getvalue()).decode("ascii")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        print(f"Skipping unreadable image {path}: {e}")
        return None
    return f"data:{mime};base64,{data}"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def iter_task_config_paths(
    task_id: str,
    instruction_dir: Path,
    evaluation_examples_root: Path,
) -> list[Path]:
    """OSWorld stores each task's instruction in `{id}.json` under examples/vs_code (or synthetic seed)."""
    return _dedupe_paths(
        [
            instruction_dir / f"{task_id}.json",
        ]
    )


def read_instruction_from_task_config(
    task_id: str,
    instruction_dir: Path,
    evaluation_examples_root: Path,
) -> tuple[str, Path | None]:
    """Load `instruction` from the first existing task config JSON. Returns (text, path used)."""
    for p in iter_task_config_paths(task_id, instruction_dir, evaluation_examples_root):
        if not p.is_file():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            return str(payload.get("instruction", "")).strip(), p
        except Exception:
            continue
    return "", None


def step_action_text(step: dict[str, Any]) -> str:
    action = step.get("action")
    if action is None:
        return ""
    if isinstance(action, str):
        return action.strip()
    return json.dumps(action, ensure_ascii=False)


def merge_steps_by_step_num(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge consecutive rows that share the same `step_num`.
    - response: keep the first row's response
    - action: join all actions with '\n'
    - screenshot_file: keep the last row's screenshot_file
    """
    if not steps:
        return []

    merged: list[dict[str, Any]] = []
    current_group: list[dict[str, Any]] = []
    current_step_num = object()

    def flush_group(group: list[dict[str, Any]]) -> None:
        if not group:
            return
        first = group[0]
        action_lines: list[str] = []
        for g in group:
            act = step_action_text(g)
            if act:
                action_lines.append(act)
        merged.append(
            {
                "step_num": first.get("step_num"),
                "response": first.get("response", ""),
                "action": "\n".join(action_lines),
                "screenshot_file": group[-1].get("screenshot_file"),
            }
        )

    for s in steps:
        sn = s.get("step_num")
        if not current_group:
            current_group = [s]
            current_step_num = sn
            continue
        if sn == current_step_num:
            current_group.append(s)
            continue
        flush_group(current_group)
        current_group = [s]
        current_step_num = sn

    flush_group(current_group)
    return merged


def format_steps_text_only(steps: list[dict[str, Any]], role: str) -> str:
    """Plain-text listing: no screenshots. Only response + executed action per step."""
    lines: list[str] = [f"### {role} trajectory ({len(steps)} step line(s))"]
    for s in steps:
        sn = s.get("step_num")
        prefix = f"Step {sn}" if sn is not None else "Step"
        act = step_action_text(s)
        resp = str(s.get("response", "") or "").strip()
        lines.append(f"\n--- {prefix} ---\nAction:\n{act}\n\nResponse:\n{resp}")
    return "\n".join(lines)


def build_agent_only_input(
    instruction: str,
    agent_steps: list[dict[str, Any]],
    agent_task_dir: Path,
    include_response: bool = True,
) -> list[dict[str, Any]]:
    step_order_line = (
        "For each step, data is ordered as: Thinking/Response -> Executed Action -> Screenshot.\n"
        if include_response
        else "For each step, data is ordered as: Executed Action -> Screenshot.\n"
    )
    prompt = (
        "You are a strict verifier for one GUI task.\n"
        "You will receive the task instruction and an agent step sequence.\n"
        f"{step_order_line}"
        f"## Task instruction\n{instruction if instruction else '(missing)'}\n\n"
        "## Your job\n"
        "1) Analyze the task instruction and set the criteria for task completion. All tasks in the intruction should be completed to get the pass.\n"
        "2) Decide if the agent correctly completed the task objective (pass/fail).\n"
        "3) If fail, provide a SHORT reason (3-4 sentences), concrete and behavior-focused. This should be detailed enough to help the agent to improve without seeing the trajectory. Include which sub-task it failed, which component if did not ground correctly, or why the progress stucked.\n\n"
        "Return STRICT JSON only, with this exact schema:\n"
        "{\n"
        '  "task_completion_criteria": "list of task completion criteria"\n'
        '  "verification_process": "check the task completion criteria one by one based on the trajectory and screenshots."\n'
        '  "agent_pass": true or false,\n'
        '  "agent_failure_analysis": "detailed reason in 3-4 sentences, do not directly refer the criteria but mention again why it failed based on the analysis with the criteria; use empty string if agent_pass is true"\n'
        "}\n"
        "Rules:\n"
        '- Pass only if the agent completed all required tasks in the instruction correctly. Check whether each task is faithfully completed one by one.'
        '- Becareful that the model may think the right plan in thought but the wrong action in the execution. Check again with the screenshot evidence.'
        '- Do not trust self-reports like "done", "completed", or DONE action by themselves.\n'
        "- Judge by actual trajectory behavior and screenshot evidence.\n"
        '- Do not rely on literal "PASS"/"FAIL" labels in terminate messages.\n'
        "- Be concise. English only in JSON values.\n"
    )
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]

    for s in agent_steps:
        step_num = s.get("step_num")
        action = step_action_text(s)
        response = str(s.get("response", "") or "").strip()
        step_text = f"Step {step_num}\n"
        if include_response:
            step_text += f"Thinking/Response:\n{response}\n\n"
        step_text += f"Action:\n{action}"
        content.append(
            {
                "type": "input_text",
                "text": step_text,
            }
        )
        screenshot_name = s.get("screenshot_file")
        if screenshot_name:
            img_uri = encode_image(agent_task_dir / screenshot_name)
            if img_uri:
                content.append({"type": "input_image", "image_url": img_uri})
    return content


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trajectory verification via gpt-5-mini."
    )
    p.add_argument(
        "--traj-dir",
        type=Path,
        default=Path(
            os.environ.get("OSWORLD_TRAJ_DIR", str(DEFAULT_TRAJ_DIR))
        ),
        help="Directory containing per-task folders with traj.jsonl (single agent run).",
    )
    p.add_argument(
        "--instruction-dir",
        type=Path,
        default=DEFAULT_INSTRUCTION_DIR,
        help="Primary directory for {task_id}.json (instruction + config). Falls back under --evaluation-examples-root.",
    )
    p.add_argument(
        "--evaluation-examples-root",
        type=Path,
        default=DEFAULT_EVALUATION_EXAMPLES_ROOT,
        help="OSWorld evaluation_examples root; used to resolve examples/vs_code/... if needed.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON path.",
    )
    p.add_argument("--model", type=str, default=MODEL)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process the first N tasks (after sorting by task id).",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max verification attempts per trajectory. Early-stop on first fail.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
        help="Number of parallel workers across tasks.",
    )
    p.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse existing results in --out by task_id; if omitted, re-infer all tasks.",
    )
    p.add_argument(
        "--without-response",
        action="store_true",
        help="Do not include trajectory 'response' text; pass only action and screenshot.",
    )
    return p.parse_args()


def evaluate_one_task(
    task_id: str,
    *,
    args: argparse.Namespace,
    traj_dir: Path,
) -> dict[str, Any]:
    traj_path = traj_dir / task_id / "traj.jsonl"
    if not traj_path.exists():
        _inst, _inst_path = read_instruction_from_task_config(
            task_id, args.instruction_dir, args.evaluation_examples_root
        )
        return {
            "task_id": task_id,
            "instruction": _inst,
            "instruction_config_path": str(_inst_path) if _inst_path else None,
            "error": "missing traj.jsonl",
        }

    agent_steps_raw = read_traj(traj_path)
    agent_steps = merge_steps_by_step_num(agent_steps_raw)
    instruction, instruction_config_path = read_instruction_from_task_config(
        task_id, args.instruction_dir, args.evaluation_examples_root
    )

    if not agent_steps:
        return {
            "task_id": task_id,
            "instruction": instruction,
            "instruction_config_path": str(instruction_config_path)
            if instruction_config_path
            else None,
            "error": "empty trajectory",
            "agent_steps": len(agent_steps),
            "agent_steps_raw": len(agent_steps_raw),
        }

    user_content = build_agent_only_input(
        instruction=instruction,
        agent_steps=agent_steps,
        agent_task_dir=traj_dir / task_id,
        include_response=not args.without_response,
    )
    try:
        client = OpenAI()
        max_attempts = max(1, int(args.max_attempts))
        attempts: list[dict[str, Any]] = []
        final_pass = True

        for attempt_idx in range(1, max_attempts + 1):
            resp = client.responses.create(
                model=args.model,
                input=[
                    {
                        "role": "user",
                        "content": user_content,
                    }
                ],
                service_tier="flex",
            )
            raw = (resp.output_text or "").strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"raw_output": raw, "parse_error": True}

            agent_pass = parsed.get("agent_pass")
            is_pass = bool(agent_pass is True)
            attempts.append(
                {
                    "attempt": attempt_idx,
                    "agent_pass": is_pass,
                    "result": parsed,
                }
            )

            if not is_pass:
                final_pass = False
                break

        final_result = attempts[-1]["result"] if attempts else {}
        return {
            "task_id": task_id,
            "instruction": instruction,
            "instruction_config_path": str(instruction_config_path)
            if instruction_config_path
            else None,
            "agent_steps": len(agent_steps),
            "agent_steps_raw": len(agent_steps_raw),
            "traj_path": str(traj_path),
            "max_attempts": max_attempts,
            "num_attempts_run": len(attempts),
            "agent_pass": final_pass,
            "result": final_result,
            "attempt_results": attempts,
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "instruction": instruction,
            "instruction_config_path": str(instruction_config_path)
            if instruction_config_path
            else None,
            "agent_steps": len(agent_steps),
            "agent_steps_raw": len(agent_steps_raw),
            "error": str(e),
        }


def load_cached_results(out_path: Path) -> list[dict[str, Any]]:
    if not out_path.is_file():
        return []
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except Exception:
        return []
    return []


def main() -> None:
    args = parse_args()
    traj_dir: Path = args.traj_dir
    if not traj_dir.is_dir():
        raise SystemExit(f"Trajectory dir not found: {traj_dir}")

    # Check that all tasks in instruction_dir have completed traj.jsonl and result.txt
    if args.instruction_dir.is_dir():
        instruction_task_ids = sorted(
            p.stem for p in args.instruction_dir.iterdir() if p.suffix == ".json"
        )
        incomplete: list[str] = []
        for tid in instruction_task_ids:
            task_dir = traj_dir / tid
            missing = [
                f
                for f in ("traj.jsonl", "result.txt")
                if not (task_dir / f).exists()
            ]
            if missing:
                incomplete.append(f"{tid}: missing {', '.join(missing)}")
        if incomplete:
            msg = (
                f"[Error] {len(incomplete)} task(s) in instruction_dir have not completed:\n"
                + "\n".join(f"  {line}" for line in incomplete)
            )
            raise SystemExit(msg)

    task_ids = sorted([p.name for p in traj_dir.iterdir() if p.is_dir()])
    if args.limit and args.limit > 0:
        task_ids = task_ids[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cached_map: dict[str, dict[str, Any]] = {}
    if args.use_cache:
        cached_results = load_cached_results(args.out)
        cached_map = {
            str(item.get("task_id")): item
            for item in cached_results
            if item.get("task_id") is not None
        }

    all_results: list[dict[str, Any]] = []
    uncached_task_ids = [task_id for task_id in task_ids if task_id not in cached_map]
    if cached_map:
        all_results.extend(cached_map[task_id] for task_id in task_ids if task_id in cached_map)

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        futures = {
            ex.submit(
                evaluate_one_task,
                task_id,
                args=args,
                traj_dir=traj_dir,
            ): task_id
            for task_id in uncached_task_ids
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Verify tasks", unit="task"):
            all_results.append(fut.result())
            # Incremental cache flush: resume-friendly during parallel runs.
            all_results.sort(key=lambda x: x.get("task_id", ""))
            args.out.write_text(
                json.dumps(all_results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    all_results.sort(key=lambda x: x.get("task_id", ""))
    args.out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] Wrote {args.out} ({len(all_results)} record(s))")


if __name__ == "__main__":
    main()
