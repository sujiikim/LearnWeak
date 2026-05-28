#!/usr/bin/env python3
"""
Generate new GUI training queries per docker config with gpt-5-mini (domain-agnostic).

For each config JSON under a target directory:
1) **Docker env specs** come from ``--configs-dir`` (defaults in this repo point at the bundled ``vs_code`` example — ``config*.json`` only).
   **Prior task instructions** come from ``--prior-instructions-dir`` (defaults point at ``synthetic/seed/examples/vs_code``). These two directories are different; do not merge them.
   Each prior ``*.json`` may expose ``instruction`` at the root and/or under ``parsed.queries[]`` / ``per_config[].parsed.queries[]`` (e.g. generated bundles).
2) Load teacher-pass/student-fail gap analysis report.
3) Include the current config's setup and provide_info (joined by "\\n").
4) Attach domain screenshots listed in final_screenshots.json as image inputs.
5) Ask gpt-5-mini to propose new training instructions targeting student gaps.

Use ``--dry-run`` to print each config's text prompt only (no API call, no output file).

API calls for different configs run in parallel (``--workers``); screenshots are encoded once and shared.
Progress bars use ``tqdm``.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

MODEL = "gpt-5-mini"
MAX_IMAGE_PIXELS = 750_000

_DEFAULT_EVAL_ROOT = Path(
    os.environ.get("OSWORLD_EVALUATION_EXAMPLES", "/c2/kangsan/OSWorld/evaluation_examples")
)
# Docker/container configs (config0.json …) — separate from seed task instructions.
DEFAULT_CONFIGS_DIR = _DEFAULT_EVAL_ROOT / "synthetic" / "manual" / "libreoffice_impress"
DEFAULT_PRIOR_INSTRUCTIONS_DIR = (
    _DEFAULT_EVAL_ROOT / "synthetic" / "seed" / "examples" / "libreoffice_impress"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate new GUI training queries per docker config with gpt-5-mini (any domain)."
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="",
        help=(
            "Optional short label for prompts and output meta (e.g. vs_code, vlc). "
            "Empty = fully domain-neutral wording; config + screenshots define the app."
        ),
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=DEFAULT_CONFIGS_DIR,
        help=(
            "Directory of config*.json docker env specs (default: bundled vs_code example path). "
            "Not the same folder as prior instructions."
        ),
    )
    parser.add_argument(
        "--prior-instructions-dir",
        type=Path,
        default=DEFAULT_PRIOR_INSTRUCTIONS_DIR,
        help=(
            "Directory of *.json files with prior task instructions (default: "
            "synthetic/seed/examples/vs_code). Mined fields: root `instruction`, "
            "`parsed.queries[].instruction`, `per_config[].parsed.queries[].instruction`. "
            "Separate from --configs-dir."
        ),
    )
    parser.add_argument(
        "--fail-report",
        type=Path,
        default=None,
        help="Path to teacher_pass_student_fail_report.json.",
    )
    parser.add_argument(
        "--no-fail-report",
        action="store_true",
        help=(
            "Do not use student gap analysis. Generate diverse screenshot-driven tasks "
            "that are maximally different from prior instructions."
        ),
    )
    parser.add_argument(
        "--final-screenshots",
        "--final-screenshot",
        type=Path,
        dest="final_screenshots",
        required=True,
        help="Path to final_screenshots.json (JSON list of absolute image paths).",
    )
    parser.add_argument(
        "--queries-per-config",
        type=int,
        default=3,
        help="How many new instructions to generate for each config file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help=f"Model name (default: {MODEL}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <final-screenshots directory>/new_queries_per_config.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the text prompt for each config only, then exit (no API, no output file).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Parallel OpenAI calls for configs (default: 0 = min(config_count, 12); "
            "1 = serial). Capped by number of configs."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def encode_image(path: Path, max_pixels: int = MAX_IMAGE_PIXELS) -> str | None:
    if not path.exists():
        return None
    ext = path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with Image.open(path) as img:
        w, h = img.size
        total = w * h
        if total > max_pixels:
            scale = (max_pixels / float(total)) ** 0.5
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        fmt = "PNG" if mime == "image/png" else "JPEG"
        if fmt == "JPEG" and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(buf, format=fmt, quality=90, optimize=True)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{data}"


def strip_json_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?", text, re.IGNORECASE)
    if m:
        text = text[m.end() :]
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3].rstrip()
    return text.strip()


def parse_model_json(raw: str) -> dict[str, Any]:
    cleaned = strip_json_fence(raw)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Model output is not a JSON object.")
    return parsed


def load_config_files(configs_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    files = sorted(configs_dir.glob("config*.json"))
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        payload = load_json(path)
        if isinstance(payload, dict):
            loaded.append((path, payload))
    return loaded


def _normalize_instruction(text: str) -> str:
    return " ".join(text.split())


def _instructions_from_parsed_block(parsed: dict[str, Any]) -> list[str]:
    queries = parsed.get("queries")
    if not isinstance(queries, list):
        return []
    out: list[str] = []
    for item in queries:
        if not isinstance(item, dict):
            continue
        inst = item.get("instruction")
        if isinstance(inst, str) and inst.strip():
            out.append(_normalize_instruction(inst))
    return out


def _extract_instructions_from_json_dict(payload: dict[str, Any]) -> list[str]:
    """
    Support:
    - Root task files: { "instruction": "..." }
    - Generator output: { "parsed": { "queries": [ { "instruction": "..." } ] } }
    - Same with per_config: { "per_config": [ { "parsed": { "queries": ... } } ] }
    """
    found: list[str] = []
    inst = payload.get("instruction")
    if isinstance(inst, str) and inst.strip():
        found.append(_normalize_instruction(inst))

    parsed = payload.get("parsed")
    if isinstance(parsed, dict):
        found.extend(_instructions_from_parsed_block(parsed))

    per_cfg = payload.get("per_config")
    if isinstance(per_cfg, list):
        for block in per_cfg:
            if not isinstance(block, dict):
                continue
            p = block.get("parsed")
            if isinstance(p, dict):
                found.extend(_instructions_from_parsed_block(p))
    return found


def load_prior_instructions_from_dir(prior_dir: Path) -> list[str]:
    """
    Load instruction strings from every *.json under prior_dir.
    Files are processed in sorted name order; duplicate instruction texts are dropped
    (``parsed`` and ``per_config`` in the same file can overlap).
    """
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"--prior-instructions-dir is not a directory: {prior_dir}")
    seen: set[str] = set()
    ordered: list[str] = []
    for path in sorted(prior_dir.glob("*.json")):
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for s in _extract_instructions_from_json_dict(payload):
            if s not in seen:
                seen.add(s)
                ordered.append(s)
    return ordered


def merge_optional_instructions_from_configs(
    configs: list[tuple[Path, dict[str, Any]]],
    existing: list[str],
) -> list[str]:
    """Append instructions from config JSONs if present (dedupe, preserve order)."""
    seen = set(existing)
    merged = list(existing)
    for _, payload in configs:
        instruction = payload.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        norm = " ".join(instruction.split())
        if norm not in seen:
            seen.add(norm)
            merged.append(norm)
    return merged


# Paths like /home/user/.../file.py (used in docker configs and provide_info headers).
_PATH_IN_TEXT_RE = re.compile(r"/home/user(?:/[\w.\-+]+)+(?:\.[a-zA-Z0-9]+)?")


def _extract_paths_from_config_entries(entries: Any) -> list[str]:
    """Collect filesystem paths implied by launch/download/command steps."""
    found: list[str] = []
    if not isinstance(entries, list):
        return found
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        etype = entry.get("type")
        params = entry.get("parameters")
        if not isinstance(params, dict):
            continue
        if etype == "download":
            files = params.get("files")
            if isinstance(files, list):
                for f in files:
                    if isinstance(f, dict):
                        p = f.get("path")
                        if isinstance(p, str) and p.strip():
                            found.append(p.strip())
        elif etype == "launch":
            cmd = params.get("command")
            if isinstance(cmd, list) and len(cmd) >= 2:
                for arg in cmd[1:]:
                    if isinstance(arg, str) and arg.strip():
                        a = arg.strip()
                        if a.startswith("/") or a.startswith("~"):
                            found.append(a)
        elif etype in ("command", "execute"):
            cmd = params.get("command")
            if isinstance(cmd, list):
                joined = " ".join(str(x) for x in cmd)
                found.extend(_PATH_IN_TEXT_RE.findall(joined))
    return found


def _extract_paths_from_provide_info_lines(lines: list[str]) -> list[str]:
    text = "\n".join(lines)
    return _PATH_IN_TEXT_RE.findall(text)


def _launch_has_no_path_arguments(entries: Any) -> bool:
    """True if some launch step starts only the app executable (no file/folder path args)."""
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") != "launch":
            continue
        params = entry.get("parameters")
        if not isinstance(params, dict):
            continue
        cmd = params.get("command")
        if isinstance(cmd, list) and len(cmd) == 1:
            return True
    return False


def build_workspace_grounding_text(
    *,
    config_payload: dict[str, Any],
    provide_lines: list[str],
) -> str:
    """
    Human-readable rules so generated tasks do not name repos/files absent from this setup.
    """
    entries = config_payload.get("config")
    from_config = _extract_paths_from_config_entries(entries)
    from_info = _extract_paths_from_provide_info_lines(provide_lines)
    declared = sorted({*from_config, *from_info})
    launch_no_paths = _launch_has_no_path_arguments(entries)

    lines: list[str] = []
    lines.append(
        "Generated instructions MUST match this docker setup. **Screenshots may show another "
        "user session, window layout, or file tree** that does not exist in this container. "
        "Treat **only** the declared paths below plus UI surfaces that are consistent with "
        "sections 4–6 as ground truth. "
        "Do **not** invent project, repository, document, or folder names unless they appear "
        "as a path segment or basename below."
        "However, you don't need to use the paths in the instructions. You can generate queries that are not grounded in given paths."
    )
    lines.append("")

    if not declared:
        lines.append(
            "- **No file or folder path is declared** by downloads/opens in this config JSON "
            "(or `provide_info` is empty)."
        )
        if launch_no_paths:
            lines.append(
                "- A launch step starts **only** the application executable with **no** file or "
                "folder argument — expect a default empty window, welcome screen, or starter "
                "state, not a specific on-disk project unless the UI clearly matches that."
            )
    else:
        lines.append(
            "- **Declared paths**"
            "You may also refer to a directory by its **basename** when it "
            "matches a declared path (e.g. `Workspace` for `/home/user/Workspace`)."
        )
        for p in declared:
            lines.append(f"  - `{p}`")
        lines.append(
            "- Do not require files or folders outside this set unless the task is purely "
            "about global application UI (no specific on-disk path)."
        )

    lines.append(
        "- In `rationale`, briefly note which declared path or generic UI surface the task uses."
    )
    return "\n".join(lines)


def build_prompt(
    *,
    config_path: Path,
    config_payload: dict[str, Any],
    prior_instructions: list[str],
    gap_analysis: Any | None,
    use_fail_report: bool,
    queries_per_config: int,
    domain_label: str = "",
) -> str:
    cfg_id = config_payload.get("id", config_path.stem)
    config_block = json.dumps(config_payload.get("config", []), ensure_ascii=False, indent=2)

    provide_info = config_payload.get("provide_info", [])
    provide_lines: list[str] = []
    if isinstance(provide_info, list):
        for item in provide_info:
            if isinstance(item, str) and item.strip():
                provide_lines.append(item)
    provide_info_joined = "\n".join(provide_lines).strip()

    prior_block = "\n".join(f"- {s}" for s in prior_instructions)
    gap_block = (
        json.dumps(gap_analysis, ensure_ascii=False, indent=2)
        if gap_analysis is not None
        else "{}"
    )

    if not provide_info_joined:
        provide_info_joined = "(empty)"

    workspace_block = build_workspace_grounding_text(
        config_payload=config_payload,
        provide_lines=provide_lines,
    )

    if use_fail_report:
        goal_block = (
            "Goal:\n"
            "- Propose new task instructions that specifically improve abilities where the student still fails.\n"
            "- Keep tasks realistic for the given config environment.\n"
            "- **Obey the Workspace / path contract (section 3)** — do not name repos, folders, or files that this docker setup does not open or download.\n"
            "- While instructions are grounded in the config and failure analysis, they should be diverse and distinct from the prior instructions.\n"
            "- Do not duplicate or lightly paraphrase the prior instructions. Use different features, workflows, and subtasks.\n"
            "- Tasks should not be a tutorial or a step-by-step guide. They should be a easy and concise end-user request the agent must figure out how to execute.\n",
            "- Tasks should be possible to finish in a few steps even for a beginner. Do not include more than two sub-tasks in each instruction."
        )
        section2 = (
            "2) Student weakness analysis (teacher pass, student fail):\n"
            f"{gap_block}\n\n"
        )
        requirement_gap = "- Must target one or more weak abilities from the analysis.\n"
        query_object_schema = (
            '      "reference_config_id": "string",\n'
            '      "instruction": "string",\n'
            '      "targets_student_gaps": ["string"],\n'
            '      "rationale": "one short sentence"\n'
        )
    else:
        goal_block = (
            "Goal:\n"
            "- Propose new task instructions by exploring what appears in screenshots.\n"
            "- Prioritize new features/workflows/subtasks not present in prior instructions.\n"
            "- Keep tasks easy, short, and realistic for the given config environment.\n"
            "- **Section 3 (Workspace / path contract) overrides screenshots:** a file browser or tree may show a sample project that is **not** part of this docker config — do not name it.\n"
            "- While instructions are grounded in the config and the screenshots, they should be diverse and distinct from the prior instructions.\n"
            "- Do not duplicate or lightly paraphrase the prior instructions. Use different features, workflows, and subtasks.\n"
            "- Tasks should not be a tutorial or a step-by-step guide. They should be a easy and concise end-user request the agent must figure out how to execute.\n\n"
            "- Tasks should be possible to finish in a few steps even for a beginner. Do not include more than two simple sub-tasks in each instruction. Instructions should be short, easy, and concise."
        )
        section2 = (
            "2) Student weakness analysis:\n"
            "(Not used in this run. Use screenshots + sections 3–6; never invent on-disk projects.)\n\n"
        )
        requirement_gap = (
            "- Must maximize diversity and novelty versus prior instructions (new functionality/subtasks/workflows).\n"
            "- Each query object must include **only** `reference_config_id`, `instruction`, and `rationale`. "
            "**Do not** include `targets_student_gaps` (not even as an empty array).\n"
        )
        query_object_schema = (
            '      "reference_config_id": "string",\n'
            '      "instruction": "a short string",\n'
            '      "rationale": "one short sentence"\n'
        )
        # f"{prior_block}\n\n"
    return (
        f"{goal_block}"
        "Input context:\n"
        "1) Prior instructions already used (avoid overlap/paraphrase):\n"
        f"{prior_block}\n\n"
        f"{section2}"
        "3) Workspace / path contract (mandatory — read before writing tasks):\n"
        f"{workspace_block}\n\n"
        f"4) Current docker `config` array to target (config id: {cfg_id}):\n"
        f"{config_block}\n\n"
        "5) Extra file/folder/code context from this config (`provide_info`):\n"
        f"{provide_info_joined}\n\n"
        "Requirements:\n"
        f"- Generate exactly {queries_per_config} instructions.\n"
        "- Each instruction must be concise end-user style English.\n"
        "- Do **not** include more than two simple and easy sub-tasks in each instruction. Do not generate too complex and long instructions with multiple sub-tasks. Make it as **simple and easy** as possible.\n"
        "- **Every instruction MUST satisfy section 3** (no fictional repositories or paths).\n"
        f"{requirement_gap}"
        "- Must be feasible with this config and attached context. Instructions should be easy to execute even for a beginner.\n"
        "- It is not mandatory to use the files, folders, and paths in the config. You can generate queries that are not grounded in given paths."
        "- Do not copy or lightly paraphrase prior instructions. Generate as diverse as possible while keeping the tasks simple and easy.\n"
        "- Instructions should be **less than 15 words long.** And generated tasks are **significantly different from each other.**\n\n"
        "Return STRICT JSON only:\n"
        "{\n"
        '  "queries": [\n'
        "    {\n"
        f"{query_object_schema}"
        "    }\n"
        "  ]\n"
        "}\n"
    )


def _strip_targets_student_gaps_when_no_fail_report(
    use_fail_report: bool, queries: list[Any]
) -> None:
    """When there is no gap report, output must not contain ``targets_student_gaps``."""
    if use_fail_report:
        return
    for item in queries:
        if isinstance(item, dict) and "targets_student_gaps" in item:
            del item["targets_student_gaps"]


def prebuild_screenshot_image_parts(
    screenshot_paths: list[str],
) -> tuple[list[dict[str, Any]], int]:
    """
    Encode each screenshot once; reuse across all config API calls (parallel-safe).
    """
    parts: list[dict[str, Any]] = []
    attached = 0
    paths_iter = screenshot_paths
    if screenshot_paths:
        paths_iter = tqdm(
            screenshot_paths,
            desc="Screenshots",
            unit="img",
            leave=False,
        )
    for raw in paths_iter:
        uri = encode_image(Path(raw))
        if not uri:
            continue
        parts.append({"type": "input_image", "image_url": uri})
        attached += 1
    return parts, attached


def _resolve_worker_count(requested: int, n_configs: int) -> int:
    if n_configs <= 0:
        return 1
    if requested <= 0:
        return max(1, min(n_configs, 12))
    return max(1, min(requested, n_configs))


def _process_one_config_parallel(
    index: int,
    config_path: Path,
    payload: dict[str, Any],
    *,
    prior_instructions: list[str],
    gap_analysis: Any | None,
    use_fail_report: bool,
    queries_per_config: int,
    domain_label: str,
    image_parts: list[dict[str, Any]],
    client: OpenAI,
    model: str,
) -> tuple[int, dict[str, Any], list[Any], list[str]]:
    """One config: build prompt, call API, parse. Returns (index, per_config dict, queries, errors)."""
    cfg_id = str(payload.get("id", config_path.stem))
    prompt = build_prompt(
        config_path=config_path,
        config_payload=payload,
        prior_instructions=prior_instructions,
        gap_analysis=gap_analysis,
        use_fail_report=use_fail_report,
        queries_per_config=queries_per_config,
        domain_label=domain_label,
    )
    content = [{"type": "input_text", "text": prompt}] + image_parts
    screenshots_attached = len(image_parts)

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        service_tier="flex",
    )
    output_text = (response.output_text or "").strip()

    parsed: dict[str, Any] | None = None
    parse_error: str | None = None
    queries: list[Any] = []
    errors: list[str] = []

    try:
        parsed = parse_model_json(output_text)
        q = parsed.get("queries")
        if isinstance(q, list):
            queries = q
            _strip_targets_student_gaps_when_no_fail_report(use_fail_report, queries)
    except Exception as exc:
        parse_error = str(exc)
        errors.append(f"{cfg_id}: parse_error={parse_error}")

    if len(queries) != queries_per_config:
        errors.append(
            f"{cfg_id}: expected {queries_per_config} queries, got {len(queries)}"
        )

    for item in queries:
        if isinstance(item, dict):
            item.setdefault("reference_config_id", cfg_id)

    per_config_result = {
        "config_file": str(config_path),
        "config_id": cfg_id,
        "screenshots_attached": screenshots_attached,
        "prompt": prompt,
        "raw_model_output": output_text,
        "parsed": parsed,
        "parse_error": parse_error,
    }
    return index, per_config_result, queries, errors


def main() -> None:
    args = parse_args()
    if args.queries_per_config <= 0:
        raise SystemExit("--queries-per-config must be positive.")

    configs = load_config_files(args.configs_dir)
    if not configs:
        raise SystemExit(f"No config*.json found in: {args.configs_dir}")

    use_fail_report = not args.no_fail_report
    gap_analysis: Any | None = None
    if use_fail_report:
        if args.fail_report is None:
            raise SystemExit("--fail-report is required unless --no-fail-report is used.")
        if not args.fail_report.exists():
            raise SystemExit(f"--fail-report file not found: {args.fail_report}")
        report = load_json(args.fail_report)
        if not isinstance(report, dict):
            report = {"raw_report": report}
        gap_analysis = report.get("gpt_analysis", report)

    screenshots_payload = load_json(args.final_screenshots)
    if not isinstance(screenshots_payload, list):
        raise SystemExit("--final-screenshots JSON must be a list of absolute paths.")
    screenshot_paths = [p for p in screenshots_payload if isinstance(p, str) and p.strip()]

    try:
        prior_instructions = load_prior_instructions_from_dir(args.prior_instructions_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    prior_instructions = merge_optional_instructions_from_configs(configs, prior_instructions)
    if not prior_instructions:
        raise SystemExit(
            "No prior instructions found under --prior-instructions-dir. Add JSON files with "
            "root `instruction`, or `parsed.queries[].instruction` / "
            "`per_config[].parsed.queries[].instruction` (see script defaults for an example layout)."
        )

    if args.dry_run:
        n_shots = len(screenshot_paths)
        print(
            f"[DRY RUN] {len(configs)} config(s); would attach {n_shots} screenshot(s) from "
            f"{args.final_screenshots} (images not loaded).\n",
            flush=True,
        )
        dry_iter = tqdm(configs, desc="Dry-run prompts", unit="cfg")
        for config_path, payload in dry_iter:
            cfg_id = str(payload.get("id", config_path.stem))
            prompt = build_prompt(
                config_path=config_path,
                config_payload=payload,
                prior_instructions=prior_instructions,
                gap_analysis=gap_analysis,
                use_fail_report=use_fail_report,
                queries_per_config=args.queries_per_config,
                domain_label=args.domain,
            )
            print(
                f"======== DRY RUN: {cfg_id} ({config_path.name}) ========\n",
                flush=True,
            )
            print(prompt, end="" if prompt.endswith("\n") else "\n", flush=True)
        return

    client = OpenAI()
    image_parts, screenshots_encoded = prebuild_screenshot_image_parts(screenshot_paths)
    n_workers = _resolve_worker_count(args.workers, len(configs))

    futures = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for idx, (config_path, payload) in enumerate(configs):
            futures.append(
                executor.submit(
                    _process_one_config_parallel,
                    idx,
                    config_path,
                    payload,
                    prior_instructions=prior_instructions,
                    gap_analysis=gap_analysis,
                    use_fail_report=use_fail_report,
                    queries_per_config=args.queries_per_config,
                    domain_label=args.domain,
                    image_parts=image_parts,
                    client=client,
                    model=args.model,
                )
            )
        results: list[
            tuple[int, dict[str, Any], list[Any], list[str]]
        ] = []
        done_iter = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="OpenAI configs",
            unit="cfg",
        )
        for fut in done_iter:
            results.append(fut.result())

    results.sort(key=lambda x: x[0])
    per_config_results: list[dict[str, Any]] = []
    merged_queries: list[Any] = []
    all_errors: list[str] = []
    for _idx, pcr, queries, errs in results:
        per_config_results.append(pcr)
        merged_queries.extend(queries)
        all_errors.extend(errs)

    if args.out is not None:
        out_path = args.out
    else:
        filename = (
            "new_queries_per_config_nofail.json"
            if args.no_fail_report
            else "new_queries_per_config.json"
        )
        out_path = args.final_screenshots.parent / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "meta": {
            "model": args.model,
            "domain": args.domain.strip() or None,
            "configs_dir": str(args.configs_dir),
            "prior_instructions_dir": str(args.prior_instructions_dir),
            "use_fail_report": use_fail_report,
            "fail_report_path": str(args.fail_report) if args.fail_report else None,
            "final_screenshots_path": str(args.final_screenshots),
            "queries_per_config": args.queries_per_config,
            "parallel_workers": n_workers,
            "config_count": len(configs),
            "prior_instruction_count": len(prior_instructions),
            "screenshots_used_count": len(screenshot_paths),
            "screenshot_images_encoded_once": screenshots_encoded,
            "merged_query_count": len(merged_queries),
        },
        "per_config": per_config_results,
        "parsed": {"queries": merged_queries},
        "errors": all_errors,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] wrote: {out_path}")
    if all_errors:
        print("[WARN] issues detected:")
        for err in all_errors:
            print(f"- {err}")


if __name__ == "__main__":
    main()
