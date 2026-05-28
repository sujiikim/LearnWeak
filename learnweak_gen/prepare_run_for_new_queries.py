#!/usr/bin/env python3
"""
Build evaluation example JSONs from new_queries_per_config*.json outputs.

For each entry in per_config[].parsed.queries, loads the referenced manual VS Code
config JSON, sets id=task{n}, config_id from reference_config_id, instruction from
the query, and writes one file per task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _iter_queries_with_config_path(data: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Yield (config_file_path, reference_config_id, query_dict) in file order."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for block in data.get("per_config") or []:
        cfg_path = block.get("config_file")
        parsed = block.get("parsed") or {}
        queries = parsed.get("queries") or []
        if not cfg_path or not queries:
            continue
        for q in queries:
            rid = q.get("reference_config_id") or block.get("config_id")
            if not rid:
                continue
            out.append((str(cfg_path), str(rid), q))
    return out


def _merge_task(
    base: dict[str, Any],
    task_id: str,
    config_id: str,
    instruction: str,
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "id": task_id,
        "config_id": config_id,
        "instruction": instruction,
    }
    for key, val in base.items():
        if key == "id":
            continue
        merged[key] = val
    return merged


def main() -> None:
    step_num = 1
    repo = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain to process.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="new_queries_per_config JSON files (in order). Default: nofail then full.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write task{n}.json files.",
    )
    parser.add_argument(
        "--test-json",
        type=Path,
        required=True,
        help="Path to test_{domain}.json listing task stems (no .json).",
    )
    args = parser.parse_args()

    domain = args.domain

    # default_nofail = (
    #     repo
    #     / f"results/dataset_generation/{domain}/opencua-7b_synthetic_step{step_num}/"
    #     "pyautogui/screenshot/vllm_opencua/new_queries_per_config_nofail.json"
    # )
    # default_full = (
    #     repo
    #     / f"results/dataset_generation/{domain}/opencua-7b_synthetic_step{step_num}/"
    #     "pyautogui/screenshot/vllm_opencua/new_queries_per_config.json"
    # )
    # default_out = repo / f"evaluation_examples/synthetic/iter{step_num+1}/examples/{domain}"
    # default_test = repo / f"evaluation_examples/synthetic/iter{step_num+1}/test_{domain}.json"

    inputs = [p.expanduser().resolve() for p in args.inputs]
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect (config_path, config_id, instruction) from all inputs in order.
    rows: list[tuple[str, str, str]] = []
    for inp in inputs:
        if not inp.is_file():
            raise FileNotFoundError(f"Input JSON not found: {inp}")
        with inp.open(encoding="utf-8") as f:
            data = json.load(f)
        for cfg_path, ref_id, q in _iter_queries_with_config_path(data):
            instr = q.get("instruction")
            if not instr:
                continue
            rows.append((cfg_path, ref_id, instr))

    task_names: list[str] = []
    for i, (cfg_path, config_id, instruction) in enumerate(rows, start=1):
        task_id = f"task{i}"
        stem = task_id
        task_names.append(stem)

        cfg_file = Path(cfg_path)
        if not cfg_file.is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_file}")

        with cfg_file.open(encoding="utf-8") as f:
            base = json.load(f)

        merged = _merge_task(base, task_id, config_id, instruction)
        out_path = out_dir / f"{stem}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
            f.write("\n")

    test_payload = {domain: task_names}
    test_path = args.test_json.expanduser().resolve()
    test_path.parent.mkdir(parents=True, exist_ok=True)
    with test_path.open("w", encoding="utf-8") as f:
        json.dump(test_payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {len(task_names)} tasks under {out_dir}")
    print(f"Wrote {test_path}")


if __name__ == "__main__":
    main()
