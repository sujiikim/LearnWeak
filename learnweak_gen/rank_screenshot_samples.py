#!/usr/bin/env python3
"""
Rank screenshot candidates with GPT-5-mini and keep top-k meaningful shots.

Input JSON format (same as sample_screenshots.json):
[
  "/abs/path/to/shot_1.png",
  "/abs/path/to/shot_2.png",
  ...
]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image

MODEL = "gpt-5-mini"
MAX_IMAGE_PIXELS = 1_000_000
DEFAULT_SELECT_COUNT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use GPT-5-mini to pick the most informative screenshots for domain coverage."
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=False,
        help="JSON file containing screenshot candidate paths (typically 20).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Output JSON path (default: <input-json parent>/final_screenshots.json). "
            "Saved as a JSON array of selected screenshot paths."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help=f"Model name (default: {MODEL}).",
    )
    parser.add_argument(
        "--select-count",
        type=int,
        default=DEFAULT_SELECT_COUNT,
        help="How many screenshots to select (default: 10).",
    )

    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=MAX_IMAGE_PIXELS,
        help="Downscale images over this pixel count before upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print only the text prompt that would be sent to the model and exit "
            "without encoding images or calling the API."
        ),
    )
    return parser.parse_args()


def load_screenshot_paths(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list of file path strings.")

    out: list[str] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Input item at index {idx} is not a non-empty string.")
        out.append(item.strip())
    return out


def encode_image(path: Path, max_pixels: int) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Screenshot file not found: {path}")

    ext = path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"

    with Image.open(path) as img:
        width, height = img.size
        total_pixels = width * height
        if total_pixels > max_pixels:
            scale = (max_pixels / float(total_pixels)) ** 0.5
            resized = (max(1, int(width * scale)), max(1, int(height * scale)))
            img = img.resize(resized, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        fmt = "PNG" if mime == "image/png" else "JPEG"
        if fmt == "JPEG" and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(buffer, format=fmt, quality=90, optimize=True)
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def strip_json_fence(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*\n?", text, flags=re.IGNORECASE)
    if match:
        text = text[match.end() :]
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3].rstrip()
    return text.strip()


def parse_model_json(raw_text: str) -> dict[str, Any]:
    cleaned = strip_json_fence(raw_text)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Model output is not a JSON object.")
    return parsed


def build_prompt(select_count: int) -> str:
    return (
        "You are evaluating screenshots from a single software domain.\n"
        "Goal: select the screenshots that maximize understanding of the domain's "
        "features and UI components.\n\n"
        "You will receive candidate screenshots in this pattern:\n"
        "Image 0: <image>\n"
        "Image 1: <image>\n"
        "...\n"
        "Use the number in each 'Image N' label as the index.\n\n"
        f"Select exactly {select_count} screenshots.\n"
        "Prioritize:\n"
        "1) Coverage of distinct major features/workflows.\n"
        "2) Diversity of visible UI components/layout states.\n"
        "3) Informational richness (settings/panels/dialogs/menus/output views).\n"
        "Avoid near-duplicates and low-information transitional frames.\n\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "selected_indices": [int, ...],\n'
        '  "reasons": [\n'
        "    {\n"
        '      "index": int,\n'
        '      "reason": "short reason focused on coverage value"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        f"- selected_indices length must be exactly {select_count}.\n"
        "- selected_indices must contain unique integers only.\n"
        "- reasons length must match selected_indices length.\n"
        "- Each reasons[i].index must be in selected_indices.\n"
    )


def request_ranking(
    client: OpenAI,
    model: str,
    screenshot_paths: list[str],
    select_count: int,
    max_pixels: int,
) -> dict[str, Any]:
    prompt = build_prompt(select_count)
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]

    for idx, raw_path in enumerate(screenshot_paths):
        uri = encode_image(Path(raw_path), max_pixels=max_pixels)
        content.append({"type": "input_text", "text": f"Image {idx}:"})
        content.append({"type": "input_image", "image_url": uri})

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        service_tier="flex",
    )
    output_text = (response.output_text or "").strip()
    return parse_model_json(output_text)


def validate_selection(
    result: dict[str, Any], total: int, select_count: int
) -> tuple[list[int], list[dict[str, str]]]:
    if "selected_indices" not in result:
        raise ValueError("Model output missing 'selected_indices'.")
    selected = result["selected_indices"]
    if not isinstance(selected, list):
        raise ValueError("'selected_indices' must be a list.")
    filtered: list[int] = []
    seen: set[int] = set()
    for i in selected:
        if not isinstance(i, int):
            raise ValueError("'selected_indices' must contain integers only.")
        if 0 <= i < total and i not in seen:
            seen.add(i)
            filtered.append(i)
    # if len(filtered) != select_count:
    #     raise ValueError(
    #         f"'selected_indices' must yield exactly {select_count} unique indices in "
    #         f"[0, {total - 1}] after ignoring out-of-range values; got {len(filtered)}."
    #     )
    selected = filtered

    reasons = result.get("reasons", [])
    if not isinstance(reasons, list):
        raise ValueError("'reasons' must be a list.")
    reason_map: dict[int, str] = {}
    for r in reasons:
        if not isinstance(r, dict):
            continue
        idx = r.get("index")
        txt = r.get("reason")
        if isinstance(idx, int) and isinstance(txt, str) and txt.strip():
            reason_map[idx] = txt.strip()

    normalized_reasons: list[dict[str, str]] = []
    for i in selected:
        normalized_reasons.append(
            {"index": str(i), "reason": reason_map.get(i, "")}
        )
    return selected, normalized_reasons


def main() -> None:
    args = parse_args()
    if args.select_count <= 0:
        raise SystemExit("--select-count must be positive.")
    if args.dry_run:
        print(build_prompt(args.select_count))
        return
    if args.input_json is None:
        raise SystemExit("--input-json is required unless --dry-run is set.")

    screenshot_paths = load_screenshot_paths(args.input_json)
    total = len(screenshot_paths)
    output_path = (
        args.output_json
        if args.output_json is not None
        else args.input_json.parent / "final_screenshots.json"
    )

    if total == 0:
        raise SystemExit("No screenshot paths found in input JSON.")
    if args.select_count > total:
        raise SystemExit(
            f"--select-count ({args.select_count}) cannot exceed input count ({total})."
        )

    print(f"[INFO] Loaded {total} screenshot candidates.")
    print(f"[INFO] Requesting top {args.select_count} via {args.model} ...")

    client = OpenAI()
    model_result = request_ranking(
        client=client,
        model=args.model,
        screenshot_paths=screenshot_paths,
        select_count=args.select_count,
        max_pixels=args.max_image_pixels,
    )
    selected_indices, _ = validate_selection(
        model_result, total=total, select_count=args.select_count
    )

    selected_paths = [screenshot_paths[i] for i in selected_indices]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(selected_paths, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[INFO] Saved selection to: {output_path}")
    for rank, (idx, pth) in enumerate(zip(selected_indices, selected_paths), start=1):
        print(f"{rank:02d}. idx={idx} | {pth}")


if __name__ == "__main__":
    main()
