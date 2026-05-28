#!/usr/bin/env python3
"""Select the most diverse screenshot samples from two vs_code folders.

Pipeline:
1) Recursively collect image files from input folders.
2) Extract image embeddings with CLIP (optional multi-process: one model per worker).
3) Build kNN distances in embedding space.
4) Rank by kNN uniqueness score and run farthest-point sampling
   to pick the final diverse set.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, TypeVar

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

_T = TypeVar("_T")


def _default_load_workers() -> int:
    n = os.cpu_count() or 8
    return max(1, min(16, n))


def _load_image_rgb(path_str: str) -> tuple[str, np.ndarray | None, str]:
    """Worker: read one image; return (path, RGB uint8 array or None, error message)."""
    try:
        with Image.open(path_str) as im:
            arr = np.asarray(im.convert("RGB"))
        return (path_str, arr, "")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        return (path_str, None, str(e))

# INPUT_FOLDERS = [
#     "/c2/kangsan/OSWorld/results/dataset_generation/vs_code/opencua-7b_synthetic_step1/pyautogui/screenshot/vllm_opencua/vs_code",
#     "/c2/kangsan/OSWorld/results/dataset_generation/vs_code/evocua-32b_synthetic_step1/pyautogui/screenshot/meituan/EvoCUA-32B-20260105/vs_code",
#     "/c2/kangsan/OSWorld/results/dataset_generation/vs_code/opencua-7b_synthetic_step2/pyautogui/screenshot/vllm_opencua/vs_code",
#     "/c2/kangsan/OSWorld/results/dataset_generation/vs_code/evocua-32b_synthetic_step2/pyautogui/screenshot/meituan/EvoCUA-32B-20260105/vs_code",
#     "/c2/kangsan/OSWorld/results/dataset_generation/vs_code/opencua-7b_synthetic_step3/pyautogui/screenshot/vllm_opencua/vs_code",
#     "/c2/kangsan/OSWorld/results/dataset_generation/vs_code/evocua-32b_synthetic_step3/pyautogui/screenshot/meituan/EvoCUA-32B-20260105/vs_code",
#     "/c2/kangsan/OSWorld/results/dataset_generation/vs_code/opencua-7b_synthetic_step4/pyautogui/screenshot/vllm_opencua/vs_code",
#     "/c2/kangsan/OSWorld/results/dataset_generation/vs_code/evocua-32b_synthetic_step4/pyautogui/screenshot/meituan/EvoCUA-32B-20260105/vs_code",
#     "/c2/kangsan/OSWorld/results/dataset_generation/vs_code/opencua-7b_synthetic_step5/pyautogui/screenshot/vllm_opencua/vs_code",
#     "/c2/kangsan/OSWorld/results/dataset_generation/vs_code/evocua-32b_synthetic_step5/pyautogui/screenshot/meituan/EvoCUA-32B-20260105/vs_code"
# ]

INPUT_FOLDERS = [
    "/c2/kangsan/OSWorld/results/dataset_generation/libreoffice_impress/opencua-7b_synthetic_step1/pyautogui/screenshot/vllm_opencua/libreoffice_impress",
    "/c2/kangsan/OSWorld/results/dataset_generation/libreoffice_impress/evocua-32b_synthetic_step1/pyautogui/screenshot/meituan/EvoCUA-32B-20260105/libreoffice_impress",
]

@dataclass
class SelectionResult:
    selected_paths: List[str]
    knn_uniqueness_scores: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find top-N most diverse screenshots from two folders."
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=INPUT_FOLDERS,
        help="One or more root folders to recursively search for images.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of diverse screenshots to select.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=20,
        help="k for nearest-neighbor uniqueness scoring.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for image encoding.",
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parallel processes for reading/decoding images before CLIP "
            f"(default: {_default_load_workers()}). Use 1 for sequential I/O."
        ),
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model name.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Backend for CLIP: cpu, or CUDA when available (see --gpu-index).",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        metavar="I",
        help=(
            "When using CUDA, all CLIP processes are pinned to this single GPU index "
            "(default: 0). Ignored with --device cpu."
        ),
    )
    parser.add_argument(
        "--inference-workers",
        type=int,
        default=4,
        metavar="N",
        help=(
            "Number of parallel CLIP processes (each loads its own model). "
            "Use 1 for a single process."
        ),
    )
    parser.add_argument(
        "--inference-devices",
        nargs="+",
        default=None,
        metavar="DEVICE",
        help=(
            "Override devices per worker. Default: every worker uses the same GPU "
            "(--gpu-index, e.g. cuda:0). Provide at least as many entries as active workers."
        ),
    )
    parser.add_argument(
        "--output-json",
        # default="unique_screenshot_samples.json",
        help="Output path for selected samples and scores.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        default=None,
        help=(
            "If set, also save absolute selected image paths to "
            "<output-dir>/sample_screenshots.json."
        ),
    )
    return parser.parse_args()


def collect_images(folders: Sequence[str]) -> List[Path]:
    image_paths: List[Path] = []
    for folder in folders:
        root = Path(folder)
        if not root.exists():
            print(f"[WARN] Folder does not exist: {root}")
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                image_paths.append(p.resolve())
    image_paths = sorted(set(image_paths))
    return image_paths


def build_encoder_and_processor(model_name: str):
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return model, processor


def pick_device(device_arg: str) -> str:
    import torch

    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_single_clip_device(device_arg: str, gpu_index: int) -> str:
    """Pick device for one CLIP process; CUDA is pinned to cuda:{gpu_index}."""
    import torch

    base = pick_device(device_arg)
    if base != "cuda":
        return base
    if not torch.cuda.is_available():
        return "cpu"
    n_gpu = torch.cuda.device_count()
    if gpu_index < 0 or gpu_index >= n_gpu:
        raise SystemExit(
            f"--gpu-index {gpu_index} is out of range (found {n_gpu} CUDA device(s))."
        )
    return f"cuda:{gpu_index}"


def _resolve_inference_devices_auto(
    n_workers: int, device_arg: str, gpu_index: int
) -> List[str]:
    """Assign every inference worker to the same CUDA device (see --gpu-index)."""
    import torch

    if device_arg == "cpu":
        return ["cpu"] * n_workers
    if not torch.cuda.is_available():
        return ["cpu"] * n_workers

    n_gpu = torch.cuda.device_count()
    if n_gpu <= 0:
        return ["cpu"] * n_workers
    if gpu_index < 0 or gpu_index >= n_gpu:
        raise SystemExit(
            f"--gpu-index {gpu_index} is out of range (found {n_gpu} CUDA device(s))."
        )
    dev = f"cuda:{gpu_index}"
    return [dev] * n_workers


def _split_into_n_shards(xs: List[_T], n: int) -> List[List[_T]]:
    """Split xs into exactly n non-empty contiguous shards (n <= len(xs))."""
    if n <= 0:
        return []
    if len(xs) < n:
        raise ValueError("n must be <= len(xs) for non-empty shards")
    base, rem = divmod(len(xs), n)
    out: List[List[_T]] = []
    i = 0
    for j in range(n):
        take = base + (1 if j < rem else 0)
        out.append(xs[i : i + take])
        i += take
    return out


def _decode_batch_sequential(batch_paths: Sequence[Path]) -> tuple[List[Image.Image], List[Path]]:
    batch_imgs: List[Image.Image] = []
    batch_valid: List[Path] = []
    for p in batch_paths:
        try:
            batch_imgs.append(Image.open(p).convert("RGB"))
            batch_valid.append(p)
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
            print(f"[WARN] Skip (unreadable): {p} ({e})")
    return batch_imgs, batch_valid


def _decode_batch_parallel(
    pool,
    batch_paths: Sequence[Path],
    num_workers: int,
) -> tuple[List[Image.Image], List[Path]]:
    path_strs = [str(p) for p in batch_paths]
    n = len(path_strs)
    chunksize = max(1, n // max(1, num_workers * 4))
    results = pool.map(_load_image_rgb, path_strs, chunksize=chunksize)
    batch_imgs: List[Image.Image] = []
    batch_valid: List[Path] = []
    for p, (_ps, arr, err) in zip(batch_paths, results):
        if arr is not None:
            batch_imgs.append(Image.fromarray(arr))
            batch_valid.append(p)
        else:
            print(f"[WARN] Skip (unreadable): {p} ({err})")
    return batch_imgs, batch_valid


def _decode_batch_parallel_threads(
    executor: ThreadPoolExecutor,
    batch_paths: Sequence[Path],
) -> tuple[List[Image.Image], List[Path]]:
    """Parallel decode via threads (safe inside multiprocessing.Pool workers)."""
    path_strs = [str(p) for p in batch_paths]
    results = list(executor.map(_load_image_rgb, path_strs))
    batch_imgs: List[Image.Image] = []
    batch_valid: List[Path] = []
    for p, (_ps, arr, err) in zip(batch_paths, results):
        if arr is not None:
            batch_imgs.append(Image.fromarray(arr))
            batch_valid.append(p)
        else:
            print(f"[WARN] Skip (unreadable): {p} ({err})")
    return batch_imgs, batch_valid


def _encode_batches(
    image_paths: Sequence[Path],
    model,
    processor,
    device: str,
    batch_size: int,
    load_pool,
    load_thread_executor: ThreadPoolExecutor | None,
    num_load_workers: int,
    projection_dim: int,
    tqdm_desc: str,
    show_progress: bool,
) -> tuple[np.ndarray, List[Path]]:
    import torch

    all_embs: List[np.ndarray] = []
    valid_paths: List[Path] = []
    n_paths = len(image_paths)
    n_batches = (n_paths + batch_size - 1) // batch_size if n_paths else 0
    batch_iter = range(0, n_paths, batch_size)
    iterator = (
        tqdm(
            batch_iter,
            total=n_batches,
            desc=tqdm_desc,
            unit="batch",
            mininterval=0.3,
            dynamic_ncols=True,
        )
        if show_progress
        else batch_iter
    )
    for i in iterator:
        batch_paths = image_paths[i : i + batch_size]
        if load_pool is not None:
            batch_imgs, batch_valid = _decode_batch_parallel(
                load_pool, batch_paths, num_load_workers
            )
        elif load_thread_executor is not None:
            batch_imgs, batch_valid = _decode_batch_parallel_threads(
                load_thread_executor, batch_paths
            )
        else:
            batch_imgs, batch_valid = _decode_batch_sequential(batch_paths)
        if not batch_imgs:
            continue
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = torch.nn.functional.normalize(emb, dim=-1)
        all_embs.append(emb.detach().cpu().numpy().astype(np.float32))
        valid_paths.extend(batch_valid)
        for img in batch_imgs:
            img.close()

    if not all_embs:
        return np.zeros((0, projection_dim), dtype=np.float32), []
    return np.concatenate(all_embs, axis=0), valid_paths


def _clip_encode_shard_job(
    job: tuple[int, list[str], str, int, str, int],
) -> tuple[int, np.ndarray, list[str]]:
    """Top-level worker for multiprocessing (must be picklable)."""
    shard_id, path_strs, model_name, batch_size, device_str, shard_load_workers = job
    paths = [Path(p) for p in path_strs]
    model, processor = build_encoder_and_processor(model_name)
    model = model.to(device_str)
    model.eval()
    projection_dim = int(model.config.projection_dim)

    # Pool workers are daemonized; they cannot spawn child processes. Use threads for I/O.
    if shard_load_workers > 1:
        with ThreadPoolExecutor(max_workers=shard_load_workers) as thread_exe:
            feats, valid_paths = _encode_batches(
                paths,
                model,
                processor,
                device_str,
                batch_size,
                None,
                thread_exe,
                shard_load_workers,
                projection_dim,
                tqdm_desc=f"CLIP shard {shard_id}",
                show_progress=True,
            )
    else:
        feats, valid_paths = _encode_batches(
            paths,
            model,
            processor,
            device_str,
            batch_size,
            None,
            None,
            shard_load_workers,
            projection_dim,
            tqdm_desc=f"CLIP shard {shard_id}",
            show_progress=True,
        )

    out_paths = [str(p) for p in valid_paths]
    return shard_id, feats, out_paths


def encode_images(
    image_paths: Sequence[Path],
    model_name: str,
    batch_size: int,
    device_arg: str,
    num_load_workers: int,
    inference_workers: int,
    inference_devices: Sequence[str] | None,
    gpu_index: int,
) -> tuple[np.ndarray, List[Path]]:
    if num_load_workers < 1:
        raise ValueError("num_load_workers must be >= 1")
    if inference_workers < 1:
        raise ValueError("inference_workers must be >= 1")

    path_list = list(image_paths)
    n_img = len(path_list)

    # One process: same behavior as before (single model + optional I/O pool).
    if inference_workers == 1:
        model, processor = build_encoder_and_processor(model_name)
        device = _resolve_single_clip_device(device_arg, gpu_index)
        model = model.to(device)
        model.eval()
        projection_dim = int(model.config.projection_dim)
        print(f"[INFO] CLIP inference: 1 process on {device}")

        load_ctx = mp.get_context("spawn")
        load_pool = None
        if num_load_workers > 1:
            load_pool = load_ctx.Pool(processes=num_load_workers)
        try:
            feats, valid_paths = _encode_batches(
                path_list,
                model,
                processor,
                device,
                batch_size,
                load_pool,
                None,
                num_load_workers,
                projection_dim,
                tqdm_desc="Encoding images",
                show_progress=True,
            )
        finally:
            if load_pool is not None:
                load_pool.close()
                load_pool.join()

        if feats.shape[0] == 0:
            raise SystemExit("No readable images could be encoded.")
        return feats, valid_paths

    # Parallel CLIP: shard paths across processes; each loads its own model.
    eff_workers = min(inference_workers, max(1, n_img))
    if eff_workers < inference_workers:
        print(
            f"[INFO] Using {eff_workers} inference worker(s) "
            f"(capped by image count {n_img})."
        )

    if inference_devices is not None:
        if len(inference_devices) < eff_workers:
            raise SystemExit(
                f"Need at least {eff_workers} --inference-devices entries "
                f"(got {len(inference_devices)})."
            )
        devices = list(inference_devices[:eff_workers])
    else:
        devices = _resolve_inference_devices_auto(eff_workers, device_arg, gpu_index)

    print(
        f"[INFO] CLIP inference device map ({eff_workers} worker(s), same GPU): {devices}"
    )

    path_strs = [str(p) for p in path_list]
    shards = _split_into_n_shards(path_strs, eff_workers)
    per_shard_load = max(1, num_load_workers // eff_workers)

    jobs = [
        (sid, shard, model_name, batch_size, devices[sid], per_shard_load)
        for sid, shard in enumerate(shards)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=eff_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_clip_encode_shard_job, jobs),
                total=len(jobs),
                desc="CLIP shards",
                unit="shard",
                mininterval=0.2,
                dynamic_ncols=True,
            )
        )

    results_sorted = sorted(results, key=lambda r: r[0])
    feat_blocks = [r[1] for r in results_sorted]
    path_blocks = [r[2] for r in results_sorted]

    total_rows = sum(b.shape[0] for b in feat_blocks)
    if total_rows == 0:
        raise SystemExit("No readable images could be encoded.")

    features = np.concatenate(feat_blocks, axis=0)
    merged_paths: List[Path] = [Path(p) for block in path_blocks for p in block]
    return features, merged_paths


def compute_knn_uniqueness(features: np.ndarray, k: int) -> np.ndarray:
    n = len(features)
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    k_eff = min(k + 1, n)
    nn = NearestNeighbors(
        n_neighbors=k_eff,
        metric="cosine",
        algorithm="auto",
        n_jobs=-1,
    )
    nn.fit(features)
    distances, _ = nn.kneighbors(features)
    # Remove self-neighbor at index 0 (distance 0).
    if k_eff > 1:
        local = distances[:, 1:]
    else:
        local = distances
    return local.mean(axis=1).astype(np.float32)


def farthest_point_sampling(features: np.ndarray, n_select: int, seed_idx: int) -> List[int]:
    n = len(features)
    if n_select >= n:
        return list(range(n))

    selected = [seed_idx]
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[seed_idx] = True

    # Cosine distance because embeddings are L2-normalized.
    sims = features @ features.T
    dists = 1.0 - sims
    min_dist_to_selected = dists[seed_idx].copy()

    for _ in range(1, n_select):
        min_dist_to_selected[selected_mask] = -1.0
        nxt = int(np.argmax(min_dist_to_selected))
        selected.append(nxt)
        selected_mask[nxt] = True
        min_dist_to_selected = np.minimum(min_dist_to_selected, dists[nxt])

    return selected


def select_diverse_samples(
    image_paths: Sequence[Path],
    features: np.ndarray,
    top_n: int,
    knn_scores: np.ndarray,
) -> SelectionResult:
    n = len(image_paths)
    top_n = min(top_n, n)
    if top_n == 0:
        return SelectionResult(selected_paths=[], knn_uniqueness_scores=[])

    # Use the most unique sample (highest kNN distance) as FPS seed.
    seed_idx = int(np.argmax(knn_scores))
    chosen_indices = farthest_point_sampling(features, top_n, seed_idx)

    # Sort selected results by uniqueness score (desc) for readable output.
    chosen_indices = sorted(chosen_indices, key=lambda i: float(knn_scores[i]), reverse=True)
    selected_paths = [str(image_paths[i]) for i in chosen_indices]
    selected_scores = [float(knn_scores[i]) for i in chosen_indices]
    return SelectionResult(selected_paths, selected_scores)


def main() -> None:
    args = parse_args()

    load_workers = args.load_workers
    if load_workers is None:
        load_workers = _default_load_workers()

    collected_paths = collect_images(args.folders)
    print(f"[INFO] Collected images: {len(collected_paths)}")
    if len(collected_paths) == 0:
        raise SystemExit("No images found in the given folders.")

    print(f"[INFO] Image load workers (per CLIP process): {load_workers}")
    features, image_paths = encode_images(
        image_paths=collected_paths,
        model_name=args.clip_model,
        batch_size=args.batch_size,
        device_arg=args.device,
        num_load_workers=load_workers,
        inference_workers=args.inference_workers,
        inference_devices=args.inference_devices,
        gpu_index=args.gpu_index,
    )
    n_skipped = len(collected_paths) - len(image_paths)
    if n_skipped:
        print(f"[INFO] Skipped unreadable/missing: {n_skipped}")
    print(f"[INFO] Encoded features shape: {features.shape}")

    knn_scores = compute_knn_uniqueness(features, args.knn_k)
    result = select_diverse_samples(
        image_paths=image_paths,
        features=features,
        top_n=args.top_n,
        knn_scores=knn_scores,
    )

    payload = {
        "num_collected": len(collected_paths),
        "num_total_images": len(image_paths),
        "top_n": min(args.top_n, len(image_paths)),
        "knn_k": args.knn_k,
        "selected_samples": [
            {"path": p, "knn_uniqueness": s}
            for p, s in zip(result.selected_paths, result.knn_uniqueness_scores)
        ],
    }
    if not args.output_json:
        args.output_json = Path(args.output_dir) / "unique_screenshot_samples.json"
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[INFO] Saved output: {output_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_output_path = output_dir / "sample_screenshots.json"
        selected_abs_paths = [str(Path(p).resolve()) for p in result.selected_paths]
        sample_output_path.write_text(
            json.dumps(selected_abs_paths, indent=2), encoding="utf-8"
        )
        print(f"[INFO] Saved selected path list: {sample_output_path}")

    print("[INFO] Selected diverse screenshots:")
    for idx, item in enumerate(payload["selected_samples"], start=1):
        print(f"{idx:02d}. score={item['knn_uniqueness']:.6f} | {item['path']}")


if __name__ == "__main__":
    main()
