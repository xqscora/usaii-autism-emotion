"""Download / import face emotion datasets into datasets/ (gitignored).

Fully automatic:
  fer2013_hf  — Hugging Face abhilash88/fer2013-enhanced

Semi-automatic (you downloaded the zip):
  fer_autism  — Mendeley FER-Autism b33pf78h62

Usage:
  python code/download_fer_datasets.py --dataset fer2013_hf
  python code/download_fer_datasets.py --dataset fer2013_hf --max-per-split 500
  python code/download_fer_datasets.py --dataset fer_autism --zip Downloads/FER-Autism.zip
  python code/download_fer_datasets.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"

FER2013_HF = "abhilash88/fer2013-enhanced"
FER2013_FALLBACK = "TrainingDataPro/faces-emotions-dataset"

FER_AUTISM_URL = "https://data.mendeley.com/datasets/b33pf78h62"


def _save_pil(img, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(img, "save"):
        img.save(path)
        return
    arr = np.asarray(img)
    if arr.ndim == 2:
        Image.fromarray(arr.astype(np.uint8)).save(path)
    else:
        Image.fromarray(arr.astype(np.uint8)).save(path)


def export_fer2013_hf(out_dir: Path, max_per_split: int | None) -> dict:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"source": FER2013_HF, "splits": {}}
    try:
        ds_dict = load_dataset(FER2013_HF)
    except Exception as e:
        print(f"primary HF dataset failed ({e}), trying fallback {FER2013_FALLBACK}")
        ds_dict = load_dataset(FER2013_FALLBACK)

    for split_name, split in ds_dict.items():
        n = len(split)
        limit = min(n, max_per_split) if max_per_split else n
        meta["splits"][split_name] = {"total": n, "exported": limit}
        print(f"export {split_name}: {limit}/{n}")
        for i in range(limit):
            row = split[i]
            emo = row.get("emotion_name") or row.get("label") or str(row.get("emotion", "unknown"))
            emo = str(emo).strip().replace(" ", "_").lower()
            sid = row.get("sample_id", i)
            img = row.get("image")
            if img is None and "pixels" in row:
                px = np.array([int(x) for x in str(row["pixels"]).split()], dtype=np.uint8)
                img = px.reshape(48, 48)
            out_path = out_dir / split_name / emo / f"{sid}.png"
            _save_pil(img, out_path)

    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def import_fer_autism_zip(zip_path: Path, out_dir: Path) -> dict:
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_unzip_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    def _extract_all(zpath: Path, dest: Path) -> None:
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(dest)

    _extract_all(zip_path, tmp)
    inner_zips = list(tmp.rglob("*.zip"))
    if inner_zips:
        print(f"nested zip -> {inner_zips[0].name}")
        _extract_all(inner_zips[0], tmp / "inner")

    src_roots = [tmp / "inner", tmp]
    img_count = 0
    for src in src_roots:
        if not src.exists():
            continue
        for p in src.rglob("*"):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            rel = p.relative_to(src)
            # drop single wrapper folder if present
            parts = rel.parts
            if len(parts) > 1 and parts[0].lower().endswith("dataset"):
                rel = Path(*parts[1:])
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(p, dest)
            img_count += 1

    shutil.rmtree(tmp, ignore_errors=True)
    n_img = sum(
        1 for _ in out_dir.rglob("*")
        if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    meta = {"source": FER_AUTISM_URL, "images": n_img, "zip": str(zip_path)}
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"FER-Autism: {n_img} images -> {out_dir}")
    return meta


def print_catalog() -> None:
    print("Auto:")
    print(f"  fer2013_hf     HF {FER2013_HF}")
    print("Semi (Download All from Mendeley, then --zip):")
    print(f"  fer_autism     {FER_AUTISM_URL}")
    print("Capture (local long-run):")
    print("  python code/face_capture.py --person-id NAME")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download FER datasets")
    ap.add_argument("--dataset", choices=["fer2013_hf", "fer_autism"], default="fer2013_hf")
    ap.add_argument("--zip", type=str, help="Local zip for fer_autism (Mendeley Download All)")
    ap.add_argument("--max-per-split", type=int, default=None, help="Cap export size (dev/test)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print_catalog()
        return

    if args.dataset == "fer2013_hf":
        out = DATASETS / "fer2013_hf"
        meta = export_fer2013_hf(out, args.max_per_split)
        print(json.dumps(meta, indent=2))
        print(f"done -> {out}")
        return

    if args.dataset == "fer_autism":
        if not args.zip:
            print(f"Download All from {FER_AUTISM_URL}")
            print("Then: python code/download_fer_datasets.py --dataset fer_autism --zip PATH.zip")
            return
        import_fer_autism_zip(Path(args.zip), DATASETS / "FER-Autism")


if __name__ == "__main__":
    main()
