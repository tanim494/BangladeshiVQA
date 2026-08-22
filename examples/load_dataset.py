#!/usr/bin/env python3
"""
Loader for the BangladeshiVQA dataset.

The GitHub repo ships only a small *sample* (21 images + their QA pairs, in
`data/sample_images/`). The **full dataset** — all 2,068 images, 7,038 QA pairs, and the
train/val/test splits — is hosted on Hugging Face (see the README "Download" section).

Run:
    python examples/load_dataset.py              # summarize the bundled sample (offline)
    python examples/load_dataset.py --full       # download + summarize the full annotations from HF
    python examples/load_dataset.py --images     # also download the full image archive from HF

Set HF_DATASET_REPO (env var or the constant below) to your Hugging Face dataset repo id
before using --full / --images.
"""
import argparse
import json
import os
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(REPO_ROOT, "data", "sample_images")

# Fill this in (or export HF_DATASET_REPO) once the full dataset is uploaded — see README.
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "tanim494/BangladeshiVQA")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _require_repo():
    if "[" in HF_DATASET_REPO:
        raise SystemExit(
            "Set HF_DATASET_REPO (env var or the constant in this file) to your uploaded "
            "Hugging Face dataset repo id first. See the README 'Download' section."
        )


def load_sample():
    """Load the bundled sample (grouped by image) and flatten to QA records."""
    grouped = load_json(os.path.join(SAMPLE_DIR, "sample_qa.json"))
    flat = []
    for img in grouped:
        for qa in img["qa_pairs"]:
            flat.append({"image_id": img["image_id"], "category": img["category"], **qa})
    return grouped, flat


def summarize_sample(grouped, flat):
    print(f"Sample: {len(grouped)} images, {len(flat)} QA pairs, "
          f"{len({g['category'] for g in grouped})} categories")
    print("Question types:", dict(Counter(r["q_type"] for r in flat)))
    print("Cognitive levels:", dict(Counter(r["level"] for r in flat)))
    ex = flat[0]
    print("\nExample:")
    print(f"  image_id: {ex['image_id']}  ({ex['q_type']}, {ex['level']})")
    print(f"  Q: {ex['question']}")
    print(f"  A: {ex['answer']}")


def load_full_from_hf():
    """Download the full annotations + splits from Hugging Face and run sanity checks."""
    _require_repo()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub to download the full dataset.")

    def hf_json(fname):
        path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=fname, repo_type="dataset")
        return load_json(path)

    qa = hf_json("annotations/qa_pairs.json")
    splits = {name: hf_json(f"annotations/splits/{name}.json") for name in ("train", "val", "test")}

    print(f"QA pairs: {len(qa)}")
    print(f"Images (unique): {len({r['image_id'] for r in qa})}")
    print("Splits:", {k: len(v) for k, v in splits.items()})
    print("Question types:", dict(Counter(r["q_type"] for r in qa)))
    print("Cognitive levels:", dict(Counter(r["level"] for r in qa)))

    # Sanity checks against the published figures.
    assert len(qa) == 7038, f"expected 7038 QA pairs, got {len(qa)}"
    assert {k: len(v) for k, v in splits.items()} == {
        "train": 5628, "val": 704, "test": 706}, "unexpected split sizes"
    imgs = {k: {r["image_id"] for r in v} for k, v in splits.items()}
    assert not (imgs["train"] & imgs["test"]), "train/test image overlap!"
    assert not (imgs["train"] & imgs["val"]), "train/val image overlap!"
    assert not (imgs["val"] & imgs["test"]), "val/test image overlap!"
    print("Checks passed: counts match and splits are disjoint by image ID.")


def download_images(dest="images"):
    """Download the full image archive from Hugging Face into `dest/`."""
    _require_repo()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub to download images.")
    path = snapshot_download(repo_id=HF_DATASET_REPO, repo_type="dataset", local_dir=dest)
    print(f"Dataset downloaded to: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Load the BangladeshiVQA dataset.")
    ap.add_argument("--full", action="store_true",
                    help="download + summarize the full annotations from Hugging Face")
    ap.add_argument("--images", action="store_true",
                    help="also download the full image archive from Hugging Face")
    args = ap.parse_args()

    if args.full or args.images:
        load_full_from_hf()
        if args.images:
            download_images()
    else:
        grouped, flat = load_sample()
        summarize_sample(grouped, flat)
        print("\n(Use --full to download and verify the complete 7,038-pair dataset from Hugging Face.)")
