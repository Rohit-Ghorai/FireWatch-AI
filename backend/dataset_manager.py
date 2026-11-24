"""
FireWatch-AI — Dataset Management & Ingestion Pipeline
======================================================
Manages dataset validation, deduplication (MD5 and perceptual hashing),
manifest generation, ingestion of new images, and reproducible leakage-free
train/val/test splits.

Classes supported:
  - 'fire'    (class 1)
  - 'no_fire' (class 0)
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import numpy as np

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["no_fire", "fire"]
CLASS_TO_IDX = {"no_fire": 0, "fire": 1}
DEFAULT_SEED = 42


def compute_md5(file_path: Path) -> str:
    """Compute MD5 hash of a file for exact duplicate detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image_path: Path, hash_size: int = 8) -> str:
    """
    Compute difference hash (dHash) for near-duplicate perceptual detection.
    Resizes image to (hash_size + 1, hash_size), converts to grayscale,
    and compares adjacent pixels.
    """
    try:
        with Image.open(image_path) as im:
            gray = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = np.array(gray)
            diff = pixels[:, 1:] > pixels[:, :-1]
            return "".join(["1" if b else "0" for b in diff.flatten()])
    except Exception:
        return ""


def hamming_distance(s1: str, s2: str) -> int:
    """Compute Hamming distance between two binary hash strings."""
    if len(s1) != len(s2) or not s1 or not s2:
        return 999
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


class DatasetManager:
    """
    Validates, audits, dedupes, and manages splits for the FireWatch-AI dataset.
    """

    def __init__(self, dataset_root: Path):
        self.root = Path(dataset_root).resolve()
        self.fire_dir = self.root / "fire"
        self.no_fire_dir = self.root / "no_fire"

    def audit_and_build_manifest(self, output_manifest: Optional[Path] = None) -> Dict:
        """
        Inspect all images in fire/ and no_fire/, validate formats and sizes,
        compute hashes, detect duplicates, and generate a comprehensive manifest.
        """
        manifest = {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset_root": str(self.root),
            "classes": CLASS_NAMES,
            "total_images": 0,
            "class_counts": {"fire": 0, "no_fire": 0},
            "valid_images": [],
            "corrupt_images": [],
            "exact_duplicates": [],
            "near_duplicates": [],
        }

        seen_md5: Dict[str, str] = {}  # md5 -> relative_path
        dhashes: List[Tuple[str, str]] = []  # (relative_path, dhash)

        for cls_name in CLASS_NAMES:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                cls_dir.mkdir(parents=True, exist_ok=True)
                continue

            for file_path in sorted(cls_dir.iterdir()):
                if not file_path.is_file():
                    continue

                rel_path = str(file_path.relative_to(self.root))
                ext = file_path.suffix.lower()

                if ext not in VALID_EXTENSIONS:
                    manifest["corrupt_images"].append({
                        "file": rel_path,
                        "reason": f"Unsupported extension: {ext}",
                    })
                    continue

                # Verify image can be opened and decoded
                try:
                    with Image.open(file_path) as im:
                        im.verify()
                    # Re-open for metadata inspection (verify() closes/invalidates the file)
                    with Image.open(file_path) as im:
                        width, height = im.size
                        mode = im.mode
                        fmt = im.format
                except Exception as e:
                    manifest["corrupt_images"].append({
                        "file": rel_path,
                        "reason": f"PIL decoding failed: {e}",
                    })
                    continue

                if width < 32 or height < 32:
                    manifest["corrupt_images"].append({
                        "file": rel_path,
                        "reason": f"Image dimension too small: {width}x{height}",
                    })
                    continue

                md5_hash = compute_md5(file_path)
                d_hash = compute_dhash(file_path)

                # Check exact duplicates
                if md5_hash in seen_md5:
                    manifest["exact_duplicates"].append({
                        "file": rel_path,
                        "duplicate_of": seen_md5[md5_hash],
                        "md5": md5_hash,
                    })
                else:
                    seen_md5[md5_hash] = rel_path

                # Check near duplicates (Hamming distance <= 4)
                for existing_path, existing_dhash in dhashes:
                    dist = hamming_distance(d_hash, existing_dhash)
                    if dist <= 4:
                        manifest["near_duplicates"].append({
                            "file": rel_path,
                            "similar_to": existing_path,
                            "hamming_distance": dist,
                        })

                dhashes.append((rel_path, d_hash))

                record = {
                    "file": rel_path,
                    "class": cls_name,
                    "label": CLASS_TO_IDX[cls_name],
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "format": fmt,
                    "bytes": file_path.stat().st_size,
                    "md5": md5_hash,
                    "dhash": d_hash,
                }
                manifest["valid_images"].append(record)
                manifest["class_counts"][cls_name] += 1
                manifest["total_images"] += 1

        if output_manifest:
            output_manifest = Path(output_manifest)
            with open(output_manifest, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            print(f"[DatasetManager] Manifest written to: {output_manifest}")

        return manifest

    def create_split(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = DEFAULT_SEED,
        output_path: Optional[Path] = None,
    ) -> Dict:
        """
        Creates a reproducible, stratified, non-overlapping train/val/test split.
        Saves relative paths and indices to avoid data leakage.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

        manifest = self.audit_and_build_manifest()
        valid = manifest["valid_images"]

        rng = random.Random(seed)

        # Group by class
        by_class: Dict[str, List[Dict]] = {"fire": [], "no_fire": []}
        for item in valid:
            by_class[item["class"]].append(item)

        train_set, val_set, test_set = [], [], []

        for cls_name, items in by_class.items():
            shuffled = list(items)
            rng.shuffle(shuffled)
            n = len(shuffled)

            if n == 0:
                continue

            n_test = max(1, int(round(n * test_ratio))) if n >= 2 else 0
            n_val = max(1, int(round(n * val_ratio))) if (n - n_test) >= 2 else 0
            n_train = n - n_val - n_test

            # Ensure train has at least 1 if n >= 3
            if n >= 3 and n_train == 0:
                n_train = 1
                if n_val > n_test:
                    n_val -= 1
                else:
                    n_test -= 1

            test_items = shuffled[:n_test]
            val_items = shuffled[n_test : n_test + n_val]
            train_items = shuffled[n_test + n_val :]

            test_set.extend(test_items)
            val_set.extend(val_items)
            train_set.extend(train_items)

        # Verify no leakage
        test_files = {x["file"] for x in test_set}
        val_files = {x["file"] for x in val_set}
        train_files = {x["file"] for x in train_set}

        assert len(test_files & val_files) == 0, "Leakage between test and val!"
        assert len(test_files & train_files) == 0, "Leakage between test and train!"
        assert len(val_files & train_files) == 0, "Leakage between val and train!"

        # Extract sorted list of all valid files for index mapping
        all_sorted_files = [x["file"] for x in valid]
        test_indices = [all_sorted_files.index(x["file"]) for x in test_set]
        val_indices = [all_sorted_files.index(x["file"]) for x in val_set]
        train_indices = [all_sorted_files.index(x["file"]) for x in train_set]

        split_dict = {
            "created_at": datetime.now(UTC).isoformat(),
            "seed": seed,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "counts": {
                "total": len(valid),
                "train": len(train_set),
                "val": len(val_set),
                "test": len(test_set),
            },
            "class_distribution": {
                "train": {
                    "fire": sum(1 for x in train_set if x["class"] == "fire"),
                    "no_fire": sum(1 for x in train_set if x["class"] == "no_fire"),
                },
                "val": {
                    "fire": sum(1 for x in val_set if x["class"] == "fire"),
                    "no_fire": sum(1 for x in val_set if x["class"] == "no_fire"),
                },
                "test": {
                    "fire": sum(1 for x in test_set if x["class"] == "fire"),
                    "no_fire": sum(1 for x in test_set if x["class"] == "no_fire"),
                },
            },
            "train_files": [x["file"] for x in train_set],
            "val_files": [x["file"] for x in val_set],
            "test_files": [x["file"] for x in test_set],
            "test_indices": sorted(test_indices),
            "val_indices": sorted(val_indices),
            "train_indices": sorted(train_indices),
            "dataset_root": str(self.root),
        }

        if output_path:
            output_path = Path(output_path)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(split_dict, f, indent=2)
            print(f"[DatasetManager] Split written to: {output_path}")

        return split_dict

    def ingest_image(
        self,
        src_path: Path,
        target_class: str,
        provenance_source: str = "manual_import",
    ) -> Tuple[bool, str]:
        """
        Validates an incoming image file and copies it to dataset/<target_class>/
        safely without overwriting or injecting duplicates.
        """
        assert target_class in CLASS_NAMES, f"Invalid class {target_class}. Must be 'fire' or 'no_fire'."
        src = Path(src_path)

        if not src.exists() or not src.is_file():
            return False, f"Source file does not exist: {src}"

        if src.suffix.lower() not in VALID_EXTENSIONS:
            return False, f"Invalid extension: {src.suffix}"

        try:
            with Image.open(src) as im:
                im.verify()
        except Exception as e:
            return False, f"Image file is corrupted: {e}"

        md5 = compute_md5(src)
        manifest = self.audit_and_build_manifest()

        # Check duplicate
        for item in manifest["valid_images"]:
            if item["md5"] == md5:
                return False, f"Exact duplicate already exists: {item['file']}"

        target_dir = self.root / target_class
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / src.name
        # If filename collision with different content, generate unique name
        if target_file.exists():
            stem = src.stem
            ext = src.suffix
            counter = 1
            while target_file.exists():
                target_file = target_dir / f"{stem}_{counter}{ext}"
                counter += 1

        shutil.copy2(src, target_file)
        print(f"[DatasetManager] Ingested {src.name} -> {target_file}")
        return True, str(target_file)


def main():
    parser = argparse.ArgumentParser(description="FireWatch-AI Dataset Manager")
    parser.add_argument("--dataset", default="../dataset", help="Path to dataset root")
    parser.add_argument("--action", choices=["audit", "split", "manifest"], default="audit")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for split")
    parser.add_argument("--manifest-out", default="dataset_manifest.json", help="Manifest output file")
    parser.add_argument("--split-out", default="test_split_indices.json", help="Split output file")
    args = parser.parse_args()

    dm = DatasetManager(Path(args.dataset))

    if args.action in ("audit", "manifest"):
        manifest = dm.audit_and_build_manifest(Path(args.manifest_out))
        print(f"\nAudit complete: {manifest['total_images']} valid images.")
        print(f"Class counts: {manifest['class_counts']}")
        if manifest["exact_duplicates"]:
            print(f"Exact duplicates found: {len(manifest['exact_duplicates'])}")
        if manifest["near_duplicates"]:
            print(f"Near duplicates found: {len(manifest['near_duplicates'])}")
        if manifest["corrupt_images"]:
            print(f"Corrupt/invalid images: {len(manifest['corrupt_images'])}")

    if args.action == "split":
        split = dm.create_split(seed=args.seed, output_path=Path(args.split_out))
        print(f"\nSplit complete. Train: {split['counts']['train']}, Val: {split['counts']['val']}, Test: {split['counts']['test']}")


if __name__ == "__main__":
    main()
