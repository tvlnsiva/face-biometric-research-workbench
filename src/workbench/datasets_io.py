"""Dataset, Ground-Truth Manifest, and Local Image Explorer I/O (`src/workbench/datasets_io.py`).

Provides read-only inspection of all permitted local datasets, SHA-256 verification, ground-truth
box and verification-pair lookups, local directory scanning, and session-local image uploads
(`reports/workbench_runs/uploads/`) without transmitting any image externally.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.common.config_loader import compute_sha256, get_project_root, resolve_project_path
from src.pipeline import decode_bgr_image
from src.workbench.contracts import WorkbenchError

PROJECT_ROOT = get_project_root().resolve()

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
MAX_UPLOAD_IMAGE_BYTES: int = 16 * 1024 * 1024
MAX_IMAGE_DIMENSION_PX: int = 8192
MAX_IMAGE_PIXELS: int = 36_000_000


def get_permitted_image_roots() -> tuple[Path, ...]:
    """Return canonical directory roots from which Workbench image/directory reads are permitted."""
    return (
        (PROJECT_ROOT / "datasets").resolve(),
        (PROJECT_ROOT / "reports" / "workbench_runs" / "uploads").resolve(),
        (PROJECT_ROOT / "reports" / "workbench_runs" / "wb1_fixtures").resolve(),
    )


def _assert_path_within_permitted_roots(candidate: Path, original_ref: str) -> Path:
    resolved = candidate.resolve()
    for root in get_permitted_image_roots():
        if resolved == root or resolved.is_relative_to(root):
            return resolved
    raise WorkbenchError(
        f"Path '{original_ref}' is outside permitted dataset/upload roots (`datasets/` and `reports/workbench_runs/uploads/`).",
        code="PATH_OUTSIDE_PERMITTED_ROOTS",
        status_code=403,
    )


def _safe_rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def get_uploads_dir() -> Path:
    up = resolve_project_path("reports/workbench_runs/uploads")
    up.mkdir(parents=True, exist_ok=True)
    return up


def parse_lfw_official_pairs(pairs_txt: Path, image_root: Path) -> list[dict[str, Any]]:
    """Parse official UMass `pairs.txt` (`10` folds × `600` pairs) directly from disk."""
    if not pairs_txt.is_file():
        return []
    lines = [ln.strip() for ln in pairs_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    header = lines[0].split()
    num_folds = int(header[0])
    per_class = int(header[1])
    pairs: list[dict[str, Any]] = []
    idx = 1
    pair_num = 0
    for fold in range(num_folds):
        split_name = "calibration_folds_1_5" if fold < 5 else "test_folds_6_10"
        for _ in range(per_class):
            parts = lines[idx].split("\t") if "\t" in lines[idx] else lines[idx].split()
            idx += 1
            subj, n1, n2 = parts[0], int(parts[1]), int(parts[2])
            p1 = image_root / subj / f"{subj}_{n1:04d}.jpg"
            p2 = image_root / subj / f"{subj}_{n2:04d}.jpg"
            pairs.append(
                {
                    "pair_id": f"lfw_f{fold+1:02d}_{pair_num:04d}_pos",
                    "fold": fold + 1,
                    "split": split_name,
                    "image_a": _safe_rel_path(p1),
                    "image_b": _safe_rel_path(p2),
                    "subject_a": subj,
                    "subject_b": subj,
                    "is_same_subject": True,
                }
            )
            pair_num += 1
        for _ in range(per_class):
            parts = lines[idx].split("\t") if "\t" in lines[idx] else lines[idx].split()
            idx += 1
            s1, n1, s2, n2 = parts[0], int(parts[1]), parts[2], int(parts[3])
            p1 = image_root / s1 / f"{s1}_{n1:04d}.jpg"
            p2 = image_root / s2 / f"{s2}_{n2:04d}.jpg"
            pairs.append(
                {
                    "pair_id": f"lfw_f{fold+1:02d}_{pair_num:04d}_neg",
                    "fold": fold + 1,
                    "split": split_name,
                    "image_a": _safe_rel_path(p1),
                    "image_b": _safe_rel_path(p2),
                    "subject_a": s1,
                    "subject_b": s2,
                    "is_same_subject": False,
                }
            )
            pair_num += 1
    return pairs


class DatasetCatalog:
    """Read-only catalog over all local evaluation manifests and permitted image splits."""

    def __init__(self) -> None:
        self._gt_by_rel_path: dict[str, dict[str, Any]] = {}
        self._lfw_pairs_cache: list[dict[str, Any]] | None = None
        self._load_ground_truth_index()

    def _load_ground_truth_index(self) -> None:
        p1_gt_path = resolve_project_path("datasets/manifests/detection_ground_truth.json")
        if p1_gt_path.is_file():
            raw_p1 = json.loads(p1_gt_path.read_text(encoding="utf-8"))
            for item in raw_p1.get("images", []):
                rel = Path(item["path"]).as_posix()
                self._gt_by_rel_path[rel] = {
                    "image_id": item["image_id"],
                    "boxes": item.get("ground_truth_boxes", []),
                    "source_manifest": "datasets/manifests/detection_ground_truth.json",
                    "annotation_note": "Human-verified integer boxes (P1 split)",
                }

        p11_gt_path = resolve_project_path("datasets/manifests/p11_detection_small_face_gt.json")
        if p11_gt_path.is_file():
            raw_p11 = json.loads(p11_gt_path.read_text(encoding="utf-8"))
            for item in raw_p11.get("images", []):
                rel = Path(item["path"]).as_posix()
                if rel not in self._gt_by_rel_path:
                    self._gt_by_rel_path[rel] = {
                        "image_id": item["image_id"],
                        "boxes": item.get("ground_truth_boxes", []),
                        "source_manifest": "datasets/manifests/p11_detection_small_face_gt.json",
                        "annotation_note": "P1.1 multi-scale regression manifest (partially YuNet-seeded; see P2-AUD-01)",
                    }

    def get_ground_truth_for_path(self, resolved_path: Path) -> dict[str, Any] | None:
        rel = _safe_rel_path(resolved_path)
        return self._gt_by_rel_path.get(rel)

    def resolve_image_ref(self, image_ref: str) -> Path:
        """Resolve a project-relative or user-selected local file path within permitted roots."""
        if not image_ref or not str(image_ref).strip():
            raise WorkbenchError("Empty `image_ref` supplied.", code="INVALID_IMAGE_REF")
        raw = str(image_ref).strip()
        cand = Path(raw)
        if not cand.is_absolute():
            cand = (PROJECT_ROOT / cand).resolve()
        else:
            cand = cand.resolve()

        cand = _assert_path_within_permitted_roots(cand, raw)
        if not cand.is_file():
            raise WorkbenchError(f"Image file not found: '{image_ref}'", code="IMAGE_NOT_FOUND", status_code=404)
        if cand.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise WorkbenchError(
                f"Unsupported image extension '{cand.suffix}'. Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}",
                code="UNSUPPORTED_IMAGE_FORMAT",
            )
        return cand

    def inspect_image(self, image_ref: str) -> dict[str, Any]:
        """Decode image metadata, native dimensions, SHA-256, and ground-truth availability."""
        path = self.resolve_image_ref(image_ref)
        bgr, stem_id, _, decode_ms = decode_bgr_image(path)
        h, w, c = bgr.shape
        sha = compute_sha256(path)
        rel = _safe_rel_path(path)
        gt = self.get_ground_truth_for_path(path)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        lum_mean = float(gray.mean())
        lum_std = float(gray.std())

        return {
            "image_ref": rel,
            "image_id": gt["image_id"] if gt else stem_id,
            "filename": path.name,
            "width": int(w),
            "height": int(h),
            "channels": int(c),
            "format": path.suffix.lstrip(".").upper(),
            "file_size_bytes": int(path.stat().st_size),
            "sha256": sha,
            "decode_ms": round(decode_ms, 3),
            "frame_quality_summary": {
                "frame_laplacian_var": round(lap_var, 2),
                "frame_brightness_mean": round(lum_mean, 2),
                "frame_contrast_std": round(lum_std, 2),
            },
            "ground_truth_available": gt is not None,
            "ground_truth_boxes": gt["boxes"] if gt else None,
            "ground_truth_source": gt["source_manifest"] if gt else None,
            "ground_truth_note": gt["annotation_note"] if gt else None,
        }

    def scan_local_directory(self, directory_path: str, limit: int = 120) -> dict[str, Any]:
        """List supported images inside a permitted local dataset or session-upload directory."""
        raw = str(directory_path).strip()
        if not raw:
            raise WorkbenchError("`directory_path` is required.", code="INVALID_DIRECTORY")
        dpath = Path(raw)
        if not dpath.is_absolute():
            dpath = (PROJECT_ROOT / dpath).resolve()
        else:
            dpath = dpath.resolve()
        dpath = _assert_path_within_permitted_roots(dpath, raw)
        if not dpath.is_dir():
            raise WorkbenchError(f"Directory not found: '{directory_path}'", code="DIRECTORY_NOT_FOUND", status_code=404)

        cap = max(1, min(int(limit), 500))
        items: list[dict[str, Any]] = []
        total_matched = 0
        for p in sorted(dpath.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                total_matched += 1
                if len(items) < cap:
                    rel = _safe_rel_path(p)
                    gt = self.get_ground_truth_for_path(p)
                    items.append(
                        {
                            "image_ref": rel,
                            "filename": p.name,
                            "file_size_bytes": int(p.stat().st_size),
                            "has_ground_truth": gt is not None,
                        }
                    )
        return {
            "directory_path": _safe_rel_path(dpath),
            "total_images_found": total_matched,
            "returned_count": len(items),
            "images": items,
        }

    def save_uploaded_image(self, filename: str, data_base64: str) -> dict[str, Any]:
        """Persist a validated local browser-selected file into `reports/workbench_runs/uploads/`."""
        if not data_base64 or not str(data_base64).strip():
            raise WorkbenchError("Empty image upload payload.", code="EMPTY_UPLOAD", status_code=400)

        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename or "upload.jpg").name)
        ext = Path(safe_stem).suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            raise WorkbenchError(
                f"Unsupported upload filename extension '{ext}'. Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}",
                code="UNSUPPORTED_IMAGE_FORMAT",
                status_code=400,
            )

        payload_str = data_base64.split(",", 1)[-1].strip() if "," in data_base64 else data_base64.strip()
        # Pre-check base64 string length before decoding (~4/3 of raw bytes)
        max_b64_chars = int(MAX_UPLOAD_IMAGE_BYTES * 4 // 3) + 1024
        if len(payload_str) > max_b64_chars:
            raise WorkbenchError(
                f"Uploaded image exceeds maximum permitted size ({MAX_UPLOAD_IMAGE_BYTES // (1024 * 1024)} MiB).",
                code="UPLOAD_TOO_LARGE",
                status_code=413,
            )
        try:
            raw_bytes = base64.b64decode(payload_str, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise WorkbenchError(f"Invalid base64 image payload: {exc}", code="INVALID_UPLOAD", status_code=400) from exc

        if len(raw_bytes) == 0:
            raise WorkbenchError("Decoded upload payload is 0 bytes.", code="EMPTY_UPLOAD", status_code=400)
        if len(raw_bytes) > MAX_UPLOAD_IMAGE_BYTES:
            raise WorkbenchError(
                f"Uploaded image ({len(raw_bytes)} bytes) exceeds {MAX_UPLOAD_IMAGE_BYTES} byte limit.",
                code="UPLOAD_TOO_LARGE",
                status_code=413,
            )

        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None:
            raise WorkbenchError("Uploaded file could not be decoded as a valid BGR image.", code="INVALID_IMAGE", status_code=400)

        ih, iw = decoded.shape[:2]
        if iw > MAX_IMAGE_DIMENSION_PX or ih > MAX_IMAGE_DIMENSION_PX or (iw * ih) > MAX_IMAGE_PIXELS:
            raise WorkbenchError(
                f"Image dimensions ({iw}x{ih}) exceed safeguard ceiling ({MAX_IMAGE_DIMENSION_PX}px / {MAX_IMAGE_PIXELS} pixels).",
                code="IMAGE_DIMENSIONS_EXCEEDED",
                status_code=422,
            )

        digest12 = hashlib.sha256(raw_bytes).hexdigest()[:12]
        out_path = get_uploads_dir() / f"{digest12}_{safe_stem}"
        if not out_path.exists():
            out_path.write_bytes(raw_bytes)
        return self.inspect_image(_safe_rel_path(out_path))

    def _get_lfw_pairs(self) -> list[dict[str, Any]]:
        if self._lfw_pairs_cache is None:
            pairs_txt = resolve_project_path("datasets/evaluation/lfw_research_only/pairs.txt")
            img_root = resolve_project_path("datasets/evaluation/lfw_research_only/lfw_funneled")
            self._lfw_pairs_cache = parse_lfw_official_pairs(pairs_txt, img_root)
        return self._lfw_pairs_cache

    def list_datasets(self) -> list[dict[str, Any]]:
        """Return metadata, licensing, SHA-256 manifests, and split counts for all local datasets."""
        ds_manifest_path = resolve_project_path("datasets/manifests/dataset_manifest.json")
        p1_gt_path = resolve_project_path("datasets/manifests/detection_ground_truth.json")
        p11_gt_path = resolve_project_path("datasets/manifests/p11_detection_small_face_gt.json")
        p11_ver_path = resolve_project_path("datasets/manifests/p11_verification_pairs.json")
        lfw_pairs_path = resolve_project_path("datasets/evaluation/lfw_research_only/pairs.txt")
        lfw_dir = resolve_project_path("datasets/evaluation/lfw_research_only/lfw_funneled")

        datasets: list[dict[str, Any]] = [
            {
                "dataset_id": "nasa_dev_samples",
                "name": "NASA Development Samples (P1 Sanity Split)",
                "source": "NASA / Wikimedia Commons",
                "license": "U.S. Public Domain (17 U.S.C. § 105)",
                "provenance": "Curated single-face, 3-astronaut crew, and zero-face lunar module images.",
                "restrictions": "Unrestricted Public Domain engineering & testing.",
                "num_images": 3,
                "num_identities": 4,
                "num_faces": 4,
                "evaluation_split": "development_samples",
                "has_detection_gt": False,
                "has_verification_pairs": False,
                "manifest_path": "datasets/manifests/dataset_manifest.json",
                "manifest_sha256": compute_sha256(ds_manifest_path) if ds_manifest_path.is_file() else None,
            },
            {
                "dataset_id": "nasa_p1_detection",
                "name": "NASA P1 Human-Verified Detection Split (6 Images / 15 Faces)",
                "source": "NASA Images Archive / Wikimedia Commons",
                "license": "U.S. Public Domain (17 U.S.C. § 105)",
                "provenance": "Human-verified integer bounding boxes + Earthrise negative control.",
                "restrictions": "Unrestricted Public Domain; contains medium/large faces (`0` `<32 px` faces).",
                "num_images": 6,
                "num_identities": 15,
                "num_faces": 15,
                "evaluation_split": "detection_evaluation",
                "has_detection_gt": True,
                "has_verification_pairs": False,
                "manifest_path": "datasets/manifests/detection_ground_truth.json",
                "manifest_sha256": compute_sha256(p1_gt_path) if p1_gt_path.is_file() else None,
            },
            {
                "dataset_id": "nasa_p11_multiscale",
                "name": "NASA P1.1 Multi-Scale Detection Regression Split (9 Images / 86 Faces)",
                "source": "NASA Public Domain Crew & Mission Ceremonies",
                "license": "U.S. Public Domain (17 U.S.C. § 105)",
                "provenance": "23 small (`<32 px`), 45 medium (`32–64 px`), and 18 large (`>64 px`) faces.",
                "restrictions": "Public Domain imagery; note `P2-AUD-01`: `71/86` boxes across 3 crowd scenes were YuNet-seeded.",
                "num_images": 9,
                "num_identities": 86,
                "num_faces": 86,
                "evaluation_split": "p11_multiscale_detection",
                "has_detection_gt": True,
                "has_verification_pairs": False,
                "manifest_path": "datasets/manifests/p11_detection_small_face_gt.json",
                "manifest_sha256": compute_sha256(p11_gt_path) if p11_gt_path.is_file() else None,
            },
            {
                "dataset_id": "nasa_p11_verification",
                "name": "NASA P1.1 Multi-Subject Verification Split (16 Images / 120 Pairs)",
                "source": "NASA Astronaut Portraits & Mission Archives",
                "license": "U.S. Public Domain (17 U.S.C. § 105)",
                "provenance": "9 astronauts (`9` genuine pairs, `111` impostor pairs).",
                "restrictions": "Public Domain integration smoke-test split (`1/111` FAR floor, ceiling saturation — NOT evidence of general recognition accuracy).",
                "num_images": 16,
                "num_identities": 9,
                "num_faces": 16,
                "evaluation_split": "p11_verification_120pairs",
                "has_detection_gt": False,
                "has_verification_pairs": True,
                "positive_pairs": 9,
                "negative_pairs": 111,
                "manifest_path": "datasets/manifests/p11_verification_pairs.json",
                "manifest_sha256": compute_sha256(p11_ver_path) if p11_ver_path.is_file() else None,
            },
        ]

        if lfw_pairs_path.is_file() and lfw_dir.is_dir():
            datasets.append(
                {
                    "dataset_id": "lfw_research_split",
                    "name": "UMass LFW Research Benchmark (6,000 Pairs / 13,233 Gallery Images)",
                    "source": "UMass Amherst Labeled Faces in the Wild (`lfw_funneled` + `pairs.txt`)",
                    "license": "Academic Research Benchmark (Huang et al., 2007)",
                    "provenance": "`5,749` total folders (`13,233` images); `4,281` paired subjects across `10` official subject-disjoint folds (`2,086` Calib `Folds 1–5` + `2,195` Test `Folds 6–10`, `0` overlap).",
                    "restrictions": "RESEARCH & ALGORITHMIC EVALUATION ONLY (`production_eligible: false`; gitignored).",
                    "num_images": 13233,
                    "num_identities": 5749,
                    "num_paired_identities": 4281,
                    "num_faces": 13233,
                    "evaluation_split": "Calib Folds 1–5 (3,000 pairs) / Held-Out Test Folds 6–10 (3,000 pairs)",
                    "has_detection_gt": False,
                    "has_verification_pairs": True,
                    "positive_pairs": 3000,
                    "negative_pairs": 3000,
                    "manifest_path": "datasets/evaluation/lfw_research_only/pairs.txt",
                    "manifest_sha256": compute_sha256(lfw_pairs_path),
                }
            )
        return datasets

    def get_dataset_items(
        self,
        dataset_id: str,
        kind: str = "images",
        split_filter: str = "all",
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Return paginated images or ground-truth pairs plus summary distributions."""
        off = max(0, int(offset))
        lim = max(1, min(int(limit), 200))

        if dataset_id == "nasa_dev_samples":
            m_path = resolve_project_path("datasets/manifests/dataset_manifest.json")
            if not m_path.is_file():
                return {"dataset_id": dataset_id, "kind": "images", "total": 0, "offset": off, "limit": lim, "items": [], "distributions": {}, "note": "Local dataset manifest not installed; see README.md Dataset Acquisition Policy."}
            raw = json.loads(m_path.read_text(encoding="utf-8"))
            imgs = raw["splits"]["development_samples"]
            items = [
                {
                    "image_id": m["image_id"],
                    "image_ref": m["path"],
                    "category": m["category"],
                    "resolution": m["resolution"],
                    "sha256": m["sha256"],
                }
                for m in imgs
            ]
            return {"dataset_id": dataset_id, "kind": "images", "total": len(items), "offset": off, "limit": lim, "items": items[off : off + lim], "distributions": {}}

        if dataset_id in {"nasa_p1_detection", "nasa_p11_multiscale"}:
            m_rel = (
                "datasets/manifests/detection_ground_truth.json"
                if dataset_id == "nasa_p1_detection"
                else "datasets/manifests/p11_detection_small_face_gt.json"
            )
            m_path = resolve_project_path(m_rel)
            if not m_path.is_file():
                return {"dataset_id": dataset_id, "kind": "images", "total": 0, "offset": off, "limit": lim, "items": [], "distributions": {}, "note": "Local dataset manifest not installed; see README.md Dataset Acquisition Policy."}
            raw = json.loads(m_path.read_text(encoding="utf-8"))
            imgs = raw.get("images", [])
            small_c, med_c, large_c = 0, 0, 0
            items = []
            for m in imgs:
                boxes = m.get("ground_truth_boxes", [])
                for b in boxes:
                    md = min(float(b["w"]), float(b["h"]))
                    if md < 32.0:
                        small_c += 1
                    elif md <= 64.0:
                        med_c += 1
                    else:
                        large_c += 1
                items.append(
                    {
                        "image_id": m["image_id"],
                        "image_ref": m["path"],
                        "category": m.get("category", "scene"),
                        "width": m.get("width"),
                        "height": m.get("height"),
                        "gt_face_count": len(boxes),
                        "ground_truth_boxes": boxes,
                    }
                )
            return {
                "dataset_id": dataset_id,
                "kind": "images",
                "total": len(items),
                "offset": off,
                "limit": lim,
                "items": items[off : off + lim],
                "distributions": {
                    "face_size_buckets": {
                        "small_lt_32px": small_c,
                        "medium_32_64px": med_c,
                        "large_gt_64px": large_c,
                    }
                },
            }

        if dataset_id == "nasa_p11_verification":
            m_path = resolve_project_path("datasets/manifests/p11_verification_pairs.json")
            if not m_path.is_file():
                return {"dataset_id": dataset_id, "kind": kind, "total": 0, "offset": off, "limit": lim, "items": [], "distributions": {}, "note": "Local dataset manifest not installed; see README.md Dataset Acquisition Policy."}
            raw = json.loads(m_path.read_text(encoding="utf-8"))
            if kind == "pairs":
                pairs = raw.get("pairs", [])
                return {
                    "dataset_id": dataset_id,
                    "kind": "pairs",
                    "total": len(pairs),
                    "offset": off,
                    "limit": lim,
                    "items": pairs[off : off + lim],
                    "distributions": {
                        "pair_balance": {
                            "positive_genuine": int(raw.get("num_genuine_pairs", 9)),
                            "negative_impostor": int(raw.get("num_impostor_pairs", 111)),
                        }
                    },
                }
            imgs = raw.get("images", [])
            items = [
                {
                    "image_id": m["image_id"],
                    "image_ref": m["path"],
                    "subject_tag_eval_only": m["subject_id"],
                    "sha256": m["sha256"],
                }
                for m in imgs
            ]
            return {"dataset_id": dataset_id, "kind": "images", "total": len(items), "offset": off, "limit": lim, "items": items[off : off + lim], "distributions": {}}

        if dataset_id == "lfw_research_split":
            pairs = self._get_lfw_pairs()
            if split_filter in {"calibration_folds_1_5", "test_folds_6_10"}:
                filtered = [p for p in pairs if p["split"] == split_filter]
            else:
                filtered = pairs

            if kind == "pairs":
                pos_n = sum(1 for p in filtered if p["is_same_subject"])
                neg_n = len(filtered) - pos_n
                return {
                    "dataset_id": dataset_id,
                    "kind": "pairs",
                    "total": len(filtered),
                    "offset": off,
                    "limit": lim,
                    "items": filtered[off : off + lim],
                    "distributions": {
                        "pair_balance": {"positive_genuine": pos_n, "negative_impostor": neg_n},
                        "identity_disjoint_splits": {
                            "calibration_folds_1_to_5_subjects": 2086,
                            "test_folds_6_to_10_subjects": 2195,
                            "cross_split_overlap": 0,
                        },
                    },
                }

            seen: set[str] = set()
            img_items: list[dict[str, Any]] = []
            for p in filtered:
                for key_r, key_s in (("image_a", "subject_a"), ("image_b", "subject_b")):
                    ref = p[key_r]
                    if ref not in seen:
                        seen.add(ref)
                        img_items.append(
                            {
                                "image_id": Path(ref).stem,
                                "image_ref": ref,
                                "subject_tag_eval_only": p[key_s],
                                "fold": p["fold"],
                                "split": p["split"],
                            }
                        )
            return {
                "dataset_id": dataset_id,
                "kind": "images",
                "total": len(img_items),
                "offset": off,
                "limit": lim,
                "items": img_items[off : off + lim],
                "distributions": {
                    "identity_disjoint_splits": {
                        "calibration_folds_1_to_5_subjects": 2086,
                        "test_folds_6_to_10_subjects": 2195,
                        "cross_split_overlap": 0,
                    }
                },
            }

        raise WorkbenchError(f"Unknown dataset_id='{dataset_id}'.", code="UNKNOWN_DATASET", status_code=404)
