"""Configuration and model manifest loading utilities.

Centralizes path resolution and schema checks so model and benchmark paths are
never hardcoded across validation scripts or tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the canonical project root directory."""
    return Path(__file__).resolve().parents[2]


def load_yaml_config(relative_or_abs_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file relative to the project root if not absolute."""
    path = Path(relative_or_abs_path)
    if not path.is_absolute():
        path = get_project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in YAML config: {path}")
    return data


def load_model_manifest(
    manifest_path: str | Path = "models/manifests/model_manifest.json",
) -> dict[str, Any]:
    """Load the JSON model manifest relative to the project root if not absolute."""
    path = Path(manifest_path)
    if not path.is_absolute():
        path = get_project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Model manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "models" not in data:
        raise ValueError(f"Invalid model manifest structure in: {path}")
    return data


def compute_sha256(file_path: str | Path) -> str:
    """Compute lowercase hexadecimal SHA-256 digest of a file."""
    path = Path(file_path)
    if not path.is_absolute():
        path = get_project_root() / path
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def resolve_project_path(relative_path: str | Path) -> Path:
    """Resolve a project-relative path against the repository root."""
    path = Path(relative_path)
    return path if path.is_absolute() else (get_project_root() / path)
