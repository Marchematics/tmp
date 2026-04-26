"""Centralized path resolution.

Reads `config.yaml` at the project root (or one set via OVD_CONFIG env var).
Expands `${VAR:-default}` placeholders against the current shell environment,
mirroring the bash-style default-value syntax.

Exposes:
    PROJECT_ROOT  — Path to the project root (where config.yaml lives by default)
    OUTPUTS_DIR   — Path for experiment outputs
    DATASETS      — dict[str, Path] for the five evaluation datasets
    DETECTORS     — dict[str, str] of HF/Ultralytics model identifiers

If pyyaml is unavailable the loader falls back to a tiny built-in parser that
handles the strict subset used by the shipped config.yaml (top-level keys,
one level of nested mappings, scalar leaves with optional `${VAR:-default}`
expansion). This keeps the project importable in stripped-down environments.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict


_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"
_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: str) -> str:
    def replace(match: re.Match) -> str:
        var, default = match.group(1), match.group(2) or ""
        return os.environ.get(var, default)
    return _ENV_PATTERN.sub(replace, value)


def _expand(node: Any) -> Any:
    if isinstance(node, str):
        return _expand_env(node)
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v) for v in node]
    return node


def _minimal_yaml_parse(text: str) -> Dict[str, Any]:
    """Stripped-down parser for the exact shape of config.yaml: top-level
    `key: value` and one-level nested `key:\\n  subkey: value`. No quotes,
    no lists, no anchors. Comments (`#`) and blank lines ignored."""
    result: Dict[str, Any] = {}
    current_key: str | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  ") and current_key is not None:
            sub_key, _, sub_val = line.strip().partition(":")
            if not result.get(current_key) or not isinstance(result[current_key], dict):
                result[current_key] = {}
            result[current_key][sub_key.strip()] = sub_val.strip()
        else:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                result[key] = {}
                current_key = key
            else:
                result[key] = val
                current_key = None
    return result


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    config_path = Path(path) if path else Path(os.environ.get("OVD_CONFIG", _DEFAULT_CONFIG))
    text = config_path.read_text()
    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load(text)
    except ImportError:
        cfg = _minimal_yaml_parse(text)
    return _expand(cfg)


_cfg = load_config()
PROJECT_ROOT = Path(_cfg.get("project_root", "./")).resolve()
OUTPUTS_DIR = Path(_cfg.get("outputs_dir", str(PROJECT_ROOT / "outputs"))).resolve()
DATASETS: Dict[str, Path] = {
    name: Path(_expand(p)) for name, p in (_cfg.get("datasets") or {}).items()
}
DETECTORS: Dict[str, str] = dict(_cfg.get("detector_models") or {})
