import json
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[2]


def discover_latest(scratch_base: Path, glob: str) -> Path:
    """
    Return the folder containing the newest manifest under scratch_base.

    "Newest" is measured by the most recent ``weights.pt`` modification time
    inside the folder. Folders without any
    weights.pt fall back to the manifest.json modification time.
    """
    candidates: list[tuple[float, Path]] = []
    for sub in scratch_base.glob(glob):
        if not sub.is_dir():
            continue
        manifest = sub / "manifest.json"
        if not manifest.exists():
            continue
        weight_times = [p.stat().st_mtime for p in sub.rglob("weights.pt")]
        sort_key = max(weight_times) if weight_times else manifest.stat().st_mtime
        candidates.append((sort_key, sub))
    if not candidates:
        raise FileNotFoundError(
            f"No manifest.json found under {scratch_base}/{glob}. "
            "Run the corresponding training script first (it writes manifest.json)."
        )
    candidates.sort(key=lambda t: t[0], reverse=True)
    chosen = candidates[0][1]
    print(f"  Auto-discovered experiment dir: {chosen}")
    return chosen


def _resolve_path(p: Optional[Path] = None, base_p: Path = ROOT) -> Optional[Path]:
    """
    Return the path if given
    Check if is absolute and if not return it attached to the base path
    """
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else (base_p / p)


def load_manifest(path) -> dict:
    """Load a manifest JSON and resolve its internal paths."""
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest.json not found at {path}")
    with open(path) as f:
        manifest = json.load(f)
    manifest["_manifest_path"] = path
    manifest["benchmark_csv"] = _resolve_path(manifest.get("benchmark_csv"), ROOT)
    # Non-absolute paths resolve relative to the manifest's own folder so
    # manifests stay portable across machines (absolute paths pass through).
    manifest["normalization_stats"] = _resolve_path(
        manifest.get("normalization_stats"), path.parent
    )
    for entry in manifest.get("models", []):
        entry["weights"] = _resolve_path(entry.get("weights"), path.parent)
    return manifest


def get_model_entry(manifest: dict, name: str) -> dict:
    for entry in manifest["models"]:
        if entry["name"] == name:
            return entry
    raise KeyError(
        f"Model '{name}' not in manifest (have: "
        f"{[m['name'] for m in manifest['models']]})"
    )
