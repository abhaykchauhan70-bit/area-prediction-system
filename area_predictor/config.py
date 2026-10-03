import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("target", "units", "data_path", "random_seed", "date_column"):
        if key not in cfg:
            raise KeyError(f"config.json is missing required key '{key}'")
    if cfg["split_strategy"] not in ("time", "group", "random"):
        raise ValueError("split_strategy must be 'time', 'group' or 'random'")
    return cfg


def resolve(cfg, key):
    """Resolve a path from the config relative to the project root."""
    p = Path(cfg[key])
    return p if p.is_absolute() else ROOT / p
