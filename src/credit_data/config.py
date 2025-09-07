from __future__ import annotations

import os
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


def load_config(yaml_path: str | None = None) -> Dict[str, Any]:
    load_dotenv()
    cfg: Dict[str, Any] = {}
    if yaml_path and yaml is not None and os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    # Inject common envs
    if "FRED_API_KEY" not in cfg:
        cfg["FRED_API_KEY"] = os.environ.get("FRED_API_KEY")
    return cfg
