from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict


def get_git_commit() -> str | None:
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return sha
    except Exception:
        return None


def write_manifest(output_dir: str, params: Dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": get_git_commit(),
        "params": params,
    }
    out_path = os.path.join(output_dir, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return out_path
