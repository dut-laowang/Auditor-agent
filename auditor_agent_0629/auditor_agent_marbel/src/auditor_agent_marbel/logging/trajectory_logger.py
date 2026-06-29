from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TrajectoryLogger:
    def __init__(self, path: Path, run_meta: Dict[str, Any]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_meta = dict(run_meta)
        self.step = 0
        self.log("run_start", {"timestamp": time.time()})

    def log(self, event_type: str, data: Dict[str, Any]) -> None:
        self.step += 1
        row = {
            "step": self.step,
            "event_type": event_type,
            **self.run_meta,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self, status: str, error: Optional[str] = None) -> None:
        data: Dict[str, Any] = {"timestamp": time.time(), "status": status}
        if error:
            data["error"] = error
        self.log("run_end", data)

