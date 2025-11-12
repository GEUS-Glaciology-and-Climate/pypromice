from __future__ import annotations
import os, time
from pathlib import Path
from typing import Dict

METRICS_PATH = os.getenv("METRICS_FILE", "")  # e.g., /var/lib/node_exporter/textfile_collector/aws_mail.prom

class Metrics:
    def __init__(self, path: str | None = None):
        self.path = path or METRICS_PATH
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}

    def inc(self, name: str, by: int = 1):
        self.counters[name] = self.counters.get(name, 0) + by

    def set_gauge(self, name: str, value: float):
        self.gauges[name] = float(value)

    def write(self):
        if not self.path:
            return
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# HELP aws_mail_ingest_metrics Generated {time.time():.0f}",
                 "# TYPE aws_mail_ingest counter"]
        for k, v in sorted(self.counters.items()):
            lines.append(f"aws_mail_ingest{{metric=\"{k}\"}} {v}")
        for k, v in sorted(self.gauges.items()):
            lines.append(f"aws_mail_ingest_gauge{{metric=\"{k}\"}} {v}")
        tmp = p.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(p)