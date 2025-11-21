from __future__ import annotations
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Mapping, MutableMapping

DEFAULT_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_FORMAT = os.getenv("LOG_FORMAT", "text")  # "text" | "json"
DEFAULT_FILE = os.getenv("LOG_FILE", "")         # empty = stdout only
DEFAULT_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10 MiB
DEFAULT_BACKUPS = int(os.getenv("LOG_BACKUPS", "5"))

APP_NAME = "aws-mail-ingest"

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: MutableMapping[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach extras (safe)
        for k, v in record.__dict__.items():
            if k.startswith("_") or k in ("args", "msg", "message", "exc_text", "exc_info"):
                continue
            if k in payload:
                continue
            try:
                json.dumps(v)  # ensure serializable
                payload[k] = v
            except Exception:
                payload[k] = str(v)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

def setup_logging(level: str | None = None,
                  fmt: str | None = None,
                  file_path: str | None = None) -> None:
    """
    Configure root logging once. Honors env vars if args are None.
    """
    if getattr(setup_logging, "_configured", False):
        return
    level = (level or DEFAULT_LEVEL).upper()
    fmt = (fmt or DEFAULT_FORMAT).lower()
    file_path = file_path if file_path is not None else DEFAULT_FILE

    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))

    # Handlers
    handlers: list[logging.Handler] = []
    if file_path:
        h = RotatingFileHandler(
            file_path, maxBytes=DEFAULT_MAX_BYTES, backupCount=DEFAULT_BACKUPS
        )
        handlers.append(h)
    else:
        handlers.append(logging.StreamHandler())

    if fmt == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    for h in handlers:
        h.setFormatter(formatter)
        root.addHandler(h)

    # Make third-party chatty libs quieter if needed
    logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)
    logging.getLogger("imaplib").setLevel(logging.WARNING)

    setup_logging._configured = True  # type: ignore[attr-defined]