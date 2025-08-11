from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass

def env(name: str, default: str | None = None) -> str:
    val = os.getenv(name)
    if val is None:
        if default is None:
            raise RuntimeError(f"Missing required env var: {name}")
        return default
    return val

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///aws_mail_ingest.db")
GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_OAUTH_TOKEN = os.getenv("GMAIL_OAUTH_TOKEN", "")
MAILBOX = os.getenv("MAILBOX", "INBOX")
BLOB_ROOT = Path(os.getenv("BLOB_ROOT", "./blobs")).resolve()

@dataclass
class IMAPConfig:
    user: str
    token: str
    mailbox: str = "INBOX"
