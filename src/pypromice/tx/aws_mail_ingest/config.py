from __future__ import annotations
import os
from configparser import ConfigParser
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
    password: str
    host: str = "image.gmail.com"
    port: int = 993
    mailbox: str = '"[Gmail]/All Mail"'

    @classmethod
    def from_files(cls, *paths: str|Path) -> 'IMAPConfig':
        """
        Initialize GmailClient from a list of config file paths.
        Expects config files with [imap] section and keys: server, port, account, password.
        """
        parser = ConfigParser()
        parser.read([str(p) for p in paths])

        return cls(
            host=parser.get("aws", "server"),
            port=parser.getint("aws", "port"),
            user=parser.get("aws", "account"),
            password=parser.get("aws", "password"),
        )


