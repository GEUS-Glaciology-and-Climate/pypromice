from __future__ import annotations
from pathlib import Path
import hashlib

class BlobStore:
    def __init__(self, root: Path):
        self.root = root

    def save_raw(self, mailbox: str, uid: int, raw_bytes: bytes) -> str:
        path = self.root / "raw" / mailbox / f"{uid}.eml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw_bytes)
        return str(path)

    def save_attachment(self, data: bytes) -> tuple[str, str]:
        sha = hashlib.sha256(data).hexdigest()
        subdir = self.root / "att" / sha[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{sha}.bin"
        if not path.exists():
            path.write_bytes(data)
        return sha, str(path)
