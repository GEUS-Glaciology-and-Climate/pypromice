from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from .models import Message

@dataclass
class ExtractedPayload:
    bytes: bytes
    source: str   # attachment|inline
    hint: str | None = None

def extract_payload(message: Message) -> ExtractedPayload | None:
    # Prefer attachments that look like SBD
    for att in message.attachments:
        if (att.mime or "").lower() in {"application/octet-stream", "application/x-sbd"} or (att.filename or "").lower().endswith(".sbd"):
            data = Path(att.blob_uri).read_bytes()
            return ExtractedPayload(bytes=data, source="attachment", hint=att.filename or att.mime)
    return None

class DecoderStub:
    VERSION = "stub-1"
    @staticmethod
    def decode(payload: ExtractedPayload, message: Message) -> dict:
        return {
            "ts_ingested": dt.datetime.utcnow().isoformat() + "Z",
            "payload_len": len(payload.bytes),
            "source": payload.source,
            "subject": message.subject,
            "from": message.from_addr,
        }