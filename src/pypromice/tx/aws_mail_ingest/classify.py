from __future__ import annotations
import email, email.policy, hashlib, re
from email.utils import parsedate_to_datetime
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.orm import Session
from .models import Message, Attachment, ClassifiedMessage
from .blobs import BlobStore

EMAIL_POLICY = email.policy.default
IMEI_RE = re.compile(r"\b(\d{14,16})\b")

def parse_envelope(raw_bytes: bytes) -> dict:
    msg = email.message_from_bytes(raw_bytes, policy=EMAIL_POLICY)
    from_addr = str(msg.get("From", ""))
    to_addr = str(msg.get("To", ""))
    subject = str(msg.get("Subject", ""))
    message_id = str(msg.get("Message-Id", ""))

    # Try to parse "Date" header
    date_header = msg.get("Date")
    try:
        header_date = parsedate_to_datetime(date_header) if date_header else None
    except Exception:
        header_date = None

    env_hash = hashlib.sha256("\n".join([from_addr, to_addr, subject, message_id]).encode()).hexdigest()
    return {
        "email_obj": msg,
        "from": from_addr,
        "to": to_addr,
        "subject": subject,
        "message_id": message_id,
        "env_hash": env_hash,
        "size": len(raw_bytes),
        "header_date": header_date,
    }

def extract_attachments(msg: email.message.Message, blob: BlobStore, message_rec: Message, session: Session) -> None:
    if not msg.is_multipart():
        return
    idx = 0
    for part in msg.walk():
        if part.is_multipart():
            continue
        cdisp = part.get("Content-Disposition", "").lower()
        ctype = part.get_content_type()
        if "attachment" in cdisp or ctype not in ("text/plain", "text/html"):
            data = part.get_payload(decode=True) or b""
            sha, path = blob.save_attachment(data)
            att = Attachment(
                message=message_rec,
                part_id=str(idx),
                filename=part.get_filename(),
                mime=ctype,
                size=len(data),
                bytes_hash=sha,
                blob_uri=path,
            )
            session.add(att)
        idx += 1

def classify_message(message: Message, session: Session) -> ClassifiedMessage:
    raw = Path(message.raw_blob_uri).read_bytes()
    msg = email.message_from_bytes(raw, policy=EMAIL_POLICY)

    text_chunks: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    text_chunks.append(part.get_content())
                except Exception:
                    pass
    else:
        if msg.get_content_type() == "text/plain":
            text_chunks.append(msg.get_content())
    body_text = "\n".join(text_chunks)

    imei = None
    for src in filter(None, [message.subject, body_text]):
        m = IMEI_RE.search(src)
        if m:
            imei = m.group(1); break

    mtype, confidence = "noise", 30
    for att in message.attachments:
        if (att.mime or "").lower() in {"application/octet-stream", "application/x-sbd"} or (att.filename or "").lower().endswith(".sbd"):
            mtype, confidence = "sbd_tx", 90
            break
    if mtype == "noise" and (message.subject or "").lower().startswith("status"):
        mtype, confidence = "status", 60

    rec = ClassifiedMessage(
        message=message,
        message_type=mtype,
        imei=imei,
        tx_counter=None,
        confidence=confidence,
    )
    session.add(rec)
    message.state = "CLASSIFIED"
    return rec