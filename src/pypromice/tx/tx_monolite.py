#!/usr/bin/env python3
"""
AWS Mail Ingest – reference implementation (Python 3.11)
=========================================================

Minimal, idempotent mail-ingest front-end for an Automatic Weather Station (AWS)
data pipeline. This decouples email fetching/normalization from payload decoding
and downstream L0→L3 processing.

Design goals
------------
- Persist raw originals (.eml and attachment blobs) for reproducibility
- Idempotent & incremental ingestion keyed by Gmail UID
- Clear state machine: NEW → CLASSIFIED → DECODED (or FAILED)
- Small ops footprint: SQLite + filesystem blobs, swap to Postgres/S3 later
- Testable seams: fetch, classify, extract, decode

What this file provides
-----------------------
- SQLite models (SQLAlchemy) for messages and attachments
- Filesystem blob store (./blobs)
- IMAP XOAUTH2 fetcher for Gmail (uses an access token you obtain separately)
- Envelope extraction and basic classification (SBD vs noise)
- Payload extraction hook
- Decoder stub (replace with your real payload decoder)
- CLI subcommands: init, ingest, classify, decode, requeue-failed, stats

Quickstart
----------
1) Create a Google Cloud project, enable Gmail API, obtain an OAuth access token
   for the Gmail account (or use an app password with IMAP if policy allows).
   Store the short-lived access token in env var GMAIL_OAUTH_TOKEN. (For prod,
   integrate a refresh-token flow; this demo keeps it simple.)

2) Set environment variables (example):

   export DATABASE_URL="sqlite:///aws_mail_ingest.db"
   export GMAIL_USER="aws.data.inbox@gmail.com"
   export GMAIL_OAUTH_TOKEN="ya29.a0..."   # access token
   export MAILBOX="INBOX"                    # optional

3) Initialize schema:

   python mail_ingest.py init

4) Ingest new mail (raw persist + envelope + attachments):

   python mail_ingest.py ingest

5) Classify messages and extract minimal metadata (IMEI etc.):

   python mail_ingest.py classify

6) Decode payloads to L0 (calls your decoder stub):

   python mail_ingest.py decode

Directory layout
----------------
- ./blobs/raw/{mailbox}/{gmail_uid}.eml            # full originals
- ./blobs/att/{sha256[:2]}/{sha256}.bin            # attachment store

Porting notes
-------------
- Swap SQLite for Postgres by changing DATABASE_URL.
- Replace the DecoderStub with your real decoder(s) and stamp decoder_version.
- (Optional) Replace the local blob store with S3; only BlobStore class changes.
- Consider replacing IMAP with Gmail API History for near-real-time deltas.

License: MIT (2025) – adapt freely.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import email
import email.policy
import hashlib
import imaplib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, Session, relationship, Mapped, mapped_column


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name)
    if val is None:
        if default is None:
            raise SystemExit(f"Missing required env var: {name}")
        return default
    return val

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///aws_mail_ingest.db")
GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_OAUTH_TOKEN = os.getenv("GMAIL_OAUTH_TOKEN", "")
MAILBOX = os.getenv("MAILBOX", "INBOX")
BLOB_ROOT = Path(os.getenv("BLOB_ROOT", "./blobs")).resolve()


# ---------------------------------------------------------------------------
# Storage layer (SQLAlchemy ORM + filesystem blobs)
# ---------------------------------------------------------------------------
Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mailbox: Mapped[str] = mapped_column(String(64), default="INBOX", index=True)
    gmail_uid: Mapped[int] = mapped_column(Integer, nullable=False)
    gmail_history_id: Mapped[Optional[int]] = mapped_column(Integer)
    message_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    internal_date: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)

    from_addr: Mapped[Optional[str]] = mapped_column(String(255))
    to_addr: Mapped[Optional[str]] = mapped_column(String(255))
    subject: Mapped[Optional[str]] = mapped_column(Text)
    size: Mapped[Optional[int]] = mapped_column(Integer)

    raw_blob_uri: Mapped[str] = mapped_column(Text)  # path or s3 URI
    envelope_hash: Mapped[str] = mapped_column(String(64))

    state: Mapped[str] = mapped_column(
        String(16), default="NEW", index=True
    )  # NEW|CLASSIFIED|DECODED|FAILED
    error: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow
    )

    attachments: Mapped[list[Attachment]] = relationship("Attachment", back_populates="message", cascade="all, delete-orphan")
    classified: Mapped[Optional[ClassifiedMessage]] = relationship("ClassifiedMessage", back_populates="message", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("mailbox", "gmail_uid", name="uq_mailbox_uid"),
    )


class Attachment(Base):
    __tablename__ = "attachments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"))

    part_id: Mapped[str] = mapped_column(String(64))
    filename: Mapped[Optional[str]] = mapped_column(String(255))
    mime: Mapped[Optional[str]] = mapped_column(String(255))
    size: Mapped[Optional[int]] = mapped_column(Integer)

    bytes_hash: Mapped[str] = mapped_column(String(64), index=True)
    blob_uri: Mapped[str] = mapped_column(Text)

    extracted_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    message: Mapped[Message] = relationship("Message", back_populates="attachments")


class ClassifiedMessage(Base):
    __tablename__ = "classified_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), unique=True)

    message_type: Mapped[str] = mapped_column(String(32))  # sbd_tx|status|noise
    imei: Mapped[Optional[str]] = mapped_column(String(32))
    tx_counter: Mapped[Optional[int]] = mapped_column(Integer)
    confidence: Mapped[Optional[int]] = mapped_column(Integer)  # 0..100

    classified_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    message: Mapped[Message] = relationship("Message", back_populates="classified")


class DecodedL0(Base):
    __tablename__ = "decoded_l0"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), unique=True)

    station_id: Mapped[Optional[str]] = mapped_column(String(64))
    deployment_id: Mapped[Optional[str]] = mapped_column(String(64))
    logger_program_version: Mapped[Optional[str]] = mapped_column(String(64))

    payload_type: Mapped[Optional[str]] = mapped_column(String(64))
    payload_hash: Mapped[str] = mapped_column(String(64))
    l0_json: Mapped[str] = mapped_column(Text)

    decoder_version: Mapped[str] = mapped_column(String(32), default="stub-1")
    decoded_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


# ---------------------------------------------------------------------------
# Blob store (local filesystem; swap for S3 later)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# IMAP (Gmail) fetcher using XOAUTH2
# ---------------------------------------------------------------------------
@dataclass
class IMAPConfig:
    user: str
    token: str
    mailbox: str = "INBOX"


class GmailFetcher:
    def __init__(self, cfg: IMAPConfig):
        self.cfg = cfg

    def _xoauth2_string(self) -> bytes:
        # ref: https://developers.google.com/gmail/imap/xoauth2-protocol
        auth_str = f"user={self.cfg.user}\x01auth=Bearer {self.cfg.token}\x01\x01"
        return auth_str.encode()

    def _connect(self) -> imaplib.IMAP4_SSL:
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        typ, _ = imap.authenticate("XOAUTH2", lambda x: self._xoauth2_string())
        if typ != "OK":
            raise RuntimeError("XOAUTH2 authentication failed")
        typ, _ = imap.select(self.cfg.mailbox)
        if typ != "OK":
            raise RuntimeError(f"Cannot select mailbox {self.cfg.mailbox}")
        return imap

    def fetch_since_uid(self, since_uid: int | None = None) -> Iterable[tuple[int, bytes]]:
        imap = self._connect()
        try:
            criteria = "ALL" if since_uid is None else f"UID {since_uid+1}:*"
            typ, data = imap.uid("SEARCH", None, criteria)
            if typ != "OK":
                raise RuntimeError("UID SEARCH failed")
            uids = [int(x) for x in data[0].split()] if data and data[0] else []
            for uid in uids:
                typ, fetch_data = imap.uid("FETCH", str(uid), "(RFC822 INTERNALDATE X-GM-MSGID X-GM-THRID X-GM-LABELS)")
                if typ != "OK" or not fetch_data:
                    continue
                # fetch_data is a list like [(b'1 (UID 123 RFC822 {bytes}', raw), b')']
                raw_bytes = b""
                internal_date = None
                for part in fetch_data:
                    if isinstance(part, tuple) and len(part) == 2 and isinstance(part[1], (bytes, bytearray)):
                        raw_bytes = part[1]
                yield uid, raw_bytes
        finally:
            imap.logout()


# ---------------------------------------------------------------------------
# Envelope & attachment extraction
# ---------------------------------------------------------------------------
EMAIL_POLICY = email.policy.default


def parse_envelope(raw_bytes: bytes) -> dict:
    msg = email.message_from_bytes(raw_bytes, policy=EMAIL_POLICY)
    from_addr = str(msg.get("From", ""))
    to_addr = str(msg.get("To", ""))
    subject = str(msg.get("Subject", ""))
    message_id = str(msg.get("Message-Id", ""))
    size = len(raw_bytes)
    # Hash a stable subset for envelope change detection
    env_hash = hashlib.sha256("\n".join([from_addr, to_addr, subject, message_id]).encode()).hexdigest()
    return {
        "from": from_addr,
        "to": to_addr,
        "subject": subject,
        "message_id": message_id,
        "size": size,
        "env_hash": env_hash,
        "email_obj": msg,
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


# ---------------------------------------------------------------------------
# Classifier & payload extractor
# ---------------------------------------------------------------------------
IMEI_RE = re.compile(r"\b(\d{14,16})\b")


def classify_message(message: Message, session: Session) -> ClassifiedMessage:
    """Very lightweight classifier. Extend with your real rules.
    - If any attachment has MIME application/octet-stream or .sbd -> sbd_tx
    - Else if subject hints -> status
    - Else -> noise
    Extract IMEI from subject/body/filename if present.
    """
    # Load raw for text search
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
            imei = m.group(1)
            break

    mtype = "noise"
    confidence = 30
    for att in message.attachments:
        if (att.mime or "").lower() in {"application/octet-stream", "application/x-sbd"} or (att.filename or "").lower().endswith(".sbd"):
            mtype = "sbd_tx"
            confidence = 90
            break
    if mtype == "noise" and (message.subject or "").lower().startswith("status"):
        mtype = "status"
        confidence = 60

    classified = ClassifiedMessage(
        message=message,
        message_type=mtype,
        imei=imei,
        tx_counter=None,
        confidence=confidence,
    )
    session.add(classified)
    message.state = "CLASSIFIED"
    return classified


@dataclass
class ExtractedPayload:
    bytes: bytes
    source: str  # attachment|inline
    hint: Optional[str] = None


def extract_payload(message: Message) -> Optional[ExtractedPayload]:
    # Prefer an attachment matching SBD heuristics
    for att in message.attachments:
        if (att.mime or "").lower() in {"application/octet-stream", "application/x-sbd"} or (att.filename or "").lower().endswith(".sbd"):
            data = Path(att.blob_uri).read_bytes()
            return ExtractedPayload(bytes=data, source="attachment", hint=att.filename or att.mime)
    # Fallback: none
    return None


# ---------------------------------------------------------------------------
# Decoder stub – replace with your real implementation
# ---------------------------------------------------------------------------
class DecoderStub:
    VERSION = "stub-1"

    @staticmethod
    def decode(payload: ExtractedPayload, message: Message) -> dict:
        # Produce a minimal, traceable record; replace with your parser.
        return {
            "ts_ingested": dt.datetime.utcnow().isoformat() + "Z",
            "payload_len": len(payload.bytes),
            "source": payload.source,
            "subject": message.subject,
            "from": message.from_addr,
        }


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def get_engine():
    return create_engine(DATABASE_URL, future=True)


def init_db():
    BLOB_ROOT.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"Initialized DB at {DATABASE_URL} and blobs at {BLOB_ROOT}")


def ingest():
    engine = get_engine()
    blob = BlobStore(BLOB_ROOT)
    cfg = IMAPConfig(user=env("GMAIL_USER"), token=env("GMAIL_OAUTH_TOKEN"), mailbox=os.getenv("MAILBOX", "INBOX"))
    fetcher = GmailFetcher(cfg)

    with Session(engine) as s:
        last_uid = s.scalar(select(Message.gmail_uid).where(Message.mailbox == cfg.mailbox).order_by(Message.gmail_uid.desc()).limit(1))
        print(f"Ingesting from Gmail since UID {last_uid if last_uid is not None else 'start'}...")
        new_count = 0
        for uid, raw in fetcher.fetch_since_uid(last_uid):
            envd = parse_envelope(raw)
            # Idempotent insert
            try:
                raw_uri = blob.save_raw(cfg.mailbox, uid, raw)
                m = Message(
                    mailbox=cfg.mailbox,
                    gmail_uid=uid,
                    gmail_history_id=None,
                    message_id=envd["message_id"],
                    internal_date=None,  # IMAP INTERNALDATE can be parsed if needed
                    from_addr=envd["from"],
                    to_addr=envd["to"],
                    subject=envd["subject"],
                    size=envd["size"],
                    raw_blob_uri=raw_uri,
                    envelope_hash=envd["env_hash"],
                )
                s.add(m)
                s.flush()
                extract_attachments(envd["email_obj"], blob, m, s)
                new_count += 1
            except Exception as e:
                # Likely duplicate due to UNIQUE(mailbox, gmail_uid)
                s.rollback()
                print(f"Skip UID {uid}: {e}")
            else:
                s.commit()
        print(f"Ingested {new_count} new messages.")


def run_classify(limit: int = 200):
    engine = get_engine()
    with Session(engine) as s:
        q = s.scalars(select(Message).where(Message.state == "NEW").order_by(Message.gmail_uid.asc()).limit(limit))
        count = 0
        for m in q:
            try:
                classify_message(m, s)
                s.commit()
                count += 1
            except Exception as e:
                m.state = "FAILED"
                m.error = f"classify: {e}"
                s.commit()
        print(f"Classified {count} messages.")


def run_decode(limit: int = 200):
    engine = get_engine()
    with Session(engine) as s:
        q = s.scalars(select(Message).where(Message.state == "CLASSIFIED").order_by(Message.gmail_uid.asc()).limit(limit))
        done = 0
        for m in q:
            try:
                payload = extract_payload(m)
                if not payload:
                    raise RuntimeError("no payload found")
                l0 = DecoderStub.decode(payload, m)
                rec = DecodedL0(
                    message_id=m.id,
                    station_id=None,
                    deployment_id=None,
                    logger_program_version=None,
                    payload_type="sbd",
                    payload_hash=hashlib.sha256(payload.bytes).hexdigest(),
                    l0_json=json.dumps(l0, sort_keys=True),
                    decoder_version=DecoderStub.VERSION,
                )
                s.add(rec)
                m.state = "DECODED"
                s.commit()
                done += 1
            except Exception as e:
                m.state = "FAILED"
                m.error = f"decode: {e}"
                s.commit()
        print(f"Decoded {done} messages.")


def requeue_failed():
    engine = get_engine()
    with Session(engine) as s:
        n = 0
        for m in s.scalars(select(Message).where(Message.state == "FAILED")):
            m.state = "NEW"
            m.error = None
            n += 1
        s.commit()
        print(f"Requeued {n} failed messages -> NEW")


def stats():
    engine = get_engine()
    with Session(engine) as s:
        total = s.scalar(select(Integer().label("cnt")).select_from(Message)) or 0
        by_state = s.execute("SELECT state, COUNT(*) FROM messages GROUP BY state").all()
        print(f"Messages: {total}")
        for state, cnt in by_state:
            print(f"  {state:10s} {cnt}")
        decoded = s.scalar(select(Integer().label("cnt")).select_from(DecodedL0)) or 0
        print(f"Decoded L0: {decoded}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aws-mail-ingest")
    sp = ap.add_subparsers(dest="cmd", required=True)

    sp.add_parser("init")
    sp.add_parser("ingest")
    sp.add_parser("classify").add_argument("--limit", type=int, default=200)
    sp.add_parser("decode").add_argument("--limit", type=int, default=200)
    sp.add_parser("requeue-failed")
    sp.add_parser("stats")

    args = ap.parse_args(argv)

    if args.cmd == "init":
        init_db()
    elif args.cmd == "ingest":
        ingest()
    elif args.cmd == "classify":
        run_classify(limit=args.limit)
    elif args.cmd == "decode":
        run_decode(limit=args.limit)
    elif args.cmd == "requeue-failed":
        requeue_failed()
    elif args.cmd == "stats":
        stats()
    else:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
