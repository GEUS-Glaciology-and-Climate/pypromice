from __future__ import annotations
import json, hashlib
from sqlalchemy import select
from sqlalchemy.orm import Session
from .config import DATABASE_URL, GMAIL_USER, GMAIL_OAUTH_TOKEN, MAILBOX, BLOB_ROOT, IMAPConfig, env
from .models import init_db as _init_db, get_engine, Message, DecodedL0
from .blobs import BlobStore
from .imap_fetch import GmailFetcher
from .classify import parse_envelope, extract_attachments, classify_message
from .decoder import extract_payload, DecoderStub

def init_db():
    BLOB_ROOT.mkdir(parents=True, exist_ok=True)
    _init_db(get_engine(DATABASE_URL))

def ingest():
    engine = get_engine(DATABASE_URL)
    blob = BlobStore(BLOB_ROOT)
    cfg = IMAPConfig(user=env("GMAIL_USER", GMAIL_USER), token=env("GMAIL_OAUTH_TOKEN", GMAIL_OAUTH_TOKEN), mailbox=MAILBOX)
    fetcher = GmailFetcher(cfg)

    with Session(engine) as s:
        last_uid = s.scalar(select(Message.gmail_uid).where(Message.mailbox == cfg.mailbox).order_by(Message.gmail_uid.desc()).limit(1))
        new_count = 0
        for uid, raw in fetcher.fetch_since_uid(last_uid):
            envd = parse_envelope(raw)
            try:
                raw_uri = blob.save_raw(cfg.mailbox, uid, raw)
                m = Message(
                    mailbox=cfg.mailbox,
                    gmail_uid=uid,
                    message_id=envd["message_id"],
                    from_addr=envd["from"],
                    to_addr=envd["to"],
                    subject=envd["subject"],
                    size=envd["size"],
                    raw_blob_uri=raw_uri,
                    envelope_hash=envd["env_hash"],
                )
                s.add(m); s.flush()
                extract_attachments(envd["email_obj"], blob, m, s)
                s.commit(); new_count += 1
            except Exception as e:
                s.rollback()  # likely duplicate
                print(f"Skip UID {uid}: {e}")
        print(f"Ingested {new_count} new messages.")

def run_classify(limit: int = 200):
    engine = get_engine(DATABASE_URL)
    with Session(engine) as s:
        q = s.scalars(select(Message).where(Message.state == "NEW").order_by(Message.gmail_uid.asc()).limit(limit))
        count = 0
        for m in q:
            try:
                classify_message(m, s)
                s.commit(); count += 1
            except Exception as e:
                m.state = "FAILED"; m.error = f"classify: {e}"; s.commit()
        print(f"Classified {count} messages.")

def run_decode(limit: int = 200):
    engine = get_engine(DATABASE_URL)
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
                s.add(rec); m.state = "DECODED"; s.commit(); done += 1
            except Exception as e:
                m.state = "FAILED"; m.error = f"decode: {e}"; s.commit()
        print(f"Decoded {done} messages.")

def stats():
    engine = get_engine(DATABASE_URL)
    with Session(engine) as s:
        total = s.scalar(select(Message).count()) if hasattr(select(Message), "count") else s.execute("SELECT COUNT(*) FROM messages").scalar_one()
        by_state = s.execute("SELECT state, COUNT(*) FROM messages GROUP BY state").all()
        print(f"Messages: {total}")
        for state, cnt in by_state:
            print(f"  {state:10s} {cnt}")
        decoded = s.execute("SELECT COUNT(*) FROM decoded_l0").scalar_one()
        print(f"Decoded L0: {decoded}")