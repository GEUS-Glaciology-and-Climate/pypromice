from __future__ import annotations
import json, hashlib, time
import logging
from sqlalchemy import select, text
from sqlalchemy.orm import Session
from .config import DATABASE_URL, GMAIL_USER, GMAIL_OAUTH_TOKEN, MAILBOX, BLOB_ROOT, IMAPConfig, env
from .loging_conf import setup_logging
from .metrics import Metrics
from .models import init_db as _init_db, get_engine, Message, DecodedL0, MailboxState
from .blobs import BlobStore
from .imap_fetch import GmailFetcher
from .classify import parse_envelope, extract_attachments, classify_message
from .decoder import extract_payload, DecoderStub

log = logging.getLogger("aws_mail_ingest.pipeline")
metrics = Metrics()

def init_db():
    setup_logging()
    BLOB_ROOT.mkdir(parents=True, exist_ok=True)
    _init_db(get_engine(DATABASE_URL))
    log.info("db.initialized", extra={"db": DATABASE_URL, "blob_root": str(BLOB_ROOT)})

def _get_last_uid(s: Session, mailbox: str) -> int:
    st = s.get(MailboxState, mailbox)
    if st: return st.last_uid
    # fallback to highest in messages table
    last = s.scalar(select(Message.gmail_uid).where(Message.mailbox == mailbox).order_by(Message.gmail_uid.desc()).limit(1))
    return int(last or 0)

def _set_last_uid(s: Session, mailbox: str, uid: int) -> None:
    st = s.get(MailboxState, mailbox)
    if not st:
        st = MailboxState(mailbox=mailbox, last_uid=uid)
        s.add(st)
    else:
        st.last_uid = max(st.last_uid, uid)
    s.commit()


def ingest(window: int = 2000, throttle_ms: int = 0, start_override: int | None = None, max_messages: int | None = None):
    """
    Windowed ingest. Resumes from checkpoint unless start_override is given.
    throttle_ms: sleep between windows to be gentle on Gmail.
    max_messages: stop after N messages (useful for backfill batches).
    """
    setup_logging()
    engine = get_engine(DATABASE_URL)
    blob = BlobStore(BLOB_ROOT)
    cfg = IMAPConfig.from_files(
        "/Users/maclu/data/credentials/accounts.ini",
        "/Users/maclu/data/credentials/credentials.ini",
    )
    window = min(window, max_messages)
    fetcher = GmailFetcher(cfg, window=window)

    with Session(engine) as s:
        start_uid = start_override if start_override is not None else _get_last_uid(s, cfg.mailbox)
        seen = 0
        log.info("ingest.start", extra={"mailbox": cfg.mailbox, "start_uid": start_uid, "window": window})
        for uid, raw in fetcher.fetch_windowed(start_uid):
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
                s.commit()
                _set_last_uid(s, cfg.mailbox, uid)
                seen += 1
                metrics.inc("ingested_messages")
                if seen % 100 == 0:
                    log.info("ingest.progress", extra={"last_uid": uid, "seen": seen})
            except Exception as e:
                s.rollback()  # likely duplicate due to UNIQUE
                log.exception("ingest.error", extra={"gmail_uid": uid})
                metrics.inc("ingest_errors")
            if max_messages and seen >= max_messages:
                break
            if throttle_ms:
                time.sleep(throttle_ms / 1000.0)
        log.info("ingest.done", extra={"seen": seen})


def run_classify(limit: int = 200):
    setup_logging()
    engine = get_engine(DATABASE_URL)
    with Session(engine) as s:
        q = s.scalars(select(Message).where(Message.state == "NEW").order_by(Message.gmail_uid.asc()).limit(limit))
        count = 0
        for m in q:
            try:
                classify_message(m, s)
                s.commit(); count += 1
                metrics.inc("classified_messages")
            except Exception:
                m.state = "FAILED"; m.error = "classify failed"
                s.commit(); metrics.inc("classify_errors")
                log.exception("classify.error", extra={"gmail_uid": m.gmail_uid})
        log.info("classify.done", extra={"count": count})
        metrics.write()

def run_decode(limit: int = 200):
    setup_logging()
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
                    message_id=m.id, station_id=None, deployment_id=None, logger_program_version=None,
                    payload_type="sbd", payload_hash=hashlib.sha256(payload.bytes).hexdigest(),
                    l0_json=json.dumps(l0, sort_keys=True), decoder_version=DecoderStub.VERSION,
                )
                s.add(rec); m.state = "DECODED"; s.commit(); done += 1
                metrics.inc("decoded_messages")
            except Exception:
                m.state = "FAILED"; m.error = "decode failed"; s.commit()
                metrics.inc("decode_errors")
                log.exception("decode.error", extra={"gmail_uid": m.gmail_uid})
        log.info("decode.done", extra={"count": done})
        metrics.write()

def stats():
    setup_logging()
    engine = get_engine(DATABASE_URL)
    with Session(engine) as s:
        total = s.execute(text("SELECT COUNT(*) FROM messages")).scalar_one()
        by_state = s.execute(text("SELECT state, COUNT(*) FROM messages GROUP BY state")).all()
        decoded = s.execute(text("SELECT COUNT(*) FROM decoded_l0")).scalar_one()
        log.info("stats",
                 extra={"messages_total": total, "decoded": decoded,
                        "by_state": {state: cnt for state, cnt in by_state}})
        print(f"Messages: {total}")
        for state, cnt in by_state:
            print(f"  {state:10s} {cnt}")
        print(f"Decoded L0: {decoded}")