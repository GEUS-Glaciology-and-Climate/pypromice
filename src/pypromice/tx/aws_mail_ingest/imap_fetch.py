from __future__ import annotations
import imaplib
import email
import email.policy
import logging
from typing import Iterable, Tuple, List, Optional, Callable
from .config import IMAPConfig

log = logging.getLogger("aws_mail_ingest.imap")

EMAIL_POLICY = email.policy.default

class GmailFetcher:
    def __init__(self, cfg: IMAPConfig, window: int = 2000, run_id: str | None = None):
        self.cfg = cfg
        self.window = window
        self.run_id = run_id


    def _connect(self) -> imaplib.IMAP4_SSL:
        log.debug("IMAP connect/select", extra={"mailbox": self.cfg.mailbox})
        imap = imaplib.IMAP4_SSL(host=self.cfg.host, port=self.cfg.port)
        typ, _ = imap.login(self.cfg.user, self.cfg.password)
        if typ != "OK":
            log.error("Authentication failed")
            raise RuntimeError("Authentication failed")
        typ, _ = imap.select(self.cfg.mailbox, readonly=True)
        if typ != "OK":
            log.error("Cannot select mailbox", extra={"mailbox": self.cfg.mailbox})
            raise RuntimeError(f"Cannot select mailbox {self.cfg.mailbox}")
        return imap

    def _uidnext(self, imap: imaplib.IMAP4_SSL) -> int:
        typ, data = imap.status(self.cfg.mailbox, "(UIDNEXT)")
        if typ != "OK" or not data:
            log.error("STATUS UIDNEXT failed")
            raise RuntimeError("STATUS UIDNEXT failed")
        # data like: [b'INBOX (UIDNEXT 1234567)']
        s = data[0].decode("utf-8", "ignore")
        num = int(s.split("UIDNEXT")[1].split(")")[0].strip())
        return num

    def _parse_fetch(self, fetch_data) -> List[Tuple[int, bytes]]:
        """Extract (uid, raw_bytes) tuples from IMAP FETCH response."""
        out: List[Tuple[int, bytes]] = []
        for part in fetch_data or []:
            if isinstance(part, tuple) and len(part) == 2 and isinstance(part[1], (bytes, bytearray)):
                header = part[0].decode("utf-8", "ignore")
                uid_val = None
                try:
                    uid_idx = header.find("UID ")
                    if uid_idx != -1:
                        uid_val = int(header[uid_idx + 4:].split()[0])
                except Exception:
                    uid_val = None
                if uid_val is not None:
                    out.append((uid_val, part[1]))
        return out

    def get_first_uid(self, mailbox: str) -> int:
        imap = self._connect()

        status, data = imap.uid("FETCH", b"1", "(UID)")
        if status != "OK" or not data or data[0] is None:
            raise RuntimeError(f"Could not get first UID for mailbox {mailbox}")

        parts = data[0][0].decode().split()
        uid_index = parts.index("UID") + 1

        return int(parts[uid_index])

    def fetch_windowed(
            self,
            start_uid: int | None = None,
            direction: str = "forward",
            exists_predicate: Optional[Callable[[List[int]], set[int]]] = None,
    ) -> Iterable[Tuple[int, bytes]]:
        """
        Forward (default): iterate [start_uid+1 .. UIDNEXT-1] in ascending windows.
        Backward: iterate from max_uid (or start_uid) downward in windows.
        If exists_predicate is provided (list[int] -> set[int]), it will:
          - log 'skipped' per window,
          - stop early if an entire window already exists in DB.
        """
        imap = self._connect()
        try:
            uidnext = self._uidnext(imap)
            max_uid = uidnext - 1
            if max_uid < 1:
                log.info("mailbox.empty", extra={"run_id": self.run_id, "mailbox": self.cfg.mailbox})
                return

            if direction == "backward":
                cur_high = min(max_uid, start_uid if start_uid is not None else max_uid)
                while cur_high >= 1:
                    cur_low = max(1, cur_high - self.window + 1)
                    # Always fetch low:high per IMAP syntax; we’ll yield in reverse.
                    typ, fetch_data = imap.uid("FETCH", f"{cur_low}:{cur_high}", "(RFC822 UID)")
                    msgs = self._parse_fetch(fetch_data)
                    # Sort descending by UID for backward processing
                    msgs.sort(key=lambda t: t[0], reverse=True)
                    uids = [u for (u, _) in msgs]
                    skipped = 0
                    if exists_predicate and uids:
                        existing = exists_predicate(uids)
                        skipped = len(existing)
                        if skipped == len(uids):
                            log.info(
                                "FETCH window",
                                extra={
                                    "run_id": self.run_id,
                                    "start_uid": cur_high,
                                    "end_uid": cur_low,
                                    "mailbox": self.cfg.mailbox,
                                    "direction": "backward",
                                    "fetched": len(uids),
                                    "skipped": skipped,
                                },
                            )
                            # early stop: entire window already present
                            return
                    log.info(
                        "FETCH window",
                        extra={
                            "run_id": self.run_id,
                            "start_uid": cur_high,
                            "end_uid": cur_low,
                            "mailbox": self.cfg.mailbox,
                            "direction": "backward",
                            "fetched": len(uids),
                            "skipped": skipped,
                        },
                    )
                    # Yield only non-existing first (if predicate provided), otherwise all
                    for uid, raw in msgs:
                        if exists_predicate:
                            # only yield if not in DB
                            if uid in existing:  # type: ignore[name-defined]
                                continue
                        yield uid, raw
                    cur_high = cur_low - 1
            else:
                # FORWARD (original behavior)
                cur_low = max(1, (start_uid or 0) + 1)
                while cur_low <= max_uid:
                    cur_high = min(cur_low + self.window - 1, max_uid)
                    typ, fetch_data = imap.uid("FETCH", f"{cur_low}:{cur_high}", "(RFC822 UID)")
                    msgs = self._parse_fetch(fetch_data)
                    uids = [u for (u, _) in msgs]
                    skipped = 0
                    if exists_predicate and uids:
                        existing = exists_predicate(uids)
                        skipped = len(existing)
                    log.info(
                        "FETCH window",
                        extra={
                            "run_id": self.run_id,
                            "start_uid": cur_low,
                            "end_uid": cur_high,
                            "mailbox": self.cfg.mailbox,
                            "direction": "forward",
                            "fetched": len(uids),
                            "skipped": skipped,
                        },
                    )
                    for uid, raw in msgs:
                        if exists_predicate and uid in existing:  # type: ignore[name-defined]
                            continue
                        yield uid, raw
                    cur_low = cur_high + 1
        finally:
            imap.logout()
            log.debug("imap.logout", extra={"run_id": self.run_id, "mailbox": self.cfg.mailbox})