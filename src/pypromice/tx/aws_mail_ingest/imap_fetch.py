from __future__ import annotations
import imaplib
import email
import email.policy
import logging
from typing import Iterable, Tuple, List, Optional, Callable
import re

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

    def _uidmin(self) -> int:
        """Get the minimum available uid in the mailbox"""
        imap = self._connect()

        typ, data = imap.fetch("1", "(UID)")
        if typ != "OK" or not data or data[0] is None:
            raise RuntimeError(f"Could not get first UID")

        m = re.search(rb"UID (\d+)", data[0])
        if not m:
            raise RuntimeError("Could not parse UID of first message")
        return int(m.group(1))

    def fetch_windowed(
            self,
            start_uid: int | None = None,
            forward: bool = True,
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
            min_uid = self._uidmin()

            if forward:
                start_uid = start_uid if start_uid else min_uid
                direction_factor = 1
            else:
                start_uid = start_uid if start_uid else max_uid
                direction_factor = -1

            log.info("Init loops", extra=dict(
                start_uid=start_uid,
                direction_factor=direction_factor,
                window=self.window,
                forward=forward,
                min_uid=min_uid,
                max_uid=max_uid,
            ))

            uid0 = start_uid
            while min_uid <= uid0 <= max_uid:
                uid1 = uid0 + self.window * direction_factor
                uid1 = min(uid1, max_uid)
                uid1 = max(uid1, min_uid)

                if uid1 == uid0:
                    break


                typ, fetch_data = imap.uid("FETCH", f"{uid0}:{uid1 - direction_factor}", "(RFC822 UID)")
                msgs = self._parse_fetch(fetch_data)
                # Sort descending by UID for backward processing
                if ~forward:
                    msgs.sort(key=lambda t: t[0], reverse=True)
                uids = [u for (u, _) in msgs]

                existing = set()
                if exists_predicate and uids:
                    existing = exists_predicate(uids)

                log.info(
                    "FETCH window",
                    extra={
                        "run_id": self.run_id,
                        "start_uid": uid0,
                        "end_uid": uid1,
                        "mailbox": self.cfg.mailbox,
                        "forward": forward,
                        "fetched": len(uids),
                        "skipped": len(existing),
                    },
                )

                for uid, raw in msgs:
                    if uid in existing:  # type: ignore[name-defined]
                        continue
                    yield uid, raw

                uid0 = uid1

        finally:
            imap.logout()
            log.debug("imap.logout", extra={"run_id": self.run_id, "mailbox": self.cfg.mailbox})
