from __future__ import annotations
import imaplib
import email
import email.policy
import logging
from typing import Iterable, Tuple
from .config import IMAPConfig

log = logging.getLogger("aws_mail_ingest.imap")

EMAIL_POLICY = email.policy.default

class GmailFetcher:
    def __init__(self, cfg: IMAPConfig, window: int = 2000):
        self.cfg = cfg
        self.window = window


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

    def fetch_windowed(self, start_uid: int | None) -> Iterable[Tuple[int, bytes]]:
        """
        Yield (uid, raw_bytes) from start_uid..(UIDNEXT-1) in windows.

        If start_uid is None, begin at 1.
        """
        imap = self._connect()
        try:
            uidnext = self._uidnext(imap)
            max_uid = uidnext - 1
            if max_uid < 1:
                log.info("Mailbox empty", extra={"mailbox": self.cfg.mailbox})
                return  # mailbox empty

            cur = max(1, (start_uid or 0) + 1)
            while cur <= max_uid:
                end = min(cur + self.window - 1, max_uid)
                log.info("FETCH window", extra={"start_uid": cur, "end_uid": end, "mailbox": self.cfg.mailbox})
                # Fetch this window; Gmail accepts ranges without pre-listing UIDs
                # Note: response is a list of tuples and trailers; we scan for the bytes
                typ, fetch_data = imap.uid("FETCH", f"{cur}:{end}", "(RFC822 UID)")
                if typ != "OK":
                    # Move forward cautiously to avoid infinite loops
                    log.warning("FETCH failed; skipping window", extra={"start_uid": cur, "end_uid": end})
                    cur = end + 1
                    continue

                # Gmail may interleave responses; parse per message
                # Each data chunk looks like: (b'123 (UID 456 RFC822 {N}', raw_bytes), b')'
                for part in fetch_data or []:
                    if isinstance(part, tuple) and len(part) == 2 and isinstance(part[1], (bytes, bytearray)):
                        # Extract UID from header string (part[0])
                        header = part[0].decode("utf-8", "ignore")
                        uid_val = None
                        # Example header contains "... UID 456 ..."
                        for tok in header.split():
                            if tok.isdigit():
                                # not reliable; find 'UID'
                                pass
                        try:
                            # safer parse
                            uid_idx = header.find("UID ")
                            if uid_idx != -1:
                                uid_val = int(header[uid_idx + 4:].split()[0])
                        except Exception:
                            uid_val = None
                        raw_bytes = part[1]
                        if uid_val is not None:
                            yield uid_val, raw_bytes
                cur = end + 1
        finally:
            imap.logout()
            log.debug("IMAP logout", extra={"mailbox": self.cfg.mailbox})