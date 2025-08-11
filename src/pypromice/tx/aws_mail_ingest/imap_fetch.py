from __future__ import annotations
import imaplib
import email
import email.policy
from typing import Iterable
from .config import IMAPConfig

EMAIL_POLICY = email.policy.default

class GmailFetcher:
    def __init__(self, cfg: IMAPConfig):
        self.cfg = cfg

    def _xoauth2_string(self) -> bytes:
        auth_str = f"user={self.cfg.user}\x01auth=Bearer {self.cfg.token}\x01\x01"
        return auth_str.encode()

    def _connect(self) -> imaplib.IMAP4_SSL:
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        typ, _ = imap.authenticate("XOAUTH2", lambda _: self._xoauth2_string())
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
            uids = [int(x) for x in (data[0].split() if data and data[0] else [])]
            for uid in uids:
                typ, fetch_data = imap.uid("FETCH", str(uid), "(RFC822)")
                if typ != "OK" or not fetch_data:
                    continue
                raw_bytes = b""
                for part in fetch_data:
                    if isinstance(part, tuple) and len(part) == 2 and isinstance(part[1], (bytes, bytearray)):
                        raw_bytes = part[1]
                yield uid, raw_bytes
        finally:
            imap.logout()