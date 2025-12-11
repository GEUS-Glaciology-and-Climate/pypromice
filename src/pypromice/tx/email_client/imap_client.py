import email.parser
from configparser import ConfigParser
from datetime import datetime
import imaplib
import logging
from mailbox import Message
from pathlib import Path
from typing import Iterator, List, Tuple

from .base_mail_client import BaseMailClient

logger = logging.getLogger(__name__)

class IMAPClient(BaseMailClient):
    """IMAP client implementing BaseMailClient interface."""

    def __init__(
            self,
            server: str,
            port: int,
            account: str,
            mailbox: str,
            password: str,
            chunk_size: int = 50,
    ):
        logger.info("AWS data from server %s, account %s" % (server, account))
        self.chunk_size = chunk_size
        self.parser = email.parser.Parser()
        logger.info("Logging in to IMAP server...")
        self.mail_server = imaplib.IMAP4_SSL(server, port)
        status, _ = self.mail_server.login(account, password)
        if status != "OK":
            raise RuntimeError("Unable to sign in!")
        logger.info("Logged in to IMAP server.")

        result, data = self.mail_server.select(mailbox=mailbox, readonly=True)
        if result.upper() != "OK":
            logger.error(f"{mailbox} mailbox not available")
            raise Exception("Unrecognised mailbox name!")

        logger.info(f"{mailbox} mailbox contains {data[0]} messages")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Logging out of IMAP server...")
        self.mail_server.close()
        resp = self.mail_server.logout()
        assert resp[0].upper() == "BYE"

    @classmethod
    def from_config_files(cls, *config_files: Path|str) -> 'IMAPClient':
        """
        Initialize IMAPClient from a list of config file paths.
        Expects config files with [imap] section and keys: server, port, account, mailbox, password.
        The config files are parsed in order, with later values overriding earlier ones.
        """
        parser = ConfigParser()
        parser.read([str(p) for p in config_files])

        return cls(
            server=parser.get("aws", "server"),
            port=parser.getint("aws", "port"),
            account=parser.get("aws", "account"),
            mailbox=parser.get("aws", "mailbox"),
            password=parser.get("aws", "password"),
        )

    # ---------------- BaseGmailClient methods ----------------
    def iter_messages_since(self, last_uid: int) -> Iterator[Tuple[int, email.message.Message]]:
        """Yield messages whose UID is greater than last_uid."""
        uids = self.new_uids(last_uid)
        if not uids:
            return
        yield from self.fetch_mails(uids)

    def fetch_message(self, uid: int) -> Message:
        """Fetch a single email by UID."""
        return self.fetch_mails([uid]).__next__()[1]

    def get_latest_uid(self) -> int:
        """Return the highest UID currently in mailbox."""
        status, data = self.mail_server.uid("search", None, "ALL")
        if status != "OK":
            raise RuntimeError("Unable to search mailbox.")
        uids = [int(x) for x in data[0].decode().split()]
        return max(uids) if uids else 0

    # ---------------- IMAP utilities ----------------
    def fetch_mails(self, uids: list[int]) -> Iterator[Tuple[int, email.message.Message]]:
        for i in range(0, len(uids), self.chunk_size):
            chunk = uids[i:i + self.chunk_size]
            uid_string = ','.join(map(str, chunk))
            status, data = self.mail_server.uid("fetch", uid_string, "(RFC822)")
            if status != "OK":
                raise RuntimeError("Failed to fetch mail chunk")
            for uid, part in zip(chunk, data):
                if isinstance(part, tuple):
                    yield uid, self.parser.parsestr(part[1].decode())

    def _imap_search_uids(self, search_string: str) -> List[str]:
        status, data = self.mail_server.uid("search", None, search_string)
        if status != "OK":
            return []
        return data[0].decode().split()

    def new_uids(self, last_uid: int) -> List[int]:
        return list(map(int, self._imap_search_uids(f"(UID {last_uid+1}:*)")))

    def uids_by_date(self, date: datetime) -> list[str]:
        """Return UIDs for messages since the given date."""
        date_str = date.strftime("%d-%b-%Y")  # e.g., "01-Jan-2025"
        return self._imap_search_uids(f'(SINCE "{date_str}")')
