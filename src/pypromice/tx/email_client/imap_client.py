import email.parser
import imaplib
import logging
from mailbox import Message
from typing import Iterator, List

from base_gmail_client import BaseGmailClient

logger = logging.getLogger(__name__)

class IMAPClient(BaseGmailClient):
    """Gmail IMAP client implementing BaseGmailClient interface."""

    def __init__(self, server: str, port: int, account: str, password: str, chunk_size: int = 50):
        self.chunk_size = chunk_size
        self.parser = email.parser.Parser()
        logger.info("Logging in to IMAP server...")
        self.mail_server = imaplib.IMAP4_SSL(server, port)
        status, _ = self.mail_server.login(account, password)
        if status != "OK":
            raise RuntimeError("Unable to sign in!")
        logger.info("Logged in to IMAP server.")

        self.mail_server.select(mailbox='"[Gmail]/All Mail"', readonly=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mail_server.logout()

    # ---------------- BaseGmailClient methods ----------------
    def iter_messages_since(self, last_uid: int) -> Iterator[Message]:
        """Yield messages whose UID is greater than last_uid."""
        uids = self.new_uids(str(last_uid))
        if not uids:
            return
        yield from self.fetch_mails(uids)

    def fetch_message(self, uid: str) -> Message:
        """Fetch a single email by UID."""
        return self.fetch_mails([uid]).__next__()

    def get_latest_uid(self) -> int:
        """Return the highest UID currently in mailbox."""
        status, data = self.mail_server.uid("search", None, "ALL")
        if status != "OK":
            raise RuntimeError("Unable to search mailbox.")
        uids = [int(x) for x in data[0].decode().split()]
        return max(uids) if uids else 0

    # ---------------- IMAP utilities ----------------
    def fetch_mails(self, uids: list[str]) -> Iterator[Message]:
        for i in range(0, len(uids), self.chunk_size):
            chunk = uids[i:i + self.chunk_size]
            uid_string = ','.join(chunk)
            status, data = self.mail_server.uid("fetch", uid_string, "(RFC822)")
            if status != "OK":
                raise RuntimeError("Failed to fetch mail chunk")
            for part in data:
                if isinstance(part, tuple):
                    yield self.parser.parsestr(part[1].decode())

    def _imap_search_uids(self, search_string: str) -> List[str]:
        status, data = self.mail_server.uid("search", None, search_string)
        if status != "OK":
            return []
        return data[0].decode().split()

    def new_uids(self, last_uid: str) -> List[str]:
        return self._imap_search_uids(f"(UID {int(last_uid)+1}:*)")

    def read_uid_from_file(self, uid_file: str) -> int:
        if not os.path.exists(uid_file):
            raise RuntimeError(f"UID file {uid_file} not found.")
        with open(uid_file) as f:
            uid = f.readline().strip()
        return int(uid)

    def write_uid_to_file(self, uid: int, uid_file: str):
        with open(uid_file, 'w') as f:
            f.write(str(uid))
