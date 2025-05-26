import email.parser
import imaplib
import logging
from configparser import ConfigParser
from datetime import datetime
from mailbox import Message
from pathlib import Path
from typing import Iterator, List
import dateutil.parser

__all__ = ["GmailClient"]

logger = logging.getLogger(__name__)

class GmailClient:

    """
    A Gmail client for fetching and filtering messages using Gmail's IMAP interface.
    Domain-specific usage: identifies messages using Gmail UIDs and supports retrieval
    of Iridium SBD messages by IMEI number.
    """

    def __init__(self, server: str, port: int, account: str, password: str, chunk_size: int = 500):
        self.chunk_size = chunk_size
        self.parser = email.parser.Parser()
        logger.info("Logging in to email server...")
        self.mail_server = imaplib.IMAP4_SSL(server, port)
        status, account_details = self.mail_server.login(account, password)
        if status != "OK":
            logger.error("Unable to sign in!")
            raise Exception("Unable to sign in!")
        logger.info("Logged in to email server")

        result, data = self.mail_server.select(mailbox='"[Gmail]/All Mail"', readonly=True)
        logger.info('Mailbox "[Gmail]/All Mail" contains %s messages', data[0].decode())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mail_server.logout()

    # E-Mail download methods
    # --------------------------------------------------------------------------------------------
    def fetch_mail(self, uid: str) -> Message | None:
        """Fetch a single email by Gmail UID."""
        return self.fetch_mails([uid]).__next__()

    def fetch_mails(self, uids: list[str], chunk_size: int = 100) -> Iterator[Message]:
        """
        Fetch all messages corresponding to the given list of UIDs.
        The messages are returned in the same order as the UIDs.
        """
        yield from self._fetch_mail_chunks(uids)

    def _fetch_mail_chunks(self, uids: List[str]) -> Iterator[Message]:
        """Fetch mails in chunks and yield them as UID -> raw mail data mapping."""
        for i in range(0, len(uids), self.chunk_size):
            chunk = uids[i:i + self.chunk_size]
            uid_string = ','.join(map(str, uids))
            status, data = self.mail_server.uid("fetch", uid_string, "(RFC822)")
            if status != "OK":
                raise RuntimeError("Failed to fetch mail chunk")

            for part in data:
                if isinstance(part, tuple):
                    # part[0] is the UID, part[1] is the raw mail data
                    yield {part[0].decode(): part[1].decode()}
                else:
                    logger.warning(f"Unexpected part type: {type(part)} in chunk {chunk}")

    # UID query methods
    # --------------------------------------------------------------------------------------------
    def _imap_search_uids(self, search_string: str) -> List[str]:
        """
        Performs an IMAP UID search and returns a decoded list of string UIDs.
        This ensures compatibility with Gmail’s byte string responses.
        """
        status, data = self.mail_server.uid("search", None, search_string)
        if status != "OK":
            logger.error(f"Error searching for mails: {data}")
            return []
        return data[0].decode().split()

    def new_uids(self, last_uid:str) -> List[str]:
        """
        Get new Gmail UIDs that are greater than the last known UID.
        """
        first_uid = int(last_uid) + 1
        search_string = f"(UID {first_uid}:*)"
        return self._imap_search_uids(search_string)

    def uids_by_date(self, date: datetime | str) -> list[str]:
        """
        Get Gmail UIDs for messages received on a specific date.
        """
        if isinstance(date, str):
            # Convert string date to datetime object
            date = dateutil.parser.parse(date)

        search_string = f"(ON {date.strftime('%d-%b-%Y')})"
        return self._imap_search_uids(search_string)

    def uids_by_subject(self, subject: str) -> List[str]:
        """Return UIDs of messages with a specific subject."""
        search_string = f'(HEADER Subject "{subject}")'
        return self._imap_search_uids(search_string)

    def uids_by_sbd(self, imei: str) -> List[str]:
        """
        Return Gmail UIDs of Iridium SBD messages for the specified IMEI number.
        """
        return self.uids_by_subject(f"SBD Msg From Unit: {imei}")

    @classmethod
    def from_config(cls, *config_files: Path) -> 'GmailClient':
        """
        Initialize GmailClient from a list of config file paths.
        Expects config files with [imap] section and keys: server, port, account, password.
        """
        parser = ConfigParser()
        parser.read([str(p) for p in config_files])

        return cls(
            server=parser.get("aws", "server"),
            port=parser.getint("aws", "port"),
            account=parser.get("aws", "account"),
            password=parser.get("aws", "password"),
        )

