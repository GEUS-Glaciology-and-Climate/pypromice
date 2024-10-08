import email.parser
import imaplib
import logging
from configparser import ConfigParser
from datetime import datetime
from mailbox import Message
from pathlib import Path
from typing import Iterator

__all__ = ["GmailClient"]

logger = logging.getLogger(__name__)

class GmailClient:

    def __init__(self, server: str, port: int, account: str, password: str, last_uid: int = 1, chunk_size: int = 500):

        self.last_uid = last_uid
        self.chunk_size = chunk_size
        self.parser = email.parser.Parser()
        logger.info("Logging in to email server...")
        self.mail_server = imaplib.IMAP4_SSL(server, port)
        typ, accountDetails = self.mail_server.login(account, password)
        if typ != "OK":
            logger.error("Not able to sign in!")
            raise Exception("Not able to sign in!")
        logger.info("Logged in to email server")

        result, data = self.mail_server.select(mailbox='"[Gmail]/All Mail"', readonly=True)
        logger.info("mailbox contains %s messages" % data[0])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mail_server.logout()
        return False


    def fetch_mail(self, uid: int) -> Message | None:
        result, data = self.mail_server.uid("fetch", str(uid), "(RFC822)")
        if result != "OK":
            logger.error(f"Error fetching mail with UID {uid}: {data}")
            return None
        if not data or len(data) < 2:
            logger.error(f"No data returned for UID {uid}")
            return None

        mail_str = data[0][1].decode()
        return self.parser.parsestr(mail_str)

    def get_mail_ids(self, date: datetime|str) -> list[int]:
        # Issue search command of the form "SEARCH UID 42:*"
        if isinstance(date, str):
            date = datetime.strptime(date, "%d-%b-%Y")

        result, data = self.mail_server.uid("search", None, f"(ON {date.strftime('%d-%b-%Y')})")
        if result != "OK":
            logger.error(f"Error searching for mails on {date}: {data}")
            return []
        message_uids = list(map(int, data[0].split()))
        logger.info(f"Found {len(message_uids)} messages on {date}")
        return message_uids

    def fetch_mails(self, uuids: list[int]) -> Iterator[Message]:
        logger.info(f"Fetching {len(uuids)} mails")
        batch_ids = ','.join(map(str, uuids))
        result, data = self.mail_server.uid("fetch", batch_ids, "(RFC822)")
        logger.info(f"Parsing string to Messages")
        if result != "OK":
            logger.info(f"Error fetching mails with UIDs {batch_ids}: {data}")
            return
        if not data or len(data) < 2:
            logger.warning(f"No data returned for UIDs {batch_ids}")
            return
        for response in data:
            if isinstance(response, tuple):
                mail_str = response[1].decode()
                message = self.parser.parsestr(mail_str)
                yield message

    def fetch_mails_chunked(self, uuids: list[int], chunk_size: int = 100) -> Iterator[Message]:
        """
        Fetch mails in chunks of size chunk_size.
        """
        for idx in range(0, len(uuids), chunk_size):
            message_uid_chunk = uuids[idx:idx + chunk_size]
            yield from self.fetch_mails(message_uid_chunk)

    def fetch_new_mails(self, last_uid:int|None = None) -> Iterator[Message]:
        if last_uid is None:
            last_uid = self.last_uid

        # Get the first 100 mails on 2025-01-01
        result, data = self.mail_server.uid("search", None, f"(ON 01-Apr-2025)")
        message_uuids = data[0].decode().split()
        len(message_uuids)

        # Issue search command of the form "SEARCH UID 42:*"
        result, data = self.mail_server.uid("search", None, f"(UID {last_uid}:*)")
        message_uids = list(map(int, data[0].split()))
        logger.info(f"Found {len(message_uids)} new messages since UID {last_uid}")

        for idx in range(0, len(message_uids), self.chunk_size):
            message_uid_chunk = message_uids[idx:idx + self.chunk_size]
            yield from self.fetch_mails(message_uid_chunk)


    @classmethod
    def from_config_files(cls, *init_file_path: Path) -> 'GmailClient':
        """
        Create a GmailClient instance from configuration files.

        The latest path takes precedence over the previous ones.

        """
        configuration = ConfigParser()
        for file_path in init_file_path:
            configuration.read(file_path)

        return cls(**configuration["aws"])

