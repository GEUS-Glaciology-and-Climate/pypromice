"""
Script to decode Iridium messages received via email using IMAP or Gmail REST API.
Supports fetching by date and caching emails locally.
"""
from email.message import Message
from pathlib import Path
import email.parser
import logging
from datetime import datetime

from pypromice.tx.email_client.base_gmail_client import BaseGmailClient
from pypromice.tx.email_client.imap_client import IMAPClient
from pypromice.tx.email_client.rest_api_client import RestAPIClient
from pypromice.tx.email_parsing import iridium
from pypromice.tx.payload_decoding.payload_decoder import decode_payload

from pypromice.tx.email_client.uid_handler import (
    read_uid_from_file,
    write_uid_to_file
)

logger = logging.getLogger(__name__)


def process_mail(email_msg: Message) -> None:
    """Parse and decode Iridium email message."""
    iridium_message = iridium.parse_mail(email_msg)

    subject = iridium_message.subject.lower()
    if "watson" in subject:
        print("Watson payload is not yet implemented")
        return
    elif "gios" in subject:
        print("GIOS payload is not yet implemented")
        return
    elif iridium_message.payload_bytes[:1].isdigit():
        print("ASCII payload is not yet implemented")
        return
    else:
        decoded_data = decode_payload(iridium_message.payload_bytes)
        print(decoded_data)


class MailBucket:
    """Simple cache for storing fetched emails on disk."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.parser = email.parser.Parser()

    def _get_path(self, uid: str) -> Path:
        return self.root_dir / f"{uid}.msg"

    def __contains__(self, uid: str):
        return self._get_path(uid).exists()

    def get(self, uid: str) -> Message:
        if uid not in self:
            raise KeyError(f"Mail with UID {uid} not found in bucket.")
        with self._get_path(uid).open("r") as fp:
            return self.parser.parse(fp)

    def set(self, uid: str, value: Message):
        if not isinstance(value, Message):
            raise ValueError(f"Value must be an email.message.Message, got {type(value)}")
        with self._get_path(uid).open("w") as fp:
            fp.write(value.as_string())

    def __getitem__(self, uid: str) -> Message:
        return self.get(uid)

    def __setitem__(self, uid: str, value: Message):
        self.set(uid, value)


def fetch_and_cache_messages(mail_client: BaseGmailClient, mail_bucket: MailBucket, date: datetime) -> list[Message]:
    """Fetch new messages from client since the given date and store in bucket."""
    messages = []
    message_ids = mail_client.uids_by_date(date)

    for message_id in message_ids:
        if message_id not in mail_bucket:
            msg = mail_client.fetch_message(message_id)
            mail_bucket[message_id] = msg
        else:
            msg = mail_bucket[message_id]
        messages.append(msg)
    return messages


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # User configuration
    client_type = "imap"  # or "rest"
    email_cache_root = Path("/data/aws-cache/mails")
    email_cache_root.mkdir(parents=True, exist_ok=True)
    last_uid_path = Path("last_aws_uid.ini")
    start_date = datetime(2025, 1, 1)  # fetch messages since this date

    # ----------------- Initialize mail client -----------------
    if client_type == "imap":
        mail_client = IMAPClient(server="imap.gmail.com",
                                 port=993,
                                 account="your_email@gmail.com",
                                 password="your_password"
                                 )
    elif client_type == "rest":
        mail_client = RestAPIClient(token_file="/path/to/token.json")
    else:
        raise RuntimeError(f"Unknown client type: {client_type}")

    mail_bucket = MailBucket(email_cache_root)
    # ----------------------------------------------------------

    # Fetch messages
    messages = fetch_and_cache_messages(mail_client, mail_bucket, start_date)
    logger.info(f"Fetched {len(messages)} messages since {start_date.date()}")

    # Process messages
    for msg in messages:
        try:
            process_mail(msg)
        except Exception as e:
            logger.error(f"Failed to process mail: {e}")
            continue

    # Update last UID
    if messages:
        new_last_uid = max(getattr(m, "uid", getattr(m, "message_id", 0)) for m in messages)
        write_uid_to_file(new_last_uid, last_uid_path)

    logger.info("Processing completed.")
