"""
This is a draft of a script to decode Iridium messages received via email.
It is not fully functional and need more functionality to be added and logging.
"""
from email.message import Message
from pathlib import Path

from gmail_client import GmailClient
from pypromice.tx import iridium
from pypromice.tx.payload_decoder import decode_payload
from datetime import datetime
import email.parser
import logging


logger = logging.getLogger(__name__)

def process_mail(email: Message) -> None:
        # %%
        # TODO: Consider a wsay to store and cache the emails. It might be relevant to integrate this function to GmailClient
        iridium_message = iridium.pare_mail(email)

        # %%
        if "watson" in iridium_message.subject.lower():
            # Watson payload
            print("Watson payload is not yet implemented")
            return
        elif "gios" in iridium_message.subject.lower():
            # GIOS payload
            print("GIOS payload is not yet implemented")
            return
        elif iridium_message.payload_bytes[:1].isdigit():
            # The values representing 0-9 (utf-8) are handled as a special case
            # where the payload is encoded as ascii
            print("ASCII payload is not yet implemented")
            return
        else:
            # Binary payload
            # TODO: Use the time_of_session and imei number to determine the station id and column names
            decoded_data = decode_payload(iridium_message.payload_bytes)

            print(decoded_data)

def main(mail_client: GmailClient):
    for mail in mail_client.fetch_new_mails():
        try:
            process_mail(mail)
        except Exception as e:
            print(f"Failed to process mail: {e}")
            continue

# %%
class MailBucket:

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.parser = email.parser.Parser()

    def _get_path(self, uid: str) -> Path:
        return self.root_dir.joinpath(f"{uid}.msg")

    def __contains__(self, uid: str):
        return self._get_path(uid).exists()


    def get(self, uid: str) -> Message:
        if uid not in self:
            raise KeyError(f"Mail with UID {uid} not found in bucket.")
        with self._get_path(uid).open("r") as fp:
            return self.parser.parse(fp)

    def set(self, uid: str, value: Message):
        if not isinstance(value, Message):
            raise ValueError(f"Value must be an instance of email.message.Message, got {type(value)}")
        with self._get_path(uid).open("w") as fp:
            fp.write(value.as_string())

    def __getitem__(self, uid: str) -> Message:
        return self.get(uid)

    def __setitem__(self, uid: str, value: Message):
        self.set(uid, value)

# %%


if __name__ == "__main__":
    # %%
    # Example usage
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # %%

    credentials_dir = Path("/Users/maclu/work/aws-operational-processing/credentials")
    accounts_path = credentials_dir.joinpath("accounts.ini")
    credentials_path = credentials_dir.joinpath("credentials.ini")

    last_uid_path = Path("/Users/maclu/data/aws-l0/tx/last_aws_uid.ini")
    with last_uid_path.open("r") as last_uid_f:
        last_uid = int(last_uid_f.readline())

    mail_client = GmailClient.from_config(accounts_path, credentials_path)

    # %%
    email_cache_root = Path("/Users/maclu/data/aws-cache/mails")
    email_cache_root.mkdir(parents=True, exist_ok=True)
    mail_bucket = MailBucket(email_cache_root)

    # %%
    self = mail_client
    last_uid = 1

    message_ids = self.uids_by_date(date=datetime(2025, 1, 1))

    message_ids_missing = [uid for uid in message_ids if uid not in mail_bucket]
    for message in mail_client.fetch_mails(message_ids_missing):
        mail_bucket[message.uid] = message


    messages = list(self.fetch_mails(message_ids))

    message_id_mapping = dict(zip(message_ids, messages))

    for id, message in message_id_mapping.items():
        with email_cache_root.joinpath(f"{id}.msg").open("w") as fp:
            fp.write(message.as_string())



    uid = message_ids[312]
    m0 = message_id_mapping[uid]
    m1 = mail_bucket[uid]


    iridium.parse_mail(m0) == iridium.parse_mail(m1)

    # %%
    # mail_client.last_uid = last_uid
    # main(mail_client)

    # logger.info("Processing completed.")