import email.parser
import logging
from mailbox import Message

from pathlib import Path
from typing import List, Iterator

from pypromice.tx.gmail_client import GmailClient
from pypromice.tx.mail_storage import LocalMailStore, wrap_cache
from pypromice.tx import iridium
from pypromice.tx.tx_mails import iridium_message

# %%

# Get new uuids
# Fetch new emails
# Decode the iridium messages and store the payload under the imei numbers
# Assign the uids to the correct stations
# Decode the new email payloads
# Store the decoded payloads (l0/tx)
# Process the stations

# %%
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# %%

accounts_path = Path("/Users/maclu/data/credentials/accounts.ini")
credentials_path = Path("/Users/maclu/data/credentials/credentials.ini")

mail_cache_path = Path("/Users/maclu/data/tx_caches/mail")
mail_cache_path.mkdir(parents=True, exist_ok=True)
mail_store = LocalMailStore(root_path=mail_cache_path)

iridium_payload_cache_path = Path("/Users/maclu/data/tx_caches/iridium_payloads")
iridium_payload_cache_path.mkdir(parents=True, exist_ok=True)


# %%

gmail_client = GmailClient.from_config(
    accounts_path,
    credentials_path,
)

# %%

class TXProcessor:
    """
    A processor for Iridium SBD messages, fetching them from the Gmail client and storing them in the mail store.
    """

    def __init__(self, gmail_client: GmailClient, mail_store: LocalMailStore):
        self.gmail_client = gmail_client
        self.mail_store = mail_store

    def get_emails(self, uids: List[str]) -> Iterator[Message]:
        missing_uuids = [u for u in uids if u not in self.mail_store]
        print(len(missing_uuids))
        logger.info(f"Fetching {len(missing_uuids)} emails from Gmail client for UIDs")
        for uid, message in zip(missing_uuids, self.gmail_client.fetch_mails(uids=missing_uuids)):
            self.mail_store[uid] = message

        for uuid in uids:
            yield self.mail_store[uuid]

    def get_iridium_messages(self, uids: List[str]) -> Iterator[iridium.IridiumMessage]:
        """
        Fetch Iridium messages from the Gmail client and parse them.
        """
        for mail in self.get_emails(uids):
            if iridium.is_iridium(mail):
                yield iridium.parse_mail(mail)

tx_processor = TXProcessor(
    gmail_client=gmail_client,
    mail_store=mail_store,
)


# %%

# %%
last_uuid = "2383394"
new_uuids = gmail_client.new_uids(last_uid=last_uuid)
logger.info(f"Found {len(new_uuids)} new UIDs since last UUID {last_uuid}")

uids = new_uuids
# %%
uids = gmail_client.uids_by_sbd("300534062923450")
uids = uids[-400:]
# %%


iridium_messages = list()
for email in tx_processor.get_emails(uids):
    if not iridium.is_iridium(email):
        continue
    iridium_message = iridium.parse_mail(email)

    output_dir = iridium_payload_cache_path / iridium_message.imei
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / iridium_message.payload_filename
    with output_path.open("wb") as f:
        f.write(iridium_message.payload_bytes)

    iridium_messages.append(iridium_message)



# %% Other examples of using the GmailClient
# uids = gmail_client.uids_by_subject('Data from station RUS_R')
# gmail_client.uids_by_date('2025-05-23')

# %%
uids = gmail_client.uids_by_sbd("300534062923450")

latest_uids = uids[-10:]

for uid in latest_uids:
    if uid in mail_store:
        mail = mail_store[uid]
    else:
        mail = gmail_client.fetch_mail(uid)
        if mail is not None:
            mail_store[uid] = mail

iridium.parse_mail(mail)



# %%


# %%
last_uuid = "2379170"

gmail_client.fetch_mail(last_uuid)["Subject"]

new_uuids = gmail_client.new_uids(last_uid=last_uuid)

new_mails = list()
for uuid in new_uuids:
    if uuid in mail_store:
        mail = mail_store[uuid]
    else:
        mail = gmail_client.fetch_mail(uuid)
        if mail is not None:
            new_mails.append(mail)

missing_uuids = {u for u in new_uuids if u not in mail_store}
existing_uuids = {u for u in new_uuids if u in mail_store}
new_mails = list(gmail_client.fetch_mails(uids=missing_uuids))
mail_store[new_uuids] = new_mails



# %%
# %%

# %%
for uuid, mail in zip(new_uuids, new_mails):
    mail_store[uuid] = mail



# %%
# %%



