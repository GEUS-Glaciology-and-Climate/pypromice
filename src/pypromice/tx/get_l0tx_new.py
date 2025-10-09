import email.parser
import logging
from mailbox import Message

from pathlib import Path
from typing import List

from pypromice.tx.gmail_client import GmailClient
from pypromice.tx.mail_storage import LocalMailStore

# %%

# Get new uuids
# Fetch new emails
# Assign the uuids to the correct stations
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

# %%

gmail_client = GmailClient.from_config(
    accounts_path,
    credentials_path,
)

# %%

last_uuid = "2379170"
# %%
new_uuids = gmail_client.new_uids(last_uid=last_uuid)
logger.info(f"Found {len(new_uuids)} new UIDs since last UUID {last_uuid}")

chunk_size = 100
for i in range(0, len(new_uuids), chunk_size):
    chunk = new_uuids[i:i + chunk_size]
    logger.info(len(chunk))

    missing_uuids = [u for u in chunk if u not in mail_store]
    for uid, message in zip(missing_uuids, gmail_client.fetch_mails(uids=missing_uuids)):
        mail_store[uid] = message


# %%
uids = gmail_client.uids_by_subject('Data from station RUS_R')

uids = uids[:200]

missing_uuids = [u for u in uids if u not in mail_store]
missing_mails = list(gmail_client.fetch_mails(uids=missing_uuids))
mail_store[missing_uuids] = missing_mails





# %%
uids = gmail_client.uids_by_sbd("300534062923450")
len(uids)

gmail_client.uids_by_date('2025-05-23')




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



