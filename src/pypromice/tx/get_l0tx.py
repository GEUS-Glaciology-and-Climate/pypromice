#!/usr/bin/env python
import os
import re
import base64
import email
import toml
import logging
from glob import glob
from datetime import datetime, timedelta
from argparse import ArgumentParser

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request

from pypromice.tx import L0tx, sortLines, isModified

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------- Gmail API ----------------
def get_gmail_service(token_file, scopes=['https://www.googleapis.com/auth/gmail.readonly']):
    creds = Credentials.from_authorized_user_file(token_file, scopes)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(token_file, 'w') as f:
            f.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def read_uid_from_file(uid_file):
    if not os.path.exists(uid_file):
        raise RuntimeError(f"UID file {uid_file} not found — incremental fetch cannot proceed.")
    with open(uid_file, 'r') as f:
        uid = f.readline().strip()
        if not uid or not uid.isdigit():
            raise RuntimeError(f"UID file {uid_file} is invalid — must contain a valid history ID.")
        return int(uid)

def write_uid_to_file(uid, uid_file):
    try:
        with open(uid_file, 'w') as f:
            f.write(str(uid))
    except Exception as e:
        logger.error(f"Could not write last uid {uid} to {uid_file}: {e}")

# ---------------- Message Fetching ----------------
def get_new_messages(service, uid_file):
    """Yield new email.message.Message objects strictly since last UID with fail-safe."""
    uid = read_uid_from_file(uid_file)
    logger.info(f"Attempting incremental fetch since UID {uid} …")

    # Fail-safe: check against the latest message historyId
    results = service.users().messages().list(userId='me', maxResults=1).execute()
    recent_msgs = results.get('messages')
    if not recent_msgs:
        raise RuntimeError("No messages found in mailbox. Cannot proceed.")

    latest_history_id = int(service.users().messages().get(
        userId='me', id=recent_msgs[0]['id'], format='metadata'
    ).execute()['historyId'])

    if uid >= latest_history_id:
        logger.warning(
            f"Last UID {uid} >= Gmail's current latest historyId {latest_history_id}. "
            "No new messages to fetch."
        )
        return

    try:
        history = service.users().history().list(
            userId='me',
            startHistoryId=uid
        ).execute()
        messages = []
        for h in history.get('history', []):
            messages.extend(h.get('messages', []))
        latest_uid = history.get('historyId')
    except HttpError as e:
        if e.resp.status == 404:
            raise RuntimeError(
                f"History ID {uid} is too old or invalid. Cannot fetch incrementally."
            )
        else:
            raise

    fetched_count = 0
    for msg in messages:
        raw_msg = service.users().messages().get(userId='me', id=msg['id'], format='raw').execute()
        raw_bytes = base64.urlsafe_b64decode(raw_msg['raw'])
        msg_obj = email.message_from_bytes(raw_bytes)
        fetched_count += 1
        yield msg_obj

    if latest_uid:
        write_uid_to_file(latest_uid, uid_file)
        logger.info(f"Updated UID file with latest historyId {latest_uid}")
    logger.info(f"Total messages fetched: {fetched_count}")

# ---------------- AWS L0tx Processing ----------------
def load_aws_config(config_dir):
    aws = {}
    toml_files = glob(os.path.join(config_dir, '*.toml'))
    for tfile in toml_files:
        conf = toml.load(tfile)
        count = 1
        for modem in conf.get('modem', []):
            name = f"{conf['station_id']}_{modem[0]}_{count}"
            if len(modem[1:]) == 1:
                aws[name] = [datetime.strptime(modem[1], '%Y-%m-%d %H:%M:%S'),
                             datetime.now() + timedelta(hours=3)]
            elif len(modem[1:]) == 2:
                aws[name] = [datetime.strptime(modem[1], '%Y-%m-%d %H:%M:%S'),
                             datetime.strptime(modem[2], '%Y-%m-%d %H:%M:%S')]
            count += 1
    return aws

def process_messages(service, aws_config, uid_file, out_dir, payload_formats, payload_types):
    os.makedirs(out_dir, exist_ok=True)
    total_matches = 0
    processed_imeis = set()

    for msg_obj in get_new_messages(service, uid_file):
        subject = msg_obj.get('Subject', '')
        date_str = msg_obj.get('Date', '')
        imei_match = re.findall(r'\d+', subject)
        imei = imei_match[0] if imei_match else None

        try:
            d = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z')
        except Exception:
            d = None

        for k, v in aws_config.items():
            if str(imei) in k and v[0] < d < v[1]:
                logger.info(f"AWS message for {k}, {d.strftime('%Y-%m-%d %H:%M:%S')}")
                l0 = L0tx(msg_obj, payload_formats, payload_types)

                if l0.msg:
                    body = l0.getEmailBody()
                    if 'Lat' in body[0] and 'Long' in body[0]:
                        lat_idx = body[0].split().index('Lat') + 2
                        lon_idx = body[0].split().index('Long') + 2
                        lat = body[0].split()[lat_idx]
                        lon = body[0].split()[lon_idx]
                        l0.msg += f',{lat},{lon}'

                    out_fn = f"{k}{l0.flag}.txt"
                    out_path = os.path.join(out_dir, out_fn)
                    with open(out_path, 'a') as f:
                        f.write(l0.msg + '\n')

                    total_matches += 1
                    processed_imeis.add(imei)

    # Post-process modified files
    modified_files = [f for f in glob(os.path.join(out_dir, '*.txt')) if isModified(f, 1)]
    for f in modified_files:
        in_dir, in_fn = os.path.split(f)
        out_fn = f"sorted_{in_fn}"
        out_pn = os.path.join(in_dir, out_fn)
        sortLines(f, out_pn)
        logger.info(f"Sorted file written: {out_fn}")

    # Summary
    logger.info(f"Total AWS messages matched: {total_matches}")
    logger.info(f"Processed IMEIs: {', '.join(sorted(processed_imeis)) if processed_imeis else 'None'}")

# ---------------- Argument Parsing ----------------
def parse_arguments():
    parser = ArgumentParser(description="AWS L0 transmission fetcher")
    parser.add_argument('-a', '--token', required=True, type=str, help='Access token file')
    parser.add_argument('-o', '--outpath', required=True, type=str, help='Path to write output')
    parser.add_argument('-c', '--config', required=True, type=str, help='Directory to config .toml files')
    parser.add_argument('-f', '--formats', required=False, type=str, help='Path to Payload format .csv file')
    parser.add_argument('-t', '--types', required=False, type=str, help='Path to Payload type .csv file')
    parser.add_argument('-u', '--uid', required=True, type=str, help='Last AWS uid .ini file')
    return parser.parse_args()

# ---------------- Main ----------------
def main():
    args = parse_arguments()

    aws_config = load_aws_config(args.config)
    service = get_gmail_service(args.token)
    process_messages(
        service,
        aws_config,
        args.uid,
        args.outpath,
        args.formats,
        args.types
    )
    logger.info("Finished incremental fetch.")

if __name__ == "__main__":
    main()
