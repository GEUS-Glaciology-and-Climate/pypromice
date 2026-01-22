#!/usr/bin/env python

import logging
import os
import re
import sys
import time
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from glob import glob
from email.utils import parsedate_to_datetime

import toml

from pypromice.resources import (
    DEFAULT_PAYLOAD_FORMATS_PATH,
    DEFAULT_PAYLOAD_TYPES_PATH,
)
from pypromice.tx.email_client.base_mail_client import BaseMailClient
from pypromice.tx.email_client.imap_client import IMAPClient
from pypromice.tx.tx import L0tx


logger = logging.getLogger(__name__)


def parse_arguments_l0tx():
    parser = ArgumentParser(description="AWS L0 transmission fetcher")
    parser.add_argument('--account', '-a', type=str, required=True, help='Email account .ini file')
    parser.add_argument('--password', '-p', type=str, required=True, help='Email credentials .ini file')
    parser.add_argument('--config', '-c', type=str, required=True, help='Directory to config .toml files')
    parser.add_argument('--uidfilepath', '--uid', '-u', type=str, required=True, help='Last AWS uid .ini file')

    parser.add_argument('--outpath', '-o', default=None, type=str, required=False, help='Path where to write output (if given)')
    parser.add_argument('--awsname', '-n', default='*', type=str, required=False, help='name of the AWS to be fetched')
    parser.add_argument('--formats', '-f',default=DEFAULT_PAYLOAD_FORMATS_PATH, type=str, required=False, help='Path to Payload format .csv file')
    parser.add_argument('--types', '-t',default=DEFAULT_PAYLOAD_TYPES_PATH, type=str, required=False, help='Path to Payload type .csv file')

    parser.add_argument("--loglevel", default="INFO", help="Logging level")

    return parser.parse_args()


def get_l0tx(account, password, config, uid_file_path, outpath, awsname, formats, types):

    logger.info("Starting L0 transmission fetch")

    # Load modem configs from TOML
    toml_path = os.path.join(config, awsname + ".toml")
    toml_list = glob(toml_path)

    aws_modem_configurations = {}

    for t in toml_list:
        conf = toml.load(t)
        count = 1
        for m in conf["modem"]:
            name = f"{conf['station_id']}_{m[0]}_{count}"

            # Make start and end datetimes timezone-aware UTC
            start = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            if len(m[1:]) == 1:
                end = datetime.now(timezone.utc) + timedelta(hours=3)
            else:
                end = datetime.strptime(m[2], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

            aws_modem_configurations[name] = [start, end]
            logger.debug(f"Configured modem window: {name} → {aws_modem_configurations[name]}")
            count += 1

    # Prepare output directory
    out_dir = outpath
    if out_dir and not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Read last processed UID
    with open(uid_file_path) as f:
        last_uid = int(f.readline())
    logger.info(f"Last processed UID: {last_uid}")

    # Fetch emails
    with IMAPClient.from_config_files(account, password) as mail_client:
        uid = fetch_transmission_data(
            uid=last_uid,
            mail_client=mail_client,
            out_dir=out_dir,
            formatter_file=formats,
            type_file=types,
            aws_modem_configurations=aws_modem_configurations,
        )

    finalize_output_files(awsname, out_dir)

    # Save last UID
    try:
        with open(uid_file_path, "w") as f:
            f.write(str(uid))
    except Exception as e:
        logger.warning(f"Could not write last uid {uid}: {e}")

    logger.info("Finished L0 transmission fetch")


def fetch_transmission_data(
    uid,
    mail_client: BaseMailClient,
    out_dir,
    formatter_file,
    type_file,
    aws_modem_configurations,
):
    for uid, message in mail_client.iter_messages_since(uid):
        logger.debug(f"Processing new email {uid}")

        # Parse subject and date
        try:
            subject = message.get_all("subject")[0]
            date_raw = message.get_all("date")[0]

            imeis = re.findall(r"[0-9]+", subject)
            if len(imeis) != 1:
                raise ValueError(f"Expected single IMEI, got {imeis}")
            imei = imeis[0]

            d = parsedate_to_datetime(date_raw)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            else:
                d = d.astimezone(timezone.utc)

            logger.debug(f"Parsed email: IMEI={imei}, date={d}")

        except Exception as e:
            logger.warning(f"Skipping email {uid}: parse failure ({e})")
            continue

        matched_any = False

        for k, v in aws_modem_configurations.items():
            if imei not in k:
                continue
            if not (v[0] < d < v[1]):
                continue

            matched_any = True
            logger.info(f"AWS message for uid={uid} IMEI={imei} {d:%Y-%m-%d %H:%M:%S} L0_file={k}.txt")

            l0 = L0tx(message, formatter_file, type_file)

            if not l0.msg:
                logger.warning("L0tx produced no message")
                continue

            logger.debug(f"Raw L0 message: {repr(l0.msg)}")

            body = l0.getEmailBody()
            if body and "Lat" in body[0] and "Long" in body[0]:
                parts = body[0].split()
                lat = parts[parts.index("Lat") + 2]
                lon = parts[parts.index("Long") + 2]
                l0.msg = f"{l0.msg},{lat},{lon}"

            if out_dir:
                out_fn = f"{k}{l0.flag}.txt"
                out_path = os.path.join(out_dir, out_fn)
                logger.debug(f"Writing to {out_fn}")
                with open(out_path, "a") as f:
                    f.write(l0.msg + "\n")

        if not matched_any:
            logger.warning(f"No modem configuration matched uid={uid} IMEI={imei}")

        logger.debug("Finished processing email")

    return uid


def finalize_output_files(aws_name, out_dir):
    if not out_dir:
        return

    files = [
        f for f in glob(f"{out_dir}/{aws_name}*.txt")
        if isModified(f, 1)
    ]

    for f in files:
        sortLines(f, f)


def findDuplicates(lines):
    unique = list(set(lines))
    logger.info(f"{len(lines) - len(unique)} duplicates found")
    return unique


def sortLines(in_file, out_file, replace_unsorted=True):
    logger.info(f"Sorting {in_file}")

    with open(in_file) as f:
        lines = f.readlines()

    unique = findDuplicates(lines.copy())
    unique.sort()

    if lines != unique:
        with open(out_file, "w") as f:
            f.writelines(unique)


def isModified(filename, hours):
    return (time.time() - os.path.getmtime(filename)) / 3600 < hours

def main():
    args = parse_arguments_l0tx()

    logging.basicConfig(
        level=args.loglevel,
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        stream=sys.stdout,
    )
    logger.info("Logging initialized")

    get_l0tx(
        account=args.account,
        password=args.password,
        config=args.config,
        uid_file_path=args.uidfilepath,
        outpath=args.outpath,
        awsname=args.awsname,
        formats=args.formats,
        types=args.types,
    )

if __name__ == "__main__":
    main()
