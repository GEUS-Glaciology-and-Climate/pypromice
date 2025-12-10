#!/usr/bin/env python

import logging
import os
import re
import sys
from argparse import ArgumentParser
from datetime import datetime, timedelta
from glob import glob

import toml

from pypromice.resources import DEFAULT_PAYLOAD_FORMATS_PATH, DEFAULT_PAYLOAD_TYPES_PATH
from pypromice.tx.email_client.base_gmail_client import BaseGmailClient
from pypromice.tx.email_client.imap_client import IMAPClient
from pypromice.tx.tx import L0tx, sortLines, isModified

logger = logging.getLogger(__name__)


def parse_arguments_l0tx():
    parser = ArgumentParser(description="AWS L0 transmission fetcher")
    parser.add_argument('--account', '-a', type=str, required=True, help='Email account .ini file')
    parser.add_argument('--password', '-p', type=str, required=True, help='Email credentials .ini file')
    parser.add_argument('--config', '-c', type=str, required=True, help='Directory to config .toml files')
    parser.add_argument('--uid', '-u', type=str, required=True, help='Last AWS uid .ini file')

    parser.add_argument('--outpath', '-o', default=None, type=str, required=False, help='Path where to write output (if given)')
    parser.add_argument('--awsname','--name', '-n', default='*', type=str, required=False, help='name of the AWS to be fetched')
    parser.add_argument('--formats', '-f',default=DEFAULT_PAYLOAD_FORMATS_PATH, type=str, required=False, help='Path to Payload format .csv file')
    parser.add_argument('--types', '-t',default=DEFAULT_PAYLOAD_TYPES_PATH, type=str, required=False, help='Path to Payload type .csv file')
    args = parser.parse_args()
    return args


def get_l0tx():
    args = parse_arguments_l0tx()
    toml_path = os.path.join(args.config, args.awsname + ".toml")
    toml_list = glob(toml_path)

    aws_modem_configurations = {}
    for t in toml_list:
        conf = toml.load(t)
        count = 1
        for m in conf["modem"]:
            name = str(conf["station_id"]) + "_" + m[0] + "_" + str(count)

            if len(m[1:]) == 1:
                aws_modem_configurations[name] = [
                    datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S"),
                    datetime.now() + timedelta(hours=3),
                ]
            elif len(m[1:]) == 2:
                aws_modem_configurations[name] = [
                    datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S"),
                    datetime.strptime(m[2], "%Y-%m-%d %H:%M:%S"),
                ]
            count += 1

    # ----------------------------------
    # Set payload formatter paths
    formatter_file = args.formats
    type_file = args.types
    # Set credential paths
    accounts_file = args.account
    credentials_file = args.password
    # Set last aws uid path
    uid_file = args.uid

    # Set output file directory
    out_dir = args.outpath
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    # ----------------------------------
    # Get last email uid
    with open(uid_file, "r") as last_uid_f:
        last_uid = int(last_uid_f.readline())

    # ----------------------------------
    with IMAPClient.from_config_files(
        accounts_file,
        credentials_file,
    ) as mail_client:
        uid = fetch(
            uid=last_uid,
            mail_client=mail_client,
            out_dir=out_dir,
            formatter_file=formatter_file,
            type_file=type_file,
            aws_modem_configurations=aws_modem_configurations,
            aws_name=args.awsname,
        )

    # Write last aws uid to ini file
    # ----------------------------------
    try:
        with open(uid_file, "w") as last_uid_f:
            last_uid_f.write(str(uid))
    except:
        logger.warning(f'Could not write last uid {uid} to {uid_file}')

    logger.info("Finished")


def fetch(
        uid,
        mail_client: BaseGmailClient,
        out_dir,
        formatter_file,
        type_file,
        aws_modem_configurations,
        aws_name='*',
):
    # Get L0tx datalines from email transmissions
    #for uid, mail in getMail(mail_server, last_uid=last_uid):
    #    message = email.message_from_string(mail)

    for uid, message in mail_client.iter_messages_since(uid):

        try:
            (imei,) = re.findall(r"[0-9]+", message.get_all("subject")[0])
            d = datetime.strptime(
                message.get_all("date")[0], "%a, %d %b %Y %H:%M:%S %Z"
            )
        except:
            imei = None
            d = None
        for k, v in aws_modem_configurations.items():
            if str(imei) in k:
                if v[0] < d < v[1]:
                    logger.info(
                        f'AWS message for {k}.txt, {d.strftime("%Y-%m-%d %H:%M:%S")}'
                    )
                    l0 = L0tx(message, formatter_file, type_file)
                    
                    if l0.msg:
                        body = l0.getEmailBody()
                        if "Lat" in body[0] and "Long" in body[0]:
                            lat_idx = body[0].split().index("Lat") + 2
                            lon_idx = lat_pos = body[0].split().index("Long") + 2
                            lat = body[0].split()[lat_idx]
                            lon = body[0].split()[lon_idx]
                            # append lat and lon to end of message
                            l0.msg = l0.msg + ",{},{}".format(lat, lon)
                        logger.info(l0.msg)

                        if out_dir is not None:
                            out_fn = str(k) + str(l0.flag) + ".txt"
                            out_path = os.sep.join((out_dir, out_fn))
                            logger.info(f"Writing to {out_fn}")

                            with open(out_path, mode="a") as out_f:
                                out_f.write(l0.msg + "\n")

    # ----------------------------------

    if out_dir is not None:

        # Find modified files (within the last hour)
        mfiles = [
            mfile
            for mfile in glob(out_dir + "/" + aws_name + "*.txt")
            if isModified(mfile, 1)
        ]

        # Sort L0tx files and add tails
        for f in mfiles:

            # Sort lines in L0tx file and remove duplicates
            in_dirn, in_fn = os.path.split(f)
            out_fn = "sorted_" + in_fn
            out_pn = os.sep.join((in_dirn, out_fn))
            sortLines(f, out_pn)

    # ---------------------------------
    return uid


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        stream=sys.stdout,
    )
    get_l0tx()
