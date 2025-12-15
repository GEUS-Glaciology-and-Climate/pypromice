#!/usr/bin/env python

import logging
import os
import re
import sys
import time
from argparse import ArgumentParser
from collections import deque
from datetime import datetime, timedelta
from glob import glob

import toml

from pypromice.resources import DEFAULT_PAYLOAD_FORMATS_PATH, DEFAULT_PAYLOAD_TYPES_PATH
from pypromice.tx.email_client.base_mail_client import BaseMailClient
from pypromice.tx.email_client.imap_client import IMAPClient
from pypromice.tx.tx import L0tx


logger = logging.getLogger(__name__)


def parse_arguments_l0tx():
    parser = ArgumentParser(description="AWS L0 transmission fetcher")
    parser.add_argument('--account', '-a', type=str, required=True, help='Email account .ini file')
    parser.add_argument('--password', '-p', type=str, required=True, help='Email credentials .ini file')
    parser.add_argument('--config', '-c', type=str, required=True, help='Directory to config .toml files')
    parser.add_argument('--uid', '-u', type=str, required=True, help='Last AWS uid .ini file')

    parser.add_argument('--outpath', '-o', default=None, type=str, required=False, help='Path where to write output (if given)')
    parser.add_argument('--awsname', '-n', default='*', type=str, required=False, help='name of the AWS to be fetched')
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

    aws_name = args.awsname

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
        uid = fetch_transmission_data(
            uid=last_uid,
            mail_client=mail_client,
            out_dir=out_dir,
            formatter_file=formatter_file,
            type_file=type_file,
            aws_modem_configurations=aws_modem_configurations,
        )

    finalize_output_files(aws_name, out_dir)

    # Write last aws uid to ini file
    # ----------------------------------
    try:
        with open(uid_file, "w") as last_uid_f:
            last_uid_f.write(str(uid))
    except:
        logger.warning(f'Could not write last uid {uid} to {uid_file}')

    logger.info("Finished")


def fetch_transmission_data(
        uid,
        mail_client: BaseMailClient,
        out_dir,
        formatter_file,
        type_file,
        aws_modem_configurations,
):
    """
    Fetches and processes messages from a mail inbox since a given UID, extracts relevant
    information based on the specified AWS modem configurations, and logs or stores the
    processed data in an output directory.

    This function iterates over messages received since the specified UID, parses the subject
    and date, checks for matching IMEI in the provided AWS modem configurations, validates
    date constraints, and processes the message content accordingly. If the conditions are
    met, it logs the message details and writes processed data to an output file if
    configured.

    Parameters
    ----------
    uid : int
        The unique identifier of the last processed email to start fetching messages from.
    mail_client : BaseMailClient
        An instance of a mail client providing access to the inbox and email messages.
    out_dir : str or None
        The path to the directory where processed messages will be written. If None, messages
        are not written to disk.
    formatter_file : Any
        The formatter file used to process the message content.
    type_file : Any
        The type file used to categorize or handle message processing.
    aws_modem_configurations : dict
        A mapping of IMEI keys to datetime range values, determining which messages to
        process based on their content and the timestamp in the AWS modem configurations.

    Returns
    -------
    int
        Returns the UID of the last processed message.
    """
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

    return uid


def finalize_output_files(aws_name, out_dir):
    """
    Finalize output files by sorting and deduplicating lines in text files.

    This function processes text files in the specified output directory that match
    the given AWS name and have been modified within the past hour. It sorts the
    lines in these files, removes duplicates, and generates new sorted files.

    Parameters
    ----------
    aws_name : str
        The prefix of the file names to process, typically related to the AWS instance name.
    out_dir : str or None
        The directory path where the files are located. If None, the function exits
        without processing any files.

    Notes
    -----
    - It only processes files matching the AWS name and `.txt` extension.
    - Modified files are identified based on a time window of the past hour.
    - The function appends "sorted_" to the filenames of the output files, which will
      contain sorted and deduplicated lines.
    """
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


def findDuplicates(lines):
    """Find duplicates lines in list of strings

    Parameters
    ---------
    lines : list
       List of strings

    Returns
    -------
    unique_lines : list
       List of unique strings
    """
    unique_lines = list(set(lines))
    duplicates_count = len(lines) - len(unique_lines)
    logger.info(f'{duplicates_count} duplicates found')
    return unique_lines


def sortLines(in_file, out_file, replace_unsorted=True):                       #Formerly called sorter.py
    """Sort lines in text file

    Parameters
    ----------
    in_file : str
        Input file path
    out_file : str
        Output file path
    replace_unsorted : bool, optional
        Flag to replace unsorted files with sorted files. The default is True.
    """
    logger.info(f'\nSorting {in_file}')

    # Open input file and read lines
    with open(in_file) as in_f:
        lines = in_f.readlines()

    # Remove duplicate lines and sort
    unique_lines = findDuplicates(lines.copy())
    unique_lines.sort()
    if lines != unique_lines:
        # Write sorted file
        with open(out_file, 'w') as out_f:
            # out_f.write(headers)
            out_f.writelines(unique_lines)

        # Replace input file with new sorted file
        if replace_unsorted:
            os.remove(in_file)
            os.rename(out_file, in_file)


def addTail(in_file, out_dir, aws_name, header_names='', lines_limit=100):
    """Generate tails file from L0tx file

    Parameters
    ----------
    in_file : str
        Input L0tx file
    out_dir : str
        Output directory for tails file
    aws_name : str
        AWS name
    header_names : str, optional
        Header names. The default is ''.
    lines_limit : int, optional
        Number of lines to append to tails file. The default is 100.
    """
    with open(in_file) as in_f:
        tail = deque(in_f, lines_limit)

    headers_lines = [l + '\n' for l in header_names.split('\n')]
    tail = list(set(headers_lines + list(tail)))
    tail.sort()

    out_fn = '_'.join((aws_name, in_file.split('/')[-1]))
    out_pn = os.sep.join((out_dir, out_fn))

    with open(out_pn, 'w') as out_f:
        #out_f.write(headers)
        out_f.writelines(tail)
        logger.info(f'Tails file written to {out_pn}')


def isModified(filename, time_threshold=1):
    """Return flag denoting if file is modified within a certain timeframe

    Parameters
    ----------
    filename : str
        File path
    time_threshold : int
        Time threshold (provided in hours)

    Returns
    -------
    bool
        Flag denoting if modified (True) or not (False)
    """
    delta = time.time() - os.path.getmtime(filename)
    delta = delta / (60*60)
    if delta < time_threshold:
        return True
    return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        stream=sys.stdout,
    )
    get_l0tx()
