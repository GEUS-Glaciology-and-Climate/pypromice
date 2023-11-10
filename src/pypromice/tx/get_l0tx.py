#!/usr/bin/env python
import email
import imaplib
import re
from argparse import ArgumentParser
from configparser import ConfigParser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import toml

from pypromice.tx import getMail, L0tx, sortLines


def get_l0tx_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(description="AWS L0 transmission fetcher")
    parser.add_argument('-a', '--account', required=True, help='Email account .ini file')
    parser.add_argument('-p', '--password', required=True, help='Email credentials .ini file')
    parser.add_argument('-c', '--config', required=True, help='Directory to config .toml files')
    parser.add_argument('-u', '--uid', required=True, help='Last AWS uid .ini file')
    parser.add_argument('-o', '--outpath', default=None, required=False,
                        help='Path where to write output (if given)')
    parser.add_argument('-n', '--name', type=str, default='*', required=False, help='name of the AWS to be fetched')
    parser.add_argument('-f', '--formats', required=False, help='Path to Payload format .csv file')
    parser.add_argument('-t', '--types', default=None, required=False, help='Path to Payload type .csv file')
    return parser


def get_l0tx(
        account_file: str,
        credentials_file: str,
        config_dir: str,
        uid_file: str,
        formatter_file: Optional[str] = None,
        name: str = '*',
        out_dir: Optional[str] = None,
        type_file: Optional[str] = None,
):
    toml_list = Path(config_dir).glob(name + '.toml')

    aws = {}
    for t in toml_list:
        conf = toml.load(t)
        count = 1
        for m in conf['modem']:
            name = str(conf['station_id']) + '_' + m[0] + '_' + str(count)

            if len(m[1:]) == 1:
                aws[name] = [datetime.strptime(m[1], '%Y-%m-%d %H:%M:%S'), datetime.now() + timedelta(hours=3)]
            elif len(m[1:]) == 2:
                aws[name] = [datetime.strptime(m[1], '%Y-%m-%d %H:%M:%S'), datetime.strptime(m[2], '%Y-%m-%d %H:%M:%S')]
            count += 1

    # ----------------------------------

    # Set output file directory
    if isinstance(out_dir, str):
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ----------------------------------

    # Define accounts and credentials ini file paths
    accounts_ini = ConfigParser()
    with open(account_file) as fp:
        accounts_ini.read_file(fp)
    with open(credentials_file) as fp:
        accounts_ini.read_file(fp)
    # Get last email uid
    with open(uid_file) as last_uid_f:
        last_uid = int(last_uid_f.readline())

    # Get credentials
    account = accounts_ini.get('aws', 'account')
    server = accounts_ini.get('aws', 'server')
    port = accounts_ini.getint('aws', 'port')
    password = accounts_ini.get('aws', 'password')
    if not password:
        password = input('password for AWS email account: ')
    print('AWS data from server %s, account %s' % (server, account))

    # ----------------------------------

    # Log in to email server
    mail_server = imaplib.IMAP4_SSL(server, port)
    typ, accountDetails = mail_server.login(account, password)
    if typ != 'OK':
        print('Not able to sign in!')
        raise Exception('Not able to sign in!')

    # Grab new emails
    result, data = mail_server.select(mailbox='"[Gmail]/All Mail"', readonly=True)
    print('mailbox contains %s messages' % data[0])

    # ----------------------------------

    # Get L0tx datalines from email transmissions
    uid = last_uid  # initialize using last_uid. If no messages, this will be written back to ini file.
    for uid, mail in getMail(mail_server, last_uid=last_uid):
        message = email.message_from_string(mail)

        try:
            imei, = re.findall(r'[0-9]+', message.get_all('subject')[0])
            d = datetime.strptime(message.get_all('date')[0], '%a, %d %b %Y %H:%M:%S %Z')
        except:
            imei = None
            d = None
        for k, v in aws.items():
            if str(imei) in k:
                if v[0] < d < v[1]:
                    print(f'AWS message for {k}.txt, {d.strftime("%Y-%m-%d %H:%M:%S")}')
                    l0 = L0tx(message, formatter_file, type_file)

                    if l0.msg:
                        body = l0.getEmailBody()
                        if 'Lat' in body[0] and 'Long' in body[0]:
                            lat_idx = body[0].split().index('Lat') + 2
                            lon_idx = lat_pos = body[0].split().index('Long') + 2
                            lat = body[0].split()[lat_idx]
                            lon = body[0].split()[lon_idx]
                            # append lat and lon to end of message
                            l0.msg = l0.msg + ',{},{}'.format(lat, lon)
                        print(l0.msg)

                        if out_dir is not None:
                            out_fn = str(k) + str(l0.flag) + '.txt'
                            out_path = Path(out_dir).joinpath(out_fn)
                            print(f'Writing to {out_fn}')

                            with out_path.open('a') as out_f:
                                out_f.write(l0.msg + '\n')

                            # ----------------------------------

    if out_dir is not None:
        # Sort L0tx files and add tails

        for file_path in Path(out_dir).glob(name + '*.txt'):
            # Sort lines in L0tx file and remove duplicates
            output_path = file_path.parent.joinpath('sorted_' + file_path.name)
            sortLines(
                file_path.as_posix(),
                output_path.as_posix(),
            )

        ## Generate tail files
        # out_dir = os.sep.join((in_dirn, 'tails'))
        # if not os.path.exists(out_dir):
        #     os.mkdir(out_dir)
        # imei = in_fn.split('.txt')[0].split('_')[1].split('-')[0]
        # name = imei_names.get(imei, 'UNKNOWN')
        # addTail(f, out_dir, name)

    # ---------------------------------

    # Close mail server if open
    if 'mail_server' in locals():
        print(f'\nClosing {account}')
        mail_server.close()
        resp = mail_server.logout()
        assert resp[0].upper() == 'BYE'

    # ---------------------------------

    # Write last aws uid to ini file
    try:
        with open(uid_file, 'w') as last_uid_f:
            last_uid_f.write(str(uid))
    except:
        print(f'Could not write last uid {uid} to {uid_file}')

    print('Finished')


if __name__ == "__main__":
    args = get_l0tx_argument_parser().parse_args()
    get_l0tx(
        account_file=args.account,
        credentials_file=args.password,
        config_dir=args.config,
        uid_file=args.uid,
        formatter_file=args.formats,
        name=args.name,
        out_dir=args.outpath,
        type_file=args.types,
    )
