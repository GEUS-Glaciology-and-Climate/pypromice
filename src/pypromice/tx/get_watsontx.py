#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:20:09 2022

Script to get L0tx transmission messages from the Watson River station using 
the tx module

@author: Penelope How, pho@geus.dk
"""
from argparse import ArgumentParser

from configparser import ConfigParser
import os, imaplib, email, unittest
from glob import glob
from datetime import datetime

from pypromice.tx import getMail, L0tx, sortLines


def parse_arguments_watson():
    parser = ArgumentParser(description="AWS L0 transmission fetcher for Watson River measurements")       
    parser.add_argument('-a', '--account', default=None, type=str, required=True, help='Email account .ini file')
    parser.add_argument('-p', '--password', default=None, type=str, required=True, help='Email credentials .ini file')                      
    parser.add_argument('-o', '--outpath', default=None, type=str, required=False, help='Path where to write output (if given)')           
    parser.add_argument('-f', '--formats', default=None, type=str, required=True, help='Path to Payload format .csv file')
    parser.add_argument('-t', '--types', default=None, type=str, required=True, help='Path to Payload type .csv file')  	
    parser.add_argument('-u', '--uid', default=None, type=str, required=True, help='Last AWS uid .ini file')	        
    args = parser.parse_args()
    return args

#------------------------------------------------------------------------------
def get_watsontx():
    """Executed from the command line"""
    args = parse_arguments_watson()
    
     # Set payload formatter paths
    formatter_file = args.formats
    type_file = args.types

 	# Set credential paths
    accounts_file = args.account
    credentials_file = args.password 

 	# Set last aws uid path
 	# last_uid = 1000000
    uid_file = args.uid
    
    # Set last aws uid path
    with open(uid_file, 'r') as last_uid_f:
        last_uid = int(last_uid_f.readline())
    
    # Set output file directory
    out_dir = args.outpath
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
#------------------------------------------------------------------------------
    
    # Define accounts and credentials ini file paths
    accounts_ini = ConfigParser()
    accounts_ini.read_file(open(accounts_file))
    accounts_ini.read(credentials_file)
                 
    # Get credentials
    account = accounts_ini.get('aws', 'account')
    server = accounts_ini.get('aws', 'server')
    port = accounts_ini.getint('aws', 'port')    
    password = accounts_ini.get('aws', 'password')
    if not password:
        password = input('password for AWS email account: ')
    print('AWS data from server %s, account %s' %(server, account))
    
#------------------------------------------------------------------------------
    
    # Log in to email server
    mail_server = imaplib.IMAP4_SSL(server, port)
    typ, accountDetails = mail_server.login(account, password)
    if typ != 'OK':
        print('Not able to sign in!')
        raise
        
    # Grab new emails
    result, data = mail_server.select(mailbox='"[Gmail]/All Mail"', 
                                      readonly=True)
    print('mailbox contains %s messages' %data[0])
    
#------------------------------------------------------------------------------
    
    # Get L0tx datalines from email transmissions
    for uid, mail in getMail(mail_server, last_uid=last_uid):
        message = email.message_from_string(mail)
        try:
            name = str(message.get_all('subject')[0])
            d = datetime.strptime(message.get_all('date')[0], 
                                  '%a, %d %b %Y %H:%M:%S %z')
        except:
            name=None
            d=None
        
        if name and 'Watson station' in name:
            print(f'Watson station message, {d.strftime("%Y-%m-%d %H:%M:%S")}')
            l0 = L0tx(message, formatter_file, type_file, 
                      sender_name=['emailrelay@konectgds.com','sbdservice'])
            
            if l0.msg: 
                out_fn = 'watson_station_tx.txt'
                out_path = os.sep.join((out_dir, out_fn))
        
                print(f'Writing to {out_fn}')
                print(l0.msg)
            
                with open(out_path, mode='a') as out_f:
                    out_f.write(l0.msg + '\n')    
    
#------------------------------------------------------------------------------

    # Sort L0tx files and add tails    
    for f in glob(out_dir+'/*.txt'):
        
        # Sort lines in L0tx file and remove duplicates
        in_dirn, in_fn = os.path.split(f)    
        out_fn = 'sorted_' + in_fn
        out_pn = os.sep.join((in_dirn, out_fn))
        sortLines(f, out_pn)
        
    # Close mail server if open
    if 'mail_server' in locals():
        print(f'\nClosing {account}')
        mail_server.close()
        resp = mail_server.logout()
        assert resp[0].upper() == 'BYE'
    
    # Write last aws uid to ini file
    try:
        with open(uid_file, 'w') as last_uid_f:
            last_uid_f.write(uid)
    except:
        print(f'Could not write last uid {uid} to {uid_file}')
         
    print('Finished')
        
if __name__ == "__main__":  
    get_watsontx()
