#!/usr/bin/env python
from argparse import ArgumentParser
from configparser import ConfigParser
import os, imaplib, email, unittest
from glob import glob
from pypromice.tx import getMail


def parse_arguments_msg():
    parser = ArgumentParser(description="AWS message downloader")       
    parser.add_argument('-a', '--account', default=None, type=str, 
                        required=True, help='Email account .ini file')
    parser.add_argument('-p', '--password', default=None, type=str, 
                        required=True, help='Email credentials .ini file')                      
    parser.add_argument('-o', '--outpath', default='.', type=str, 
                        required=False, help='Path where to write output')    	
    parser.add_argument('-u', '--uid', default=None, type=str, 
                        required=False, help='Last uid, defined from files if not given')	
    parser.add_argument('-m', '--mailbox', default='"[Gmail]/All Mail"', 
                        type=str, required=False, help='Mailbox folder to collect messages from')        
    args = parser.parse_args()
    return args

def get_msg():
    args = parse_arguments_msg()

    # Set credential paths
    accounts_file = args.account
    credentials_file = args.password 
        
    # Set last aws uid path
    if args.uid is not None:
        last_uid = int(args.uid)
    else:
        last_uid=None
          
    
    # Set output file directory
    out_dir = args.outpath

    
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

    
    # Define last UID from file names
    if not os.path.exists(out_dir):
       os.mkdir(out_dir)
       if last_uid is None:
            last_uid=0
    else:
        if last_uid is None:
            fl = list(glob(os.path.join(out_dir,'*.msg')))
            fn = [int(f.split('_')[-1].split('.')[0]) for f in fl]
            last_uid = max(fn) 
        
    print('Fetching mail from uid %i' %last_uid)
   
    
    # Log in to email server
    mail_server = imaplib.IMAP4_SSL(server, port)
    typ, accountDetails = mail_server.login(account, password) 	
    if typ != 'OK':
        print('Not able to sign in!')
        raise

    # Grab new emails
    result, data = mail_server.select(mailbox='"[Gmail]/All Mail"', readonly=True)
    print('mailbox contains %s messages' %data[0])


    # Save mail to file
    for uid, mail in getMail(mail_server, last_uid=last_uid):
        message = email.message_from_string(mail)
        outfile = str(message['subject'])+'_'+str(uid)+'.msg'
          
        with open(os.path.join(out_dir, outfile), 'w') as out:
             out.write(mail)


    # Close mail server if open
    if 'mail_server' in locals():
        print(f'\nClosing {account}')
        mail_server.close()
        resp = mail_server.logout()
        assert resp[0].upper() == 'BYE'

    
    print('Finished')
        
if __name__ == "__main__":  
    get_msg()