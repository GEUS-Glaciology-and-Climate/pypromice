#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:20:09 2022

Script to get L0tx transmission messages from a single station using the 
tx module

@author: Penelope How, pho@geus.dk
"""

from configparser import ConfigParser
import os, imaplib, email, re, toml, sys
from glob import glob
from datetime import datetime, timedelta

from tx import getMail, L0tx, sortLines, addTail

# toml_list = glob('/home/pho/python_workspace/promice/aws-data/tx/config/Roof*.toml')
toml_list = glob('/home/pho/python_workspace/promice/PROMICE-AWS-processing/src/pypromice/test/config/*.toml')

aws={}
for t in toml_list:
    conf = toml.load(t)
    # try:
    count=1
    for m in conf['modem']:
        # name = str(conf['station_id'])
        name = str(conf['station_id']) + '_' + m[0] + '_' + str(count)
        
        if len(m[1:])==1:
            # aws[name] = [m[1], datetime.now()]
            aws[name] = [datetime.strptime(m[1],'%Y-%m-%d %H:%M:%S'), datetime.now() + timedelta(hours=3)]
        elif len(m[1:])==2:
            # aws[name] = m[1:]
            aws[name] = [datetime.strptime(m[1],'%Y-%m-%d %H:%M:%S'), datetime.strptime(m[2], '%Y-%m-%d %H:%M:%S')]
        count+=1

#------------------------------------------------------------------------------

# Set payload formatter paths
formatter_file = 'payload_formats.csv'
type_file = 'payload_types.csv'

# Set credential paths
accounts_file = '../../../working_scripts/credentials/accounts.ini'
credentials_file = "../../../working_scripts/credentials/credentials.ini" 

# Set modem names path
imei_file = '../../../working_scripts/credentials/imei2name.ini'  

# Set last aws uid path
# last_uid = 1000000
last_uid = 1185000

# Logger program path
# programs_dir = 'logger_programs/Freya2015.CR1'
# programs_dir = 'logger_programs/Promice2015e.CR1'
# print('parsing %s for message formats' % programs_dir)  

# Set output file directory
out_dir = '/home/pho/python_workspace/promice/PROMICE-AWS-processing/src/pypromice/test/aws_L0'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

#------------------------------------------------------------------------------

# Define accounts and credentials ini file paths
accounts_ini = ConfigParser()
accounts_ini.readfp(open(accounts_file))
accounts_ini.read(credentials_file) 
       
# Find and read modem identifiers
imei_ini = ConfigParser()
imei_ini.readfp(open(imei_file))
imei_names = dict(imei_ini.items('imei_to_name'))
  
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
result, data = mail_server.select(mailbox='"[Gmail]/All Mail"', readonly=True)
print('mailbox contains %s messages' %data[0])

#------------------------------------------------------------------------------

# Get L0tx datalines from email transmissions
for uid, mail in getMail(mail_server, last_uid=last_uid):
    message = email.message_from_string(mail)
    try:
        imei, = re.findall(r'[0-9]+', message.get_all('subject')[0])
        d = datetime.strptime(message.get_all('date')[0], 
                              '%a, %d %b %Y %H:%M:%S %Z')
    except:
        imei = None
        d = None
    
    for k,v in aws.items():
        if str(imei) in k:
            if v[0] < d < v[1]:
                print(f'AWS message for {name}.txt, {d.strftime("%Y-%m-%d %H:%M:%S")}')
                l0 = L0tx(message, formatter_file, type_file)
                name = imei_names.get(l0.imei, 'UNKNOWN')
                
                if l0.msg:            
                    out_fn = str(k) + str(l0.flag) + '.txt'
                    # out_fn = 'AWS_%s%s.txt' % (l0.imei, l0.flag)
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

    # # Generate tail files
    # out_dir = os.sep.join((in_dirn, 'tails')) 
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
    # imei = in_fn.split('.txt')[0].split('_')[1].split('-')[0]        
    # name = imei_names.get(imei, 'UNKNOWN')
    # addTail(f, out_dir, name)
    
# Close mail server if open
if 'mail_server' in locals():
    print(f'\nClosing {account}')
    mail_server.close()
    resp = mail_server.logout()
    assert resp[0].upper() == 'BYE'

# # Write last aws uid to ini file
# try:
#     with open(uid_file, 'w') as last_uid_f:
#         last_uid_f.write(uid)
# except:
#     print(f'Could not write last uid {uid} to {uid_file}')

        
print('Finished')