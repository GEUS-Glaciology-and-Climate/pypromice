#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AWS Level 0 (L0) data transmission fetching module
"""

from collections import deque
import email, re, os, os.path, time, unittest, calendar, imaplib, pkg_resources

# Set maximum number of email lines to read
imaplib._MAXLINE = 5000000
    
#------------------------------------------------------------------------------

class PayloadFormat(object):
    '''Payload formatter object'''
    
    def __init__(self, format_file=None, type_file=None):
        '''Payload formatter initialisation
        
        Parameters
        ----------
        format_file : str
            File path to payload formatter .csv file
        type_file : str
            File path to payload type .csv file
        '''
        self.payload_type = self.readType(type_file)
        self.payload_format = self.readFormatter(format_file)
        self._addCount()

    def readType(self, in_file, delimiter=','):                                #TODO read from .txt file as well as .csv
        '''Read payload type setter from file. Outputted dictionary set as
        key[type_letter]: number_of_bytes
        
        Parameters
        ----------
        in_file : str
            Input file path
        delimiter : str, optional
            File delimiter. The default is ","
        
        Returns
        -------
        payload_typ : dict
            Payload type information
        '''
        payload_typ = {}
        if in_file is None:
            lines = self.readPkgFile('tx/payload_types.csv')
        else:
            lines = self.readFile(in_file)
        for l in lines[1:]:
            info = l.split(delimiter)
            try:
                payload_typ[info[0]] = int(info[1])
            except IndexError:
                pass
        return payload_typ
    
    def readFormatter(self, in_file, delimiter=','):
        '''Read payload formatter from file. Outputted dictionary set as
        key[number]: [expected_length, format_characters, description].
        Flag column (info[4]) used to signify if entry should be written to 
        output
        
        Parameters
        ----------
        in_file : str
            Input file path
        delimiter : str, optional
            File delimiter. The default is ","
        
        Returns
        -------
        payload_fmt : dict
            Payload format information
        '''
        payload_fmt = {}
        if in_file==None:
            lines = self.readPkgFile('tx/payload_formats.csv')
        else:
            lines = self.readFile(in_file)
        for l in lines[1:]:
            info = l.split(delimiter)  
            try:                     
                if info[4]:
                    pass
                else:
                    payload_fmt[int(info[0])] = [int(info[1]), info[2], info[3]]
            except IndexError:
                pass
        return payload_fmt                 

    def readFile(self, in_file):
        '''Read lines from file
        
        Parameters
        ----------
        in_file : str
            Input file path
        
        Returns
        -------
        lines : list
            List of file line contents
        '''
        with open(in_file, 'r') as in_f:
            lines = in_f.readlines()
        return lines

    def readPkgFile(self, fname):
        '''Read lines from internal package file
        
        Parameters
        ----------
        fname : str
            Package file name
        
        Returns
        -------
        lines : list
            List of file line contents
        '''
        with pkg_resources.resource_stream('pypromice', fname) as stream:
            lines = stream.read().decode("utf-8")
            lines = lines.split("\n")  
        return lines
 
    def _addCount(self):
        '''Add counter to payload formatter'''
        for item in self.payload_format.items():
            key, val = item
            var_count, var_def, comment = val
            assert var_count == len(var_def)
            bytes_count = 0
            for var in var_def:
                bytes_count += self.payload_type[var.lower()]
            self.payload_format[key].append(bytes_count + 1)
#------------------------------------------------------------------------------

class SbdMessage(object):
    '''SBD transmission message object'''

    def __init__(self, content, attach, imei):
        '''SBD tranmission message initialisation
        
        Parameters
        ----------
        content : str
            Email message string
        attach : email.message.Message
            Attachment object
        imei : str
            Modem number of message
        '''          
        # Get email information
        self.imei = imei
        if content:
            self.momsn = self.getKeyValue(content, ': ', 'MOMSN')
            self.mtmsn = self.getKeyValue(content, ': ', 'MTMSN')
            self.cep_radius = self.getKeyValue(content, '= ', 'CEPradius')
            self.msg_size = self.getKeyValue(content, ': ', 'Message Size (bytes)')  
            self.session_time = self.getKeyValue(content, ': ', 
                                                 'Time of Session (UTC)', 
                                                 integer=False)
            self.status = self.getStatus(content)
            self.location = self.getLocation(content)
        
        # Create empty object if no email content
        else:
            self.momsn = None
            self.mtmsn = None
            self.cep_radius = None
            self.msg_size = None
            self.session_time = None
            self.status = None
            self.location = None
        
        # Get attachment information                                           #TODO read from downloaded file as alternative to attachment                  
        if attach:
            if self.checkAttachment(attach) and self.checkAttachmentName(attach.get_filename()): 
                self.payload = self.getPayloadFromEmail(attach, self.msg_size)
            else:
                self.payload = None
        else:
            self.payload = None

    def checkAttachment(self, attach):
        '''Check if attachment is present in email.message.Message object'''
        if attach == 0:
            print('No file attached to email')
            return False
        else:
            return True

    def checkAttachmentName(self, attach_file):
        '''Check if attachment is .sbd file'''
        root, ext = os.path.splitext(attach_file)
        if ext == '.sbd' or ext == '.dat':
            return True
        else:
            print(f'Unrecognised attachment file type: {attach_file}')
            return False
    
    def getPayloadFromEmail(self, attach, message_size):
        '''Get Sbd payload from email object'''
        if attach != None:
            sbd_payload = attach.get_payload(decode=True)
            if message_size:
                assert len(sbd_payload) == message_size                        #TODO Provide if statement in case message_size not provided
        else:
            sbd_payload = None
        return sbd_payload                    
    
    def getPayloadFromFile(self, attach):
        '''Read Sbd payload from .sbd file'''
        return readSBD(attach)
             
    def getKeyValue(self, content, seps, key, integer=True):
        '''Get attribute from email via keyword'''
        line = findLine(content, key)
        value = parseValue(line, seps)
        if integer:
            try:
                value = int(value)
            except:
                print(f'Value in line could not be passed as integer: "{line}"')
                value = None
        return value
    
    def getStatus(self, content, seps1=': ', seps2=' ', key='Session Status'):
        '''Get session status from email message'''
        line = findLine(content, key)
        try:
            value = parseValue(line, seps1).split(seps2)
            return [int(value[0]), value[2]]
        except:
            print(f'Session status not parsed from line "{line}"')
            return None
                
    def getLocation(self, content, seps=' ', key='Unit Location'):
        '''Get latitude longitude unit location from email message'''
        line = findLine(content, key)
        try:
            l = line.split(seps)
            if l[2].lower() == 'lat':
                lat = float(l[4])
            
            if l[5].lower() == 'long' or l[5].lower() == 'lon':
                lon = float(l[7])
            return [lat, lon]
        except:
            print(f'Unit location not parsed from line "{line}"')
            return None

#------------------------------------------------------------------------------

class EmailMessage(SbdMessage):                                                #TODO add initialisation from .sbd file  
    '''Email message object'''

    def __init__(self, email_msg, sender_name): 
        '''Email message initialisation
        
        Parameters
        ----------
        email_msg : email.message.Message
            Email message object
        sender_name : str or list
            Email sender name/s to filter by. Filtering by sender name does not
            occur if variable is None
        '''
        if self.checkEmail(email_msg):
            
            self._email_msg = email_msg
            
            # Parse email data
            self.email_data = {}
            self.email_data = self.getEmailInfo()                              #TODO Get metadata

            # Check sender 
            if self.checkSender(sender_name):
                content, attachment = self.getEmailBody()  
                SbdMessage.__init__(self, content, attachment, self.getIMEI())
            else:
                print('Email from unrecognised sender ' + 
                      f'"f{self.email_data["from"]}", no Sbd message read')
                SbdMessage.__init__(self, None, None, None)
            
        else:
            print('Empty email message')
            self._email_msg = None
            self.email_data = None
        
    def checkEmail(self, email_msg):
        '''Check if email is Message object'''
        if not isinstance(email_msg, email.message.Message):
            print(f'Email object not valid {type(email_msg)}. ' +
                  'Expecting email.message.Message object')
            return False
        else:
            return True

    def checkSender(self, sender_name):
        '''Check email message from field matches sender name or names'''
        sender = self.email_data['from']
        if isinstance(sender_name, str):
            flag = self._checkFrom(sender, sender_name)
        elif isinstance(sender_name, list):
            flag = False
            for s in sender_name:
                f = self._checkFrom(sender, s)
                if f==True: 
                    flag = True
        else:
            flag = True
        return flag

    def getEmailBody(self):
        '''Get email message body'''
        if self._email_msg.is_multipart():
            try:
                content, attachment = self._email_msg.get_payload()  
            except:
                content = self._email_msg.get_payload()[0]
                attachment = None
                
            assert not content.is_multipart()                                  #else the decode=True on the next line makes it 
            body = content.get_payload(decode=False)                           #return None and break the rest of the parsing
        else:
            body = self._email_msg.get_payload(decode=True)#[0]
            attachment=None
        return body, attachment
    
    def getIMEI(self):
        '''Get modem identifier from email subject string'''
        try:
            imei, = re.findall(r'[0-9]+', self.email_data['subject'])
        except:
            imei=None
        return imei
        
    def getEmailInfo(self):
        '''Parse message in email object'''
        email_data = {}
        for i in ['from', 'subject', 'date']:
            email_data[i] = self._getEmailData(i)[0]
        email_data['to'] = self._getEmailData('to')            
        return email_data
    
    def _getEmailData(self, descriptor):
        '''Get email data from descriptor'''
        return self._email_msg.get_all(descriptor)
    
    def _checkFrom(self, email_from, name):
        '''Check if name is in email sender string'''
        if name not in email_from:
            return False
        else: 
            return True

#------------------------------------------------------------------------------    

class L0tx(EmailMessage, PayloadFormat):
    '''L0 tranmission data object'''

    def __init__(self, email_msg, format_file=None, type_file=None, 
                 sender_name=['sbdservice', 'ice@geus.dk','emailrelay@konectgds.com'],    #TODO don't hardcode sender names?
                 UnixEpochOffset=calendar.timegm((1970,1,1,0,0,0,0,1,0)),
                 CRbasicEpochOffset = calendar.timegm((1990,1,1,0,0,0,0,1,0))):
        '''L0tx object initialisation. 
        
        A note on the epoch, the Win32 epoch is 1st Jan 1601 but MSC epoch is 
        1st Jan 1970 (MSDN gmtime docs), same as Unix epoch. Neither Python nor 
        ANSI-C explicitly specify any epoch but CPython relies on the 
        underlying C library. CRbasic instead has the SecsSince1990() function. 
        This should always evaluate to 0 in CPython on Win32
        
        Parameters
        ----------
        email_msg : email.message.Message
            Email message object
        format_file : str
            File path to payload formatter .csv file
        type_file : str
            File path to payload type .csv file
        sender_name : str or list
            Valid sender names in email
        UnixEpochOffset : int
            Unix/MSC epoch 
        CRbasicEpochOffset : int
            CRbasic epoch
        '''
        # Initialise EmailMessage and PayloadFormatter objects
        EmailMessage.__init__(self, email_msg, sender_name)
        PayloadFormat.__init__(self, format_file, type_file)
          
        # Set epoch offset
        self.EpochOffset = UnixEpochOffset + CRbasicEpochOffset

        # Process if payload is present and in a compliant format
        if self.checkPayload() and self.checkByte(self.payload[:1]):
            
            self.bin_val, self.bin_format, self.bin_name, self.bin_len, self.bin_idx, self.bin_valid = self.getFormat()
            msg = self.getDataLine() 
            if msg:
                if self.bin_valid: 
                    
                    if self.checkLength():
                        self.flag = '-F'
                    elif self.isDiagnostics(msg):
                        self.flag = '-D'
                    elif self.isObservations(msg):
                        self.flag = ''
                    else:
                        self.flag = '-X'
                else:
                    self.flag=''
                
                self.msg = msg
            
            else:
                self.msg = None
        else:
            self.msg = None
    
    def checkPayload(self):
        '''Check message payload'''
        if self.payload == None:
           print('No .sbd file attached to this message')
           return False
        else:
            return True

    def getFirstByte(self):
        '''Get first byte in payload'''
        return self.payload[:1]
        
    def getFormat(self):
        '''Get binary format type from first byte in payload
        
        Returns
        -------
        bval : int or None
            Format value
        bfor : str or None
            Format string characters
        bname : str or None
            Format name
        blength : int or None
            Expected format length
        bidx : int
            Format index
        bool
            Valid format flag
        '''
        if self.getFirstByte().isdigit() or (self.payload[:2] == '\n' and self.imei == 300234064121930):     #TODO needed?
            return None, None, None, None, -9999, False
        
        elif 'watson' in self.email_data['subject'].lower() or 'gios' in self.email_data['subject'].lower():
            return None, None, None, None, -9999, False    
       
        else:
            bidx = ord(self.getFirstByte()) 
            try:
                bval, bfor, bname, blength = self.payload_format[bidx]
                
                # if bidx==80:                                                   # TODO. This is a temporary workaround for QAS_Lv3 formatting being different. Needs a more permanent fix!
                #     print('Checking version 3...')
                #     if len(self.payload) != blength:
                #         print('Mismatched lengths found...')
                #         if len(self.payload) == 83:
                #             print('Fetching QAS_Lv3 specific formatting...')
                #             bval, bfor, bname, blength = self.payload_format[70]
                
                return bval, bfor, bname, blength, bidx, True
            except KeyError:
                return None, None, None, None, bidx, False
            
    def checkByte(self, b):
        '''Check byte format against payload formatter object'''
        if ord(b) not in self.payload_format:
            print('Unrecognized first byte %s' %hex(ord(b)))
            return False      
        else:    
            return True
    
    def checkLength(self):                                                   
        if len(self.payload) != self.bin_len:
            print(f'Message malformed: expected {self.bin_len} bytes, ' + 
                  f'found {len(self.payload)}. Missing values to be replaced by '+
                  '"?" and extra values dropped')
            return True
        else:
            return False
 
    def check2BitNAN(self, msg, type_letter, letter_flag=['g','n','e'], nan_value=8191):
        '''Check if byte is a 2-bit NAN. This occurs when the GPS data is not 
        available and the logger sends a 2-bytes NAN instead of a 4-bytes value
        '''
        if type_letter.lower() in letter_flag:
            ValueBytes = self.getByteValue(2, msg, self.bytecounter)
            try:
                if GFP2toDEC(ValueBytes) == nan_value:
                    return True
                else:                                
                    return False
            except:
                return False
        else:
            return False
  
    def isDiagnostics(self, DataLine):
        '''Flag if message is diagnostics'''
        return '!D' in DataLine[-5:-3] or self.bin_idx % 5 == 4             

    def isObservations(self, DataLine):
        '''Flag if message is observations''' 
        return '!M' in DataLine[-2:] or self.bin_valid
    
    def isSummer(self, DataLine):
        '''Flag if message is summer message'''
        return ('!S' in DataLine and '!M' in DataLine[-2:]) or self.bin_idx % 5 in (0, 1)
    
    def isWatsonObservation(self, DataLine):
        '''Flag if message is Watson River measurement'''
        return ('watson' in DataLine.lower() or 'gios' in DataLine.lower())
    
    def isWithInstance(self, DataLine):
        '''Flag if message is with instance'''
        return '!I' in DataLine[-5:-3] or (self.bin_idx % 5 in (1, 3) and self.bin_idx != -9999)

    def getByteValue(self, ValueBytesCount, BinaryMessage, idx):
        '''Get values from byte range in binary message'''
        ValueBytes=[]
        for i in range(0,ValueBytesCount):
            try:
                ValueBytes.append(ord(BinaryMessage[idx+i: idx+i+1]))
            except:
                print('No byte found')
        return ValueBytes
    
    def updateByteCounter(self, value):
        '''Update byte counter for decoding message'''
        self.bytecounter += value

    def writeEntry(self, entry, i):
        '''Write out comma-formatted data entry from message'''
        if i == self.bin_val-1:
            return entry
        else:
            return entry + ','        
               
    def getDataLine(self):                                                     #TODO clean up
        '''Get data line from transmission message
        
        Returns
        -------
        str or None
            Dataline string if found'''
        # Retrieve payload and prime object if valid binary message        
        bin_msg = self.payload[1:]
        if self.bin_valid and bin_msg:
            print('%s-%s (binary)' %(self.imei, self.momsn) , self.bin_name)

            dataline = ''
            self.bytecounter = 0
            
            # Iterate over binary message formatting string
            for i in range(0, self.bin_val):

                type_letter = self.bin_format[i]
                num_bytes = self.payload_type[type_letter.lower()]
                
                # Check if 2-bit NaN is present
                if self.check2BitNAN(bin_msg, type_letter):
                    dataline = dataline + self.writeEntry('NAN', i)
                    self.updateByteCounter(2)
                    self.bin_len -= 2
                
                # Get byte value
                else:                      
                    ValueBytes = self.getByteValue(num_bytes, 
                                                   bin_msg, self.bytecounter)
                    self.updateByteCounter(num_bytes)
                    
                    if len(ValueBytes) == 2:
                        Value = GFP2toDEC(ValueBytes)
                    elif len(ValueBytes) == 4:
                        Value = GLI4toDEC(ValueBytes)
                    else:
                        entry = '?'
                    
                    # Decode based on formatting string                        #TODO put this in payload_type file
                    if type_letter.lower()=='g':
                        entry = str(Value/100.0)
                        
                    elif type_letter.lower()=='n':
                        entry = str(Value/100000.0)
                        
                    elif type_letter.lower() =='e':
                        entry = str(Value/100000.0)  
                        
                    elif type_letter.lower()=='f':
                        if Value == 8191:
                            entry = 'NAN'
                        elif Value == 8190:
                            entry = 'INF'
                        elif Value == -8190 or Value == -8191:               
                            entry = '-INF'
                        else:
                            entry = str(Value)    
                            
                    elif type_letter.lower()=='l':
                        if Value in (-2147483648, 2147450879):
                            entry = 'NAN'
                        else:
                            entry = str(Value)
                    
                    elif type_letter.lower()=='t':
                        entry = time.strftime('%Y-%m-%d %H:%M:%S', 
                                              time.gmtime(Value + self.EpochOffset)) + ',' + str(Value)  
                    else:
                        entry = '?'
                                        
                    # Append value outputted dataline
                    if type_letter.isupper():                                  #TODO test RAWtoSTR
                        dataline = dataline + RAWtoSTR(ValueBytes)
                    else:
                        dataline = dataline + self.writeEntry(entry, i) 
                    
            return dataline
        
        else:
            try:
                bin_msg = '2' + bin_msg.decode('cp850')                        #TODO de-bug so first byte is passed (currently misses of the first "2" of the year e.g. "022" instead of "2022")
            except:
                bin_msg = ''
            if self.isDiagnostics(bin_msg):
                desc = f'{self.imei}-{self.momsn} ASCII generic diagnostic message'
            elif self.isObservations(bin_msg) and self.isSummer(bin_msg):
                desc = f'{self.imei}-{self.momsn} ASCII generic summer observations message'
            elif self.isObservations(bin_msg) and not self.isSummer(bin_msg):
                desc = f'{self.imei}-{self.momsn} ASCII generic winter observations message'
            elif self.isWatsonObservation(bin_msg):
                desc = 'Watson River observations message'
                bin_msg = bin_msg.split('"Smp"')[-1].replace('"', '')
                    
            else:
                desc=None
                
            if desc:
                if self.isWithInstance(bin_msg):
                    desc = desc + '(+ instant.)'
                print(desc)
                return bin_msg
            else:
                print('Unrecognized message format')
                return None
                   
#------------------------------------------------------------------------------

def GFP2toDEC(Bytes):
    '''Two-bit decoder
    
    Parameters
    ----------
    Bytes : list
        List of two values
    
    Returns
    -------
    float
        Decoded value
    '''
    # print('ValueBytes received ' + str(Bytes))
    msb = Bytes[0]
    lsb = Bytes[1]    
    Csign = -2*(msb & 128)/128 + 1
    CexpM = (msb & 64)/64
    CexpL = (msb & 32)/32
    Cexp = 2*CexpM + CexpL - 3
    Cuppmant = 4096*(msb & 16)/16 + 2048*(msb & 8)/8 + 1024*(msb & 4)/4 + 512*(msb & 2)/2 + 256*(msb & 1)
    Cnum = Csign * (Cuppmant + lsb)*10**Cexp
    return round(Cnum, 3)

def GLI4toDEC(Bytes):
    '''Four-bit decoder
    
    Parameters
    ----------
    Bytes : list
        List of four values
    
    Returns
    -------
    float
        Decoded value
    '''
    Csign = int(-2 * (Bytes[0] & 0x80) / 0x80 + 1)
    byte1 = Bytes[0] & 127
    byte2 = Bytes[1]
    byte3 = Bytes[2]
    byte4 = Bytes[3]    
    return Csign * byte1 * 0x01000000 + byte2 * 0x010000 + byte3 * 0x0100 + byte4

def RAWtoSTR(Bytes):
    '''Byte-to-string decoder
    
    Parameters
    ----------
    Bytes : list
        List of values
    
    Returns
    -------
        Decoded string characters
    '''
    us = [chr(byte) for byte in Bytes] #the unicode strings
    hs = ['0x{0:02X}'.format(byte) for byte in Bytes] #the hex strings
    bs = ['0b{0:08b}'.format(byte) for byte in Bytes] #the bit strings
    return '(%s = %s = %s)' %(' '.join(us), ' '.join(hs), ' '.join(bs))
    
def findLine(content, key):
    '''Find keyword in line
    
    Parameters
    ----------
    content : str
        String to find keyword in
    key : str
        Keyword to find in string
   
    Returns
    -------
    line : str
       Line that keyword appears on
    '''
    for line in content.splitlines():
        if key in line:   
            return line

def parseValue(line, seps):
    '''Parse last value from line according to separating characters
    
    Parameters
    ----------
    line : str
        String to split 
    sep : str
        Separator characters to split line by

    Returns
    -------
    value : str
        Value extracted from line
    '''
    try:
        value = line.split(seps)[-1]
    except:
        print(f'Value not read from line "{line}"')
        value = None
    return value

def getMail(mail_server, last_uid=1):
    '''Retrieve new mail

    Parameters
    ----------
    mail_server : imaplib.IMAP_SSL
        Mail server object
    last_uid : int, optional
        Mail uid to start retrieval from. The default is 1.

    Yields
    ------
    str
        Mail uid
    str
        Mail message
    '''
    # Issue search command of the form "SEARCH UID 42:*"
    command = '(UID {}:*)'.format(last_uid)
    result, data = mail_server.uid('search', None, command)
    messages = data[0].split()
    new_uids = data[0].decode()
    # drop the last_uid (it has already been processed)
    new_uids = new_uids.replace(str(last_uid), '')
    print('Newest UID: %s' % new_uids.split(' ')[-1])
    # print('new UIDs: %s' % new_uids)

    # Yield mails
    for message_uid in messages:
        
        # SEARCH command *always* returns at least the most
        # recent message, even if it has already been synced
        if int(message_uid) > last_uid:
            print(f'\nFetching mail {message_uid.decode()}')
            result, data = mail_server.uid('fetch', message_uid, '(RFC822)')
            
            # Yield raw mail body
            yield message_uid.decode(), data[0][1].decode() 

def loadMsg(fname):
    '''Load .msg email file into format compatible with EmailMessage and 
    SbdMessage objects
    
    Parameters
    ----------
    fname : str
        File path to .msg file
    
    Returns
    -------
    email.message.Message
        Email message object
    '''
    with open(fname, 'rb') as f:
        byte = f.read()
    return email.message_from_bytes(byte)

def saveMsg(msg, fname):
    '''Save email message object to .msg file
    
    Parameters
    ----------
    msg : email.message.Message
        Email object to save to file
    fname : str
        File path to outputted .msg file
    '''
    with open(fname, 'wb') as fp:
        fp.write(bytes(msg))
    
def readSBD(sbd_file):
    '''Read encoded .sbd transmission file
    
    Parameters
    ----------
    sbd_file : str
        Filepath to encoded .sbd file
    
    Returns
    -------
    data : bytes
        Transmission message byte object
    '''
    with open(sbd_file, 'rb') as file:
         data = file.readlines()  
    return data[0]

def findDuplicates(lines):
    '''Find duplicates lines in list of strings
    
    Parameters
    ---------
    lines : list
       List of strings
    
    Returns
    -------
    unique_lines : list
       List of unique strings
    '''
    unique_lines = list(set(lines))
    duplicates_count = len(lines) - len(unique_lines)
    print(f'{duplicates_count} duplicates found')   
    return unique_lines

def sortLines(in_file, out_file, replace_unsorted=True):                       #Formerly called sorter.py
    '''Sort lines in text file

    Parameters
    ----------
    in_file : str
        Input file path
    out_file : str
        Output file path
    replace_unsorted : bool, optional
        Flag to replace unsorted files with sorted files. The default is True.
    '''
    print(f'\nSorting {in_file}')
    
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
    '''Generate tails file from L0tx file

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
    '''
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
        print(f'Tails file written to {out_pn}')

def isModified(filename, time_threshold=1):
    '''Return flag denoting if file is modified within a certain timeframe
    
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
    '''
    delta = time.time() - os.path.getmtime(filename)
    delta = delta / (60*60)
    if delta < time_threshold:
        return True
    return False

def _loadTestMsg():
    '''Load test .msg email file'''
    with pkg_resources.resource_stream('pypromice', 'test/test_email') as stream:
        byte = stream.read()
    return email.message_from_bytes(byte)

#------------------------------------------------------------------------------
        
class TestTX(unittest.TestCase): 
    def testPayloadFormat(self):
        '''Test PayloadFormat object initialisation'''
        p = PayloadFormat()
        self.assertTrue(p.payload_format[30][0]==12)
    
    def testEmailMessage(self):   
        '''Test EmailMessage object initialisation from .msg file'''
        m = _loadTestMsg()
        e = EmailMessage(m, sender_name='sbdservice')
        self.assertEqual(e.momsn, 36820)
        self.assertEqual(e.msg_size, 27)
        self.assertTrue(e.imei in '300234061165160')
        self.assertFalse(e.mtmsn)
        
    def testL0tx(self):
        '''Test L0tx object initialisation'''
        m = _loadTestMsg()
        l0 = L0tx(m)
        self.assertTrue(l0.bin_valid)
        self.assertEqual(l0.bin_val, 12)
        self.assertTrue('tfffffffffff' in l0.bin_format)
        self.assertTrue('2022-07-25 10:00:00' in l0.msg)
        self.assertFalse('?' in l0.msg)

    def testCLIl0tx(self):
        '''Test get_l0tx CLI'''
        exit_status = os.system('get_l0tx -h')
        self.assertEqual(exit_status, 0)
 
    def testCLIwatson(self):
        '''Test get_watsontx CLI'''
        exit_status = os.system('get_watsontx -h')
        self.assertEqual(exit_status, 0)
 
    def testCLImsg(self):
        '''Test get_msg CLI'''
        exit_status = os.system('get_msg -h')
        self.assertEqual(exit_status, 0)
                   
if __name__ == "__main__":  
    unittest.main()
