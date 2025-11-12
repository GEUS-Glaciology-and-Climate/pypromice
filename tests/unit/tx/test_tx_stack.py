import unittest

# All the different states either maintains their own states or they are invoked with an explicit id set/list

def fetch_mails():
    # Download all new available emails from gmail and store then locally
    # Update a database for later querying
    # Maintain a local states of the latest mail - maybe just from the db
    pass

# Where are the messages classified?

def decode_data():
    # Process
    # Ite
    pass

def invoke_data_pipeline():
    # stids
    # Data files
    pass


class TXStackTestCase(unittest.TestCase):

    def test_mail_processing(self):

        fetch_mails()
        decode_data()
        invoke_data_pipeline()


        pass

