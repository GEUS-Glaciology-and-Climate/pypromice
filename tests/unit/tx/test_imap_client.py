import tempfile
from email.message import Message
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pypromice.tx.email_client.imap_client import IMAPClient


@pytest.fixture
def mock_imap():
    with patch("imaplib.IMAP4_SSL") as mock:
        mock_connection = Mock()
        mock_connection.login.return_value = ("OK", [b"Logged in"])
        mock_connection.select.return_value = ("OK", [b"100"])
        mock.return_value = mock_connection
        yield mock_connection


@pytest.fixture
def imap_client(mock_imap):
    return IMAPClient("test.server", 993, "test@example.com", "INBOX","password")


def test_fetch_mail(imap_client, mock_imap):
    mock_imap.uid.return_value = ("OK", [(b"1 (UID 1)", b"Raw email data")])

    message = imap_client.fetch_message(1)

    assert isinstance(message, Message)


def test_new_uids(imap_client, mock_imap):
    mock_imap.uid.return_value = ("OK", [b"2 3 4"])

    uids = imap_client.new_uids(1)

    assert uids == [2, 3, 4]


def test_fetch_mails(imap_client, mock_imap):
    mock_imap.uid.return_value = ("OK", [(b"1 (UID 1)", b"Raw email data"), (b"2 (UID 2)", b"Raw email data")])

    mail_iterator = imap_client.fetch_mails([1, 2])

    mails = list(mail_iterator)
    assert len(mails) == 2
    assert isinstance(mails[0], tuple)
    assert isinstance(mails[0][1], Message)

def test_iter_messages_since():
    imap_client = Mock(spec_set=IMAPClient)
    imap_client.new_uids.return_value = [2, 3]
    imap_client.fetch_mails.return_value = [(2, Message()), (3, Message())]

    message_iterator = IMAPClient.iter_messages_since(imap_client, 1)

    messages = list(message_iterator)
    assert len(messages) == 2
    assert isinstance(messages[0], tuple)
    assert isinstance(messages[0][1], Message)
    assert imap_client.fetch_mails.call_count == 1
    assert imap_client.fetch_mails.call_args[0][0] == [2, 3]


def test_from_config_file(mock_imap):
    with tempfile.TemporaryDirectory() as config_dir:
        # Create two config files
        config_dir = Path(config_dir)
        config_file_01 = config_dir / "config_01.ini"
        config_file_02 = config_dir / "config_02.ini"
        with config_file_01.open("w") as f:
            f.write(
                "[aws]\n"
                "server = test.server\n"
                "port = 993\n"
                "account = test@example.com\n"
                "mailbox = INBOX\n"
                "password =\n"
            )
        with config_file_02.open("w") as f:
            f.write(
                "[aws]\n"        
                "password = the_secret_password\n"
            )

        # Instantiate the client
        imap_client = IMAPClient.from_config_files(config_file_01, config_file_02)

        # Assert the client was instantiated correctly
        assert isinstance(imap_client, IMAPClient)
        assert mock_imap.login.call_args[0] == ("test@example.com", "the_secret_password")

