import datetime
import tempfile
from pathlib import Path

import pytest
from unittest.mock import patch, Mock

from pypromice.tx.email_client.base_mail_client import BaseMailClient
from pypromice.tx import get_l0tx
from pypromice.resources import DEFAULT_PAYLOAD_FORMATS_PATH, DEFAULT_PAYLOAD_TYPES_PATH


@pytest.fixture
def mock_L0tx():
    with patch("pypromice.tx.get_l0tx.L0tx") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance

def test_fetch_transmission_data(mock_L0tx):
    # Prepare mocks for mail clients and message parsing (L0tx)
    uid = Mock()
    mail_client = Mock(spec_set=BaseMailClient)
    message = Mock()
    message.get_all.side_effect = lambda arg: {
        'subject': ["10042010"],
        'date': ["Thu, 11 Dec 2025 14:30:00 CET"],
    }[arg]
    mail_client.iter_messages_since.return_value = [(1, message)]
    aws_modem_configurations = {
        'FOO_L_10042010': [
            datetime.datetime(2025,12,1),
            datetime.datetime(2026,12,2),
            ]
    }
    mock_L0tx.msg = '1,2,3,4'
    mock_L0tx.getEmailBody.return_value = ['']
    mock_L0tx.flag = 'a_flag'

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_path = Path(tmpdirname)

        get_l0tx.fetch_transmission_data(
            uid = uid,
            mail_client = mail_client,
            out_dir=output_path.as_posix(),
            formatter_file=DEFAULT_PAYLOAD_FORMATS_PATH,
            type_file=DEFAULT_PAYLOAD_TYPES_PATH,
            aws_modem_configurations=aws_modem_configurations
        )

        expected_output_path = output_path / 'FOO_L_10042010a_flag.txt'
        assert expected_output_path.exists()
        assert expected_output_path.read_text() == '1,2,3,4\n'
