import re
from datetime import datetime
from email.message import Message

import attrs

class IridiumParseError(Exception):
    """
    Exception raised when parsing an Iridium message fails.
    This is a custom exception to handle specific parsing errors.
    """
    pass


@attrs.define
class IridiumMessage:
    """
    Class to represent an Iridium message.
    """

    imei: str
    momsn: int
    mtmsn: int
    time_of_session: datetime
    payload_filename: str
    payload_bytes: bytes | None

    # This is the original subject line of the email
    # NOTE:
    # This is not a part of the Iridium message, but rather a part of the email a layer above
    # It is stored here for payload decoding purposes
    subject: str | None

def is_iridium(email: Message) -> bool:
    return email["From"] == "sbdservice@sbd.iridium.com"

def parse_mail(email: Message) -> IridiumMessage:
    subject = str(email["Subject"])

    sbd_raw = None
    sdb_dict = None
    data_message: bytes | None = None
    filename = None
    for payload_message in email.get_payload():
        filename: str = payload_message.get_filename()
        if filename is None:
            # There should only be a single payload without a filename
            assert sbd_raw is None
            sbd_raw = payload_message.get_payload()
            sdb_dict = dict(
                re.findall(r"^([^:\n]+):\s*([^\n\r]+)", sbd_raw, re.MULTILINE)
            )
        else:
            assert data_message is None
            data_message = payload_message.get_payload(decode=True)

    assert sdb_dict is not None
    assert data_message is not None
    assert filename is not None

    date_string = sdb_dict["Time of Session (UTC)"]
    date_object = datetime.strptime(date_string, "%a %b %d %H:%M:%S %Y")

    return IridiumMessage(
        imei=filename.split("_")[0],
        momsn=int(sdb_dict["MOMSN"]),
        mtmsn=int(sdb_dict["MTMSN"]),
        time_of_session=date_object,
        payload_bytes=data_message,
        payload_filename=filename,
        subject=subject,
    )
