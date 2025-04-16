import email.parser
import re
from datetime import datetime
from email.message import Message
from pathlib import Path

from pypromice.tx.iridium import IridiumMessage
from pypromice.tx.payload_decoder import decode_payload, DecodeError


# %%


def fetch_mail(mail_uid: str) -> Message:
    pass


def parse_mail(email: Message) -> IridiumMessage:
    assert email["From"] == "sbdservice@sbd.iridium.com"
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
                re.findall(r"^([^:\n]+):\s*([^\n]+)", sbd_raw, re.MULTILINE)
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


def decode_iridium_message(message: IridiumMessage) -> list:
    """
    Decode the payload of an Iridium message.
    Parameters
    ----------
    message

    Returns
    -------

    """

    is_watson = "watson" in message.subject.lower()
    is_gios = "gios" in message.subject.lower()
    # The values representing 0-9 (utf-8) are handled as a special case
    # where the payload is encoded as ascii
    is_ascii_payload = message.payload_bytes[:1].isdigit()

    # TODO: Break if one of the above is true
    assert not any(
        (is_watson, is_gios, is_ascii_payload)
    ), "This code does not handle Watson or GIOS payloads"

    return decode_payload(message.payload_bytes)


# %%
input_path = Path("/Users/maclu/data/tx_mails/mails")
output_payload_dir = Path("/Users/maclu/data/tx_mails/decoded")

output_payload_dir.mkdir(exist_ok=True, parents=True)

input_files = list(input_path.glob("*.eml"))
# %%
mail_path = input_files[36]
# %%

parser = email.parser.Parser()

iridium_messages = list()
for mail_path in input_files:

    with mail_path.open() as fp:
        email_message = parser.parse(fp=fp)

    iridium_message = parse_mail(email_message)
    iridium_messages.append(iridium_message)

len(iridium_messages)

# %%
data_lines = list()
for iridium_message in iridium_messages:
    print(iridium_message.imei)
    try:
        decoded_payload = decode_iridium_message(iridium_message)
    except DecodeError as e:
        print(f"Failed to decode payload for {iridium_message.imei}: {e}. Cause: {e.__cause__}")
        decoded_payload = None

    data_lines.append((iridium_message.imei, decoded_payload))


# %% Experiments


# %%
