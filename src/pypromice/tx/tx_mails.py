import email.parser
from email.message import Message
from pathlib import Path

from partd.file import filename

from pypromice.tx.iridium import parse_mail
from pypromice.tx.payload_decoder import DecodeError, decode_payload


# %%


def fetch_mail(mail_uid: str) -> Message:
    pass


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
        filename_length = len(iridium_message.payload_filename.encode("ascii"))
        filename_length = len(iridium_message.payload_filename.encode("utf-8"))
        payload_length = len(iridium_message.payload_bytes)
        print(f"  Filename: {filename_length:5n} {iridium_message.payload_filename}")
        print(f"  Payload:  {payload_length:5n}")
        decoded_payload = decode_payload(iridium_message.payload_bytes)
    except DecodeError as e:
        print(f"Failed to decode payload for {iridium_message.imei}: {e}. Cause: {e.__cause__}")
        decoded_payload = None

    data_lines.append((iridium_message.imei, decoded_payload))


# %% Experiments
print(len(data_lines))

# %%
