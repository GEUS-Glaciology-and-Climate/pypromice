"""
This is a draft of a script to decode Iridium messages received via email.
It is not fully functional and need more functionality to be added and logging.
"""

from gmail_client import GmailClient
from pypromice.tx import iridium
from pypromice.tx.payload_decoder import decode_payload


def main():
    server = 'imap.gmail.com'
    account = 'foobar@gmail.com'
    port = 993
    password = 'password'
    with GmailClient(server=server, account=account, port=port, password=password) as gmail_client:
        for mail in gmail_client.fetch_new_mails():
            # TODO: Consider a way to store and cache the emails. It might be relevant to integrate this function to GmailClient
            iridium_message = iridium.parse_mail(mail)

            if "watson" in iridium_message.subject.lower():
                # Watson payload
                print("Watson payload is not yet implemented")
                continue
            elif "gios" in iridium_message.subject.lower():
                # GIOS payload
                print("GIOS payload is not yet implemented")
                continue
            elif iridium_message.payload_bytes[:1].isdigit():
                # The values representing 0-9 (utf-8) are handled as a special case
                # where the payload is encoded as ascii
                print("ASCII payload is not yet implemented")
                continue
            else:
                # Binary payload
                # TODO: Use the time_of_session and imei number to determine the station id and column names
                decoded_data = decode_payload(iridium_message.payload_bytes)

                print(decoded_data)

if __name__ == "__main__":
    main()