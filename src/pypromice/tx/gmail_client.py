import email.parser
import imaplib
from mailbox import Message
from typing import Iterator

__all__ = ["GmailClient"]

class GmailClient:

    def __init__(self, server: str, port: int, account: str, password: str, last_uid: int = 1):
        self.last_uid = last_uid
        self.parser = email.parser.Parser()
        self.mail_server = imaplib.IMAP4_SSL(server, port)
        typ, accountDetails = self.mail_server.login(account, password)
        if typ != "OK":
            print("Not able to sign in!")
            raise Exception("Not able to sign in!")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mail_server.logout()
        return False

    def fetch_mail(self, uid: int) -> Message | None:
        result, data = self.mail_server.uid("fetch", str(uid), "(RFC822)")
        if result != "OK":
            print(f"Error fetching mail with UID {uid}: {data}")
            return None
        if not data or len(data) < 2:
            print(f"No data returned for UID {uid}")
            return None

        mail_str = data[0][1].decode()
        return self.parser.parsestr(mail_str)


    def fetch_new_mails(self, last_uid:int|None = None) -> Iterator[Message]:
        if last_uid is None:
            last_uid = self.last_uid

        # Issue search command of the form "SEARCH UID 42:*"
        result, data = self.mail_server.uid("search", None, f"(UID {last_uid}:*)")
        message_uids = map(int, data[0].split())

        for message_uid in message_uids:
            # SEARCH command *always* returns at least the most
            # recent message, even if it has already been synced
            if message_uid > last_uid:
                yield self.fetch_mail(message_uid)
                self.last_uid = message_uid



