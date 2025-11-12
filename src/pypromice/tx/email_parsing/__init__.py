from mailbox import Message

import iridium

def parse_email(email: Message) -> iridium.IridiumMessage:
    if iridium.is_iridium(email):
        message = iridium.parse_mail(email)

        return message

    else:
        sender = email.get("From")
        subject = email.get("Subject")
        raise Exception(f"Email not supported. Sender: {sender}, Subject: {subject}")





