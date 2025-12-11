import os
import base64
import email
import logging
from typing import Iterator, List
from datetime import datetime

try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from google.auth.transport.requests import Request
except ImportError as e:
    raise ImportError(
        "The Google API dependencies are missing and required to use this feature.\n\n"
        "Install them with `pip install pypromice[google]`"
    ) from e

from pypromice.tx.email_client.base_mail_client import BaseMailClient

logger = logging.getLogger(__name__)

class RestAPIClient(BaseMailClient):
    """Gmail REST API client implementing BaseMailClient interface."""

    def __init__(self, token_file: str, scopes: List[str] | None = None):
        self.scopes = scopes or ['https://www.googleapis.com/auth/gmail.readonly']
        self.token_file = token_file
        self.creds = self._load_credentials()
        self.service = build('gmail', 'v1', credentials=self.creds)

    # ---------------- BaseGmailClient methods ----------------
    def iter_messages_since(self, last_uid: int) -> Iterator[email.message.Message]:
        history = self.get_new_messages(last_uid)
        if history:
            yield from history

    def fetch_message(self, message_id: str) -> email.message.Message:
        return self.get_message_by_id(message_id)

    def get_latest_uid(self) -> int:
        return self.get_latest_history_id()

    # ---------------- REST utilities ----------------
    def _load_credentials(self) -> Credentials:
        if not os.path.exists(self.token_file):
            raise RuntimeError(f"Token file {self.token_file} not found.")

        creds = Credentials.from_authorized_user_file(self.token_file, self.scopes)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(self.token_file, 'w') as f:
                f.write(creds.to_json())
        return creds

    def get_new_messages(self, last_uid: int) -> Iterator[email.message.Message]:
        """Incremental fetch since last UID (historyId)."""
        try:
            history = self.service.users().history().list(
                userId='me',
                startHistoryId=last_uid
            ).execute()
        except HttpError as e:
            if e.resp.status == 404:
                raise RuntimeError("History ID too old/invalid.")
            else:
                raise

        messages = []
        for h in history.get('history', []):
            messages.extend(h.get('messages', []))

        for msg in messages:
            raw_msg = self.service.users().messages().get(
                userId='me', id=msg['id'], format='raw'
            ).execute()
            raw_bytes = base64.urlsafe_b64decode(raw_msg['raw'])
            yield email.message_from_bytes(raw_bytes)

    def get_message_by_id(self, message_id: str) -> email.message.Message:
        raw_msg = self.service.users().messages().get(
            userId='me', id=message_id, format='raw'
        ).execute()
        raw_bytes = base64.urlsafe_b64decode(raw_msg['raw'])
        return email.message_from_bytes(raw_bytes)

    def uids_by_date(self, date: datetime) -> list[str]:
        """Return message IDs for emails since the given date."""
        query = f"after:{int(date.timestamp())}"  # Gmail API uses UNIX timestamps in seconds
        results = self.service.users().messages().list(userId='me', q=query).execute()
        messages = results.get('messages', [])
        return [msg['id'] for msg in messages]

    def get_latest_history_id(self) -> int:
        results = self.service.users().messages().list(userId='me', maxResults=1).execute()
        messages = results.get('messages', [])
        if not messages:
            return 0
        msg_meta = self.service.users().messages().get(
            userId='me', id=messages[0]['id'], format='metadata'
        ).execute()
        return int(msg_meta['historyId'])
