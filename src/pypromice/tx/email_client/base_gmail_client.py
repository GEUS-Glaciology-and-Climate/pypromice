from abc import ABC, abstractmethod
from typing import Iterator
import email

class BaseGmailClient(ABC):
    """Abstract base class for Gmail clients (IMAP or REST)."""

    @abstractmethod
    def iter_messages_since(self, last_uid: int) -> Iterator[email.message.Message]:
        pass

    @abstractmethod
    def fetch_message(self, uid: int | str) -> email.message.Message:
        pass

    @abstractmethod
    def get_latest_uid(self) -> int:
        pass

    @abstractmethod
    def read_uid_from_file(self, uid_file: str) -> int:
        pass

    @abstractmethod
    def write_uid_to_file(self, uid: int, uid_file: str):
        pass

    @abstractmethod
    def uids_by_date(self, date) -> list[str]:
        """Return list of message UIDs (or IDs) for emails on or after the given date."""
        pass
