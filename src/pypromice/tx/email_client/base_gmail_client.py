from abc import ABC, abstractmethod
from typing import Iterator
import email

class BaseGmailClient(ABC):
    """Abstract base class for Gmail clients (IMAP or REST)."""

    @abstractmethod
    def iter_messages_since(self, last_uid: int) -> Iterator[email.message.Message]:
        """Yield messages since the given UID/historyId."""
        pass

    @abstractmethod
    def fetch_message(self, uid: int | str) -> email.message.Message:
        """Fetch a single message by UID/historyId."""
        pass

    @abstractmethod
    def get_latest_uid(self) -> int:
        """Return the latest UID/historyId in the mailbox."""
        pass

    @abstractmethod
    def read_uid_from_file(self, uid_file: str) -> int:
        pass

    @abstractmethod
    def write_uid_to_file(self, uid: int, uid_file: str):
        pass