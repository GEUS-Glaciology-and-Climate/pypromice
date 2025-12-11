from abc import ABC, abstractmethod
from typing import Iterator, Tuple
import email

class BaseMailClient(ABC):
    """Abstract base class for Mail clients (IMAP or Gmail REST)."""

    @abstractmethod
    def iter_messages_since(self, last_uid: int) -> Iterator[Tuple[int, email.message.Message]]:
        """
        Iterates over messages since the specified unique identifier (UID).

        This method retrieves email messages starting from a given UID and returns
        an iterator over tuples. Each tuple contains the UID of the message and the
        message itself as an `email.message.Message` object.

        Parameters
        ----------
        last_uid : int
            The unique identifier (UID) of the last email message. The method will
            iterate over messages received after this UID.

        Returns
        -------
        Iterator[Tuple[int, email.message.Message]]
            An iterator that yields tuples containing the UID of each email message
            and the corresponding `email.message.Message` object.
        """

        pass

    @abstractmethod
    def fetch_message(self, uid: int | str) -> email.message.Message:
        pass

    @abstractmethod
    def get_latest_uid(self) -> int:
        pass

    @abstractmethod
    def uids_by_date(self, date) -> list[str]:
        """Return list of message UIDs (or IDs) for emails on or after the given date."""
        pass
