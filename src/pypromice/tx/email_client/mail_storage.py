import email.parser
from mailbox import Message
from pathlib import Path
from typing import MutableMapping, List

import attr

__all__ = ["LocalMailStore"]


@attr.define
class LocalMailStore(MutableMapping[str, Message]):
    """
    A local mail store that uses the filesystem to store email messages.
    Messages are stored as .eml files named by their keys.
    """
    root_path: Path = attr.field()

    def __delitem__(self, key: str, /):
        """
        Remove a message from the store by its key.
        """
        path = self.get_path(key)
        if not path.exists():
            raise KeyError(f"Message with key {key} not found in the store.")
        path.unlink()

    def __len__(self):
        """
        Get the number of messages in the store.
        """
        return len(list(self.root_path.glob("*.eml")))

    def __iter__(self):
        """
        Iterate over the keys of messages in the store.
        """
        for path in self.root_path.glob("*.eml"):
            yield path.stem


    def get_path(self, key: str) -> Path:
        """
        Get the path to the message file.
        """
        return self.root_path / f"{key}.eml"

    def __contains__(self, key: str) -> bool:
        """
        Check if the message file exists in the store.
        """
        return self.get_path(key).exists()


    def __getitem__(self, key: str|List[str]) -> Message | List[Message]:
        """
        Get a message or a list of messages from the store.
        """
        if isinstance(key, list):
            return [self[u] for u in key]

        path = self.get_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Message with key {key} not found in store.")
        with path.open() as f:
            return email.parser.Parser().parse(f)


    def __setitem__(self, key: str|List[str], message: Message|List[Message]):
        """
        Set a message or a list of messages in the store.
        """
        if isinstance(key, list):
            if not isinstance(message, list):
                raise ValueError("message must be a list when key is a list")
            if len(key) != len(message):
                raise ValueError("key and message must have the same length")
            for u, m in zip(key, message):
                self[u] = m
        else:
            path = self.get_path(key)
            with path.open("w") as f:
                f.write(message.as_string())

