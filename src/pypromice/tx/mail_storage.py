import email.parser
from mailbox import Message
from pathlib import Path
from typing import MutableMapping, List

import attr


@attr.define
class MailLake(MutableMapping[int, Message]):
    def __delitem__(self, key, /):
        """
        Remove a message from the lake.
        """
        path = self.get_path(key)
        if not path.exists():
            raise KeyError(f"Message with UUID {key} not found in lake.")
        path.unlink()

    def __len__(self):
        """
        Get the number of messages in the lake.
        """
        return len(list(self.root_path.glob("*.eml")))

    def __iter__(self):
        """
        Iterate over the UUIDs of messages in the lake.
        """
        for path in self.root_path.glob("*.eml"):
            yield int(path.stem)

    root_path: Path = attr.field()

    def get_path(self, uuid: str) -> Path:
        """
        Get the path to the message file.
        """
        return self.root_path / f"{uuid}.eml"

    def set(self, uuid: str, message: Message):
        """
        Save a message to the lake.
        """
        path = self.get_path(uuid)
        with path.open("w") as f:
            f.write(message.as_string())

    def get(self, uuid: str) -> Message:
        """
        Get a message from the lake.
        """
        path = self.get_path(uuid)
        if not path.exists():
            raise FileNotFoundError(f"Message with UUID {uuid} not found in lake.")
        with path.open() as f:
            return email.parser.Parser().parse(f)

    def __contains__(self, uuid: str) -> bool:
        """
        Check if the message file exists in the lake.
        """
        return self.get_path(uuid).exists()


    def __getitem__(self, uuid: str|List[str]) -> Message | List[Message]:
        """
        Get a message or a list of messages from the lake.
        """
        if isinstance(uuid, list):
            return [self.get(u) for u in uuid]
        return self.get(uuid)

    def __setitem__(self, uuid: str|List[str], message: Message|List[Message]):
        """

        Parameters
        ----------
        uuid
        message

        Returns
        -------

        """
        if isinstance(uuid, list):
            if not isinstance(message, list):
                raise ValueError("message must be a list when uuid is a list")
            if len(uuid) != len(message):
                raise ValueError("uuid and message must have the same length")
            for u, m in zip(uuid, message):
                self.set(u, m)
        else:
            self.set(uuid, message)
