from datetime import datetime

import attrs


@attrs.define
class IridiumMessage:
    """
    Class to represent an Iridium message.
    """

    imei: str
    momsn: int
    mtmsn: int
    time_of_session: datetime
    payload_filename: str
    payload_bytes: bytes | None

    # This is the original subject line of the email
    # NOTE:
    # This is not a part of the Iridium message, but rather a part of the email a layer above
    # It is stored here for payload decoding purposes
    subject: str | None
