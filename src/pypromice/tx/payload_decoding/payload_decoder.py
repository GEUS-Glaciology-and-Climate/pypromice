from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "decode_payload",
    "DecodeError",
]

CR_BASIC_EPOCH_OFFSET = datetime(1990, 1, 1, 0, 0, 0, 0).timestamp()

# Load payload formats from CSV file
payload_formats_path = Path(__file__).parent.joinpath("payload_formats.csv")
payload_formats = pd.read_csv(payload_formats_path, index_col=0)
payload_formats = payload_formats[payload_formats["flags"].values != "don’t use"]


class DecodeError(Exception):

    def __init__(
        self,
        format_index: int | None = None,
        format_name: str | None = None,
        letter_index: int | None = None,
        buffer_index: int | None = None,
        message: str | None = None,
    ):
        self.format_index = format_index
        self.format_name = format_name
        self.letter_index = letter_index
        self.buffer_index = buffer_index
        self.message = message

    def as_dict(self) -> dict:
        return {
            "format_index": self.format_index,
            "format_name": self.format_name,
            "letter_index": self.letter_index,
            "buffer_index": self.buffer_index,
            "message": self.message,
        }

    def __str__(self):
        return repr(self)

    def __repr__(self):
        variables_str = ", ".join(f"{k}={v}" for k, v in self.as_dict().items())

        return f"{self.__class__.__name__}({variables_str})"


def parse_gfp2(buffer: bytes) -> float:
    """Two-bit floating point decoder"""
    if len(buffer) < 2:
        raise ValueError("Buffer too short for gfp2 decoding")

    msb = buffer[0]
    lsb = buffer[1]
    Csign = -2 * (msb & 128) / 128 + 1
    CexpM = (msb & 64) / 64
    CexpL = (msb & 32) / 32
    Cexp = 2 * CexpM + CexpL - 3
    Cuppmant = (
        4096 * (msb & 16) / 16
        + 2048 * (msb & 8) / 8
        + 1024 * (msb & 4) / 4
        + 512 * (msb & 2) / 2
        + 256 * (msb & 1)
    )
    Cnum = Csign * (Cuppmant + lsb) * 10**Cexp
    return round(Cnum, 3)


def parse_gli4(buffer: bytes) -> int:
    """Four-bit decoder

    Parameters
    ----------
    buffer : bytes
        List of four values

    Returns
    -------
    float
        Decoded value
    """
    if len(buffer) < 4:
        raise ValueError("Buffer too short for gli4 decoding")

    return int.from_bytes(buffer[:4], byteorder="big", signed=True)


def decode_payload(payload: bytes) -> list:

    # The original code also had a special case for imei number 300234064121930. This should be handled differently.
    bidx = ord(payload[:1])

    # The payload format is determined by the first byte of the payload
    if bidx not in payload_formats.index:
        raise DecodeError(format_index=bidx, message=f"Unknown payload format: {bidx}")

    payload_format = payload_formats.loc[bidx]
    _, bin_format, bin_name = payload_format.iloc[:3]
    # Note: bin_val is just len(bin_format)
    indx = 1  # The first byte is the payload format
    dataline: list = []

    # Iterate over each token in the bin_format and decode the payload
    format_letter_index = None
    try:
        for format_letter_index, type_letter in enumerate(bin_format):

            if type_letter == "f":
                nan_values = (8191,)
                inf_values = (8190,)
                neg_inf_values = -8190, -8191

                value = parse_gfp2(payload[indx:])
                # Be careful with float and int comparison. Float doesn't guarantee exact values.
                # Consider using binary patterns for the special values instead.
                if value in nan_values:
                    dataline.append(np.nan)
                elif value in inf_values:
                    dataline.append(np.inf)
                elif value in neg_inf_values:
                    dataline.append(-np.inf)
                else:
                    dataline.append(value)
                indx += 2
                continue

            if type_letter == "l":
                # Encoded as a 4 byte two complement integer

                value = parse_gli4(payload[indx:])

                # The value 2147450879 seems arbitrary, but 2147483648 is the maximum value for a 4-byte signed integer
                nan_values = -2147483648, 2147450879
                if value in nan_values:
                    dataline.append(np.nan)
                else:
                    # TODO: Integers doesn't support NaN values and this will cause mixed types which should be avoided
                    dataline.append(float(value))
                indx += 4
                continue

            if type_letter == "t":
                value = parse_gli4(payload[indx:])
                time = datetime.fromtimestamp(CR_BASIC_EPOCH_OFFSET + value)
                dataline.append(time)
                indx += 4
                continue

            # GPS time or coordinate encoding
            if type_letter in ("g", "n", "e"):
                nan_values_fp2 = (8191,)
                # Check if byte is a 2-bit NAN. This occurs when the GPS data is not
                # available and the logger sends a 2-bytes NAN instead of a 4-bytes value
                if np.isnan(parse_gfp2(payload[indx:])) in nan_values_fp2:
                    # This is a 2-byte NAN
                    # The GPS data is not available
                    # We need to skip the next 2 bytes
                    dataline.append(np.nan)
                    indx += 2
                    continue
                else:
                    # This is a 4-byte value
                    value = parse_gli4(payload[indx:])

                    if type_letter == "g":
                        value /= 100.0
                    else:  # type_letter in ('n', 'e'):
                        value /= 100000.0

                    dataline.append(value)
                    indx += 4
                continue

            raise Exception(f"Unknown type letter: {type_letter}")


        # TODO: Compute and validate checksum
        if indx != len(payload):
            raise Exception(f"Payload length mismatch: {indx} != {len(payload)}")

    except Exception as e:
        raise DecodeError(bidx, bin_name, format_letter_index, indx) from e

    return dataline
