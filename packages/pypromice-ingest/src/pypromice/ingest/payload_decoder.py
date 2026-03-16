"""
Module for decoding payloads from a data logging system.

This module provides functions for decoding binary payloads into meaningful
data values based on specified binary formats. It includes specific decoders
for two- and four-byte encodings, as well as functionality for determining
payload formats and decoding entire payloads. A custom exception, `DecodeError`,
is implemented to handle errors that may occur during the decoding process.
Additionally, logging is used for debug information throughout the decoding
process.
"""
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

__all__ = [
    "DecodeError",
    "determine_payload_format",
    "decode"
]

logger = logging.getLogger(__name__)

CR_BASIC_EPOCH_OFFSET = datetime(1990, 1, 1, 0, 0, 0, 0).timestamp()


class DecodeError(Exception):
    """Decoding error exception object"""

    def __init__(
        self,
        format_index: int | None = None,
        format_name: str | None = None,
        letter_index: int | None = None,
        buffer_index: int | None = None,
        buffer_length: int | None = None,
        message: str | None = None,
    ):
        self.format_index = format_index
        self.format_name = format_name
        self.letter_index = letter_index
        self.buffer_index = buffer_index
        self.buffer_length = buffer_length
        self.message = message

    def as_dict(self) -> dict:
        return {
            "format_index": self.format_index,
            "format_name": self.format_name,
            "letter_index": self.letter_index,
            "buffer_index": self.buffer_index,
            "buffer_length": self.buffer_length,
            "message": self.message,
        }

    def __str__(self):
        return repr(self)

    def __repr__(self):
        variables_str = ", ".join(f"{k}={v}" for k, v in self.as_dict().items())

        return f"{self.__class__.__name__}({variables_str})"


def parse_gfp2(buffer: bytes) -> float:
    """Two-byte floating point decoder

    Parameters
    ----------
    buffer : bytes
        List of two values

    Returns
    -------
    float
        Decoded value
    """
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
    """Four-byte decoder

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


def determine_payload_format(payload: bytes, payload_formats_path: Path) -> str:
    """Determine payload format from lookup table, based on the first byte in
    the payload

    Parameters
    ----------
    payload : bytes
        Payload message
    payload_formats_path : Path
        File path to payload formats lookup table

    Returns
    -------
    bin_format : str
        Binary format of payload
    """
    payload_formats = pd.read_csv(payload_formats_path, index_col=0)
    payload_formats = payload_formats[payload_formats["flags"].values != "don’t use"]

    # The original code also had a special case for imei number 300234064121930. This should be handled differently.
    bidx = ord(payload[:1])

    logger.info(f"Format index: {bidx}")

    # The payload format is determined by the first byte of the payload
    if bidx not in payload_formats.index:
        raise DecodeError(format_index=bidx, message=f"Unknown payload format: {bidx}")

    payload_format = payload_formats.loc[bidx]
    _, bin_format, bin_name, flags, notes = payload_format.iloc[:5]

    logger.info(f"Format: {bin_name}, {notes}")
    logger.debug(f"Binary format: {bin_format}")

    return bin_format


def decode(bin_format: str, payload: bytes) -> list:
    """Decode a payload, based on a pre-defined format identifier

    Parameters
    ----------
    bin_format : str
        Binary payload format identifier
    payload : bytes
        Payload message

    Returns
    -------
    dataline : list
        Decoded payload
    """
    payload_length = len(payload)
    logger.info(f"Decoding payload with format: {bin_format!r}. Payload length: {payload_length}")
    logger.debug(f"Payload: {payload!r}")
    # Note: bin_val is just len(bin_format)
    indx = 1  # The first byte is the payload format
    dataline: list = []
    # Iterate over each token in the bin_format and decode the payload
    try:
        for format_letter_index, type_letter in enumerate(bin_format):
            logger.debug(
                f"Index {indx:02n} / {payload_length} Type letter: {type_letter:s}. upcuming bytes: {payload[indx:indx + 6]}..."
            )

            if type_letter == "f":
                # Encoded as 2 bytes base-10 floating point (GFP2)
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

            elif type_letter == "l":
                # Encoded as 4 bytes two complement integer (GLI4) - note the mac nan value is not a max value

                value = parse_gli4(payload[indx:])

                # The value 2147450879 seems arbitrary, but 2147483648 is the maximum value for a 4-byte signed integer
                nan_values = -2147483648, 2147450879
                if value in nan_values:
                    dataline.append(np.nan)
                else:
                    # TODO: Integers doesn't support NaN values and this will cause mixed types which should be avoided
                    dataline.append(float(value))
                indx += 4

            elif type_letter == "t":
                # timestamp as seconds since 1990-01-01 00:00:00 +0000 encoded as GLI4
                value = parse_gli4(payload[indx:])
                time = datetime.fromtimestamp(CR_BASIC_EPOCH_OFFSET + value)
                dataline.append(time)
                indx += 4

            elif type_letter in ("g", "n", "e"):
                # GPS time or coordinate encoding
                nan_values_fp2 = (8191,)
                # Check if byte is a 2-bit NAN. This occurs when the GPS data is not
                # available and the logger sends a 2-bytes NAN instead of a 4-bytes value
                if np.isnan(parse_gfp2(payload[indx:])) in nan_values_fp2:
                    # This is a 2-byte NAN
                    # The GPS data is not available
                    # We need to skip the next 2 bytes
                    dataline.append(np.nan)
                    indx += 2

                else:
                    # This is a 4-byte value
                    value = parse_gli4(payload[indx:])

                    if type_letter == "g":
                        value /= 100.0
                    else:  # type_letter in ('n', 'e'):
                        value /= 100000.0

                    dataline.append(value)
                    indx += 4
            else:
                raise Exception(f"Unknown type letter: {type_letter}")
            logger.debug(f"New value: {dataline[-1]}")

        # TODO: 2 bytes Floating points has overflow value 7999. Maybe also an underflow
        logger.debug(f"Index: {indx:02n} / {payload_length}")

        # Note: The CRBasic logger program us creating a checksum.
        # This is solely for the sending to the modem and it is not transmitted

        # TODO: Extract flags from suffix values
        # TODO: Investigate the protocol for the suffix values
        if indx != len(payload):
            raise Exception(f"Payload length mismatch: {indx} != {len(payload)}")

    except Exception as e:
        raise e
        # raise DecodeError(bidx, bin_name, format_letter_index, indx) from e
    return dataline


if __name__ == "__main__":
    import argparse
    import sys
    import pandas as pd
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Payload decoder tool for CRBasic logger")
    parser.add_argument(
        "--payload_file_path",
        "-p",
        type=Path,
        help="Path to payload file",
        default=None,
    )
    parser.add_argument(
        "--format", "-f", help="Explicitly specify decoding string", default=None
    )
    parser.add_argument("--no-log", action="store_true", help="Disable logging", default=False)
    parser.add_argument(
        "--log_level",
        "-l",
        type=str,
        help="Set logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument(
        "--payload_formats_path",
        type=Path,
        help="Path to payload formats .csv file",
    )
    parser.add_argument(
        "--drop_checksum_suffix",
        action="store_true",
        help="Remove the last two bytes from the payload",
        default=False,
    )
    args = parser.parse_args()
    if args.no_log:
        logging.disable(logging.CRITICAL)
    else:
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stderr
        )
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, args.log_level))

    if isinstance(args.payload_file_path, Path):
        with open(args.payload_file_path, "rb") as payload_file:
            payload = payload_file.read()
    else:
        # Read payload from stdin
        payload = sys.stdin.buffer.read()

    if args.drop_checksum_suffix:
        payload = payload[:-2]

    if args.format is not None:
        payload_format = args.format
    else:
        if args.payload_formats_path is None:
            raise ValueError("Payload format path must be specified if decoding format is not specified")
        payload_format = determine_payload_format(payload, args.payload_formats_path)
    decoded = decode(payload_format, payload)

    df = pd.DataFrame([decoded]).to_csv(sys.stdout, index=False, header=False)
