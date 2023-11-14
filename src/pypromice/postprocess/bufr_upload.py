import logging
from ftplib import FTP
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger()


class DMIConfig(TypedDict):
    host: str
    user: str
    passwd: str


def concat_bufr_files(bufr_dir: Path, output_path: Path):
    with output_path.open("wb") as fp_out:
        for path in sorted(bufr_dir.glob("*.bufr")):
            with path.open("rb") as fp_in:
                fp_out.write(fp_in.read())


def upload_bufr(path: Path, dmi_config: DMIConfig):
    with FTP(host=dmi_config["host"]) as ftp:
        ftp.login(user=dmi_config["user"], passwd=dmi_config["passwd"])
        ftp.cwd("upload")
        with path.open("b") as fp:
            ftp.storbinary(f"STOR {path.name}", fp=fp)
