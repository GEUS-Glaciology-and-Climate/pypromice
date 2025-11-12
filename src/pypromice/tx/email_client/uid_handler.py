"""
UID handling functions, for reading and writing mail UID identifier to file.
"""
import os
import logging
logger = logging.getLogger(__name__)

def read_uid_from_file(uid_file: str) -> int:
    if not os.path.exists(uid_file):
        raise RuntimeError(f"UID file {uid_file} not found.")
    with open(uid_file, 'r') as f:
        uid = f.readline().strip()
        if not uid.isdigit():
            raise RuntimeError(f"Invalid UID in file {uid_file}. Must be numeric.")
        return int(uid)


def write_uid_to_file(uid: int, uid_file: str):
    try:
        with open(uid_file, 'w') as f:
            f.write(str(uid))
        logger.info(f"Updated UID file with latest historyId {uid}")
    except Exception as e:
        logger.error(f"Could not write UID to file {uid_file}: {e}")