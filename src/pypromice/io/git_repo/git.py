import subprocess
import os
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


def get_commit_hash_and_check_dirty(file_path: str | Path) -> str:
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if file_path.is_dir():
        repo_path = file_path
    else:
        repo_path = file_path.parent

    try:
        # Ensure the file path is relative to the repository

        # Get the latest commit hash for the file
        commit_hash = (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    repo_path,
                    "log",
                    "-n",
                    "1",
                    "--pretty=format:%H",
                ],
                stderr=subprocess.STDOUT,
            )
            .strip()
            .decode("utf-8")
        )

        # Check if the file is dirty (has uncommitted changes)
        diff_output = (
            subprocess.check_output(
                ["git", "-C", repo_path, "diff"],
                stderr=subprocess.STDOUT,
            )
            .strip()
            .decode("utf-8")
        )

        # If diff_output is not empty, the file has uncommitted changes
        is_dirty = len(diff_output) > 0

        if is_dirty:
            logger.warning(f"Warning: The file {file_path} is dirty compared to the last commit. {commit_hash}")
            return f'{commit_hash} (dirty)'
        if commit_hash == "":
            logger.warning(f"Warning: The file {file_path} is not under version control.")
            return 'unknown'

        return commit_hash
    except subprocess.CalledProcessError as e:
        logger.warning(f"Error: {e.output.decode('utf-8')}")
        return 'unknown'
