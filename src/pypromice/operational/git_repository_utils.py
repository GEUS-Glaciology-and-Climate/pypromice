import datetime
import logging
import subprocess
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class DataRepositoryError(Exception):
    def __init__(self, stdout: bytes, stderr: bytes):
        stderr_str = stderr.decode("utf8")
        stdout_str = stdout.decode("utf8")
        super().__init__(f"{stdout_str}\n{stderr_str}")


def prepare_git_repository(path: Path, branch: str):
    git_commands = [
        [
            "checkout",
            branch,
        ],
        ["pull",],
        [
            "reset",
            "--hard",
            "HEAD",
        ],
        [
            "clean",
            "-fd",
        ],
    ]
    for git_command in git_commands:
        execute_git_command(path, git_command)


def execute_git_command(repository_path: Path, git_command: List[str]):
    args = ["git", "-C", repository_path.as_posix()] + git_command
    logger.info(" ".join(args))
    cp = subprocess.run(
        args=args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if cp.returncode > 0:
        raise DataRepositoryError(cp.stdout, cp.stderr)


def get_commit_date(repository_path: Path, file_path: Path):
    completed_process = subprocess.run(
        ["git", "-C", repository_path, "log", "-n 1", "--format=%aI", "--", file_path],
        stdout=subprocess.PIPE,
    )
    datetime_string = completed_process.stdout.decode("utf-8").strip()
    return datetime.datetime.fromisoformat(datetime_string)
