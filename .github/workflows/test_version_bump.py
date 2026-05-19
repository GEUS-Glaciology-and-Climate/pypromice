import re
import subprocess
import tomllib
from pathlib import Path


def get_pyproject_version(text):
    return tomllib.loads(text)["project"]["version"]


def get_conf_version(text):
    return re.search(
        r'^version\s*=\s*[\'"]([^\'"]+)[\'"]',
        text,
        re.M,
    ).group(1)


def read_main_file(path):
    return subprocess.check_output(
        ["git", "show", f"origin/main:{path}"],
        text=True,
    )


main_pyproject = get_pyproject_version(
    read_main_file("pyproject.toml")
)
current_pyproject = get_pyproject_version(
    Path("pyproject.toml").read_text()
)

main_conf = get_conf_version(
    read_main_file("docs/conf.py")
)
current_conf = get_conf_version(
    Path("docs/conf.py").read_text()
)

assert current_pyproject != main_pyproject, "Version not bumped in pyproject.toml"
assert current_conf != main_conf, "Version not bumped in docs/conf.py"
assert current_pyproject == current_conf, "Version mismatch between files"