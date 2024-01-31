from functools import reduce
from pathlib import Path
from typing import MutableMapping, Mapping, Sequence

import toml


def update_recursive(src: MutableMapping, new: Mapping) -> MutableMapping:
    for k, v in new.items():
        if k not in src:
            src[k] = v
        elif isinstance(v, Mapping):
            update_recursive(src[k], v)
        else:
            src[k] = v
    return src


def load_toml_files(*paths: Path) -> Mapping:
    """
    Read multiple config files and merge recursively. Prioritize values from the latest file
    """
    return reduce(update_recursive, map(toml.load, paths), dict())
