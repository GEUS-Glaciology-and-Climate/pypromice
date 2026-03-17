import pytest

pytest.importorskip(
    "eccodes",
    reason="eccodes not installed. BUFR export tests skipped. "
           "Install with `pip install pypromice[eccodes]`."
)