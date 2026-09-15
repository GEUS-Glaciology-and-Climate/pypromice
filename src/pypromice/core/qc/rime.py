import logging

import xarray as xr

from pypromice.core.qc.common import flag_qc, clean_view

logger = logging.getLogger(__name__)

# Stefan-Boltzmann constant (W m-2 K-4)
SIGMA = 5.670374419e-8

DEFAULT_THRESHOLD = 3


def detect_rime(ds: xr.Dataset, threshold: float = DEFAULT_THRESHOLD) -> xr.Dataset:
    """Detect radiometer rime/icing from closeness between measured and
    theoretical (black-body) longwave radiation.

    When a radiometer's dome ices over, both its downwelling and upwelling
    longwave readings collapse toward the dome's own black-body radiation
    (computed from the radiometer's internal temperature, "t_rad") instead
    of sensing the true incoming/outgoing longwave flux. A day is flagged
    as rime-affected when, for at least 70% of its samples, both "dlr" and
    "ulr" sit within `threshold` (W/m2) of that theoretical black-body
    value, and of each other. The daily flag is then smoothed (an
    inflate-deflate-deflate-inflate morphological pass, removing
    single-day blips and filling single-day gaps) before being broadcast
    back to the original time resolution.

    Args:
        ds (xr.Dataset): Dataset containing "t_rad", "dlr" and "ulr".
        threshold (float, optional): Maximum allowed deviation (W/m2)
            between measured and theoretical longwave radiation, and
            between "dlr" and "ulr", for a sample to look rime-affected.
            Defaults to 3.

    Returns:
        xr.Dataset: Dataset with "dlr" and "ulr" flagged "RIME" in
        "<var>_qc" over rime-affected days (data itself is unchanged; use
        finalize_qc to remove it).
    """
    if not all(v in ds for v in ("t_rad", "dlr", "ulr")):
        logger.debug("detect_rime: t_rad, dlr or ulr missing, skipping")
        return ds

    # Detect against a clean copy -- an already-flagged dlr/ulr sample
    # (e.g. persistence, an earlier manual flag) must not count toward
    # "looks rime-affected" just because it's still numerically present.
    ds_clean = clean_view(ds)

    lw_rad = SIGMA * (ds_clean["t_rad"] + 273.15) ** 4
    dlr_rad_diff = abs(ds_clean["dlr"] - lw_rad)
    ulr_rad_diff = abs(ds_clean["ulr"] - lw_rad)
    lr_diff_abs = abs(ds_clean["dlr"] - ds_clean["ulr"])

    looks_rimed = (
        (dlr_rad_diff < threshold) &
        (ulr_rad_diff < threshold) &
        (lr_diff_abs < threshold)
    )

    daily_flag = looks_rimed.resample(time="1D").mean() >= 0.7

    # Inflate -> deflate -> deflate -> inflate: drop single-day blips, then
    # fill single-day gaps, in the daily rime flag.
    daily_flag = daily_flag.rolling(time=3, center=True, min_periods=1).max()
    daily_flag = daily_flag.rolling(time=3, center=True, min_periods=1).min()
    daily_flag = daily_flag.rolling(time=3, center=True, min_periods=1).min()
    daily_flag = daily_flag.rolling(time=3, center=True, min_periods=1).max()

    # Broadcast back to the original time resolution
    rime_flag = daily_flag.reindex(time=ds["time"], method="ffill").astype(bool)

    n_flagged = int(rime_flag.sum())
    logger.debug(f"detect_rime: flagging {n_flagged}/{len(ds.time)} samples as RIME")

    if n_flagged:
        ds = flag_qc(ds, "dlr", "RIME", mask=rime_flag)
        ds = flag_qc(ds, "ulr", "RIME", mask=rime_flag)
        ds = flag_qc(ds, "dsr", "RIME", mask=rime_flag)
        ds = flag_qc(ds, "usr", "RIME", mask=rime_flag)

    return ds
