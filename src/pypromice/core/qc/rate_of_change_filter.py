"""Rate-of-change (ROC) quality control for PROMICE AWS time series.

This module detects and flags physically implausible short-term variability
in automatic weather station (AWS) observations. The algorithm computes
forward and backward rates of change for selected variables, derives
adaptive thresholds from rolling percentiles, and identifies outliers that
exceed expected natural variability.

Key features
------------
- Variable-specific thresholds (tolerance `tol` and multiplicative factor
  `factor`) defined via regex matching.
- Rolling 95th-percentile ROC thresholds within configurable time windows.
- Detection using both forward and backward differences.
- Additional logic to handle data gaps and irregular sampling.
- Rescue step that unflags points consistent with linear interpolation
  within a user-defined tolerance `tol`.
- Two-pass filtering to improve robustness.
- Integration with QC flagging and optional data removal.

Parameter meaning
-----------------
- `factor`: Multiplies the rolling 95th-percentile ROC to define the
  detection threshold (higher values → less aggressive filtering).
- `tol`: Absolute tolerance used when testing whether a flagged point can
  be reconstructed by linear interpolation (higher values → more points
  rescued).

Typical workflow
----------------
    ds = rate_of_change_filter(ds)

Variables matching patterns in DEFAULT_VARIABLE_THRESHOLDS are processed.
Flagged samples receive the "ROC" QC flag and are removed from the data.

Notes
-----
- Assumes a monotonic time coordinate named "time".
- Designed for PROMICE Level-1/Level-2 processing.
- Avoids slow xarray forward/backward fill by using NumPy-based routines
  for performance on long time series.

Author: Baptiste Vandecrux
"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
import re

logger = logging.getLogger(__name__)

NO_QC_VAR = ['time','rec']

# Default variable thresholds, where factor is the multiplicative factor applied
DEFAULT_VARIABLE_THRESHOLDS = {
    r"^t_[iul]$": {"tol": 3, "factor": 2.2},
    r"^p_[iul]$": {"tol": 1.0, "factor": 2.0},
    r"^rh_[iul]$": {"tol": 2.0, "factor": 3.5},
    r"^t_i_(?:10|[1-9])$": {"tol": 0.5, "factor": 1.5},
}

# Default rate evaluator configs
DEFAULT_WINDOW = "7D"
DEFAULT_REF_FREQ = "h"
DEFAULT_MIN_PERIODS = 10


def rate_of_change_filter(ds):
    """Apply the rate-of-change outlier filter to all matching variables in a dataset.

    Selects variables in `ds.data_vars` whose names match any regex pattern in
    `DEFAULT_VARIABLE_THRESHOLDS`, then runs `flag_high_rate_of_change` to
    identify outliers. The filter is applied in up to two passes: after the
    first pass, flagged samples are temporarily set to NaN and the filter is
    rerun to catch additional outliers. Final flags are the logical OR of both
    passes.

    Parameters
    ----------
    ds: xr.Dataset
        Input dataset containing time series variables.

    Returns
    -------
    xr.Dataset
        Dataset (same object) with the rate-of-change filter applied.
    """

    patterns = [re.compile(p) for p in DEFAULT_VARIABLE_THRESHOLDS]

    vars_with_thresholds = [
        v for v in ds.data_vars
        if any(p.match(v) for p in patterns)
    ]

    for var in vars_with_thresholds:
        tol, factor = _get_params(var)
        flag_final, _, _, _,  = flag_high_rate_of_change(ds,
                                                         var,
                                                         DEFAULT_WINDOW,
                                                         DEFAULT_REF_FREQ,
                                                         DEFAULT_MIN_PERIODS,
                                                         tol,
                                                         factor)

        tmp = ds.copy(deep=True)
        tmp[var].loc[{"time": flag_final.time[flag_final]}] = np.nan  # apply first pass to temporary object

        if flag_final.any():
            flag2, _, _, _ = flag_high_rate_of_change(tmp,
                                                      var,
                                                      DEFAULT_WINDOW,
                                                      DEFAULT_REF_FREQ,
                                                      DEFAULT_MIN_PERIODS,
                                                      tol,
                                                      factor)

            flag2 = flag2.reindex_like(flag_final, fill_value=False)
            flag_final = (flag_final | flag2)

        flag_final = flag_final.reindex_like(ds.time, fill_value=False)
        logger.debug(
            f"ROC filter on {var} (tol={tol}, factor={factor}): filtering {flag_final.sum().item()}/{len(ds.time)}")

        if flag_final.any():
            ds[var] = ds[var].where(~flag_final)

    return ds


def flag_high_rate_of_change(da: xr.DataArray,
                             window: str,
                             ref_freq: str,
                             min_periods: int,
                             factor: float,
                             tol: float
) -> tuple[xr.DataArray,xr.DataArray,xr.DataArray,xr.DataArray]:
    """Flag anomalously high rates of change and refine using interpolation logic.
    Detects time steps where the rate of change exceeds a rolling percentile-
    based threshold (forward and backward differences), applies additional
    logical rules related to missing neighbors and uneven sampling, and
    finally removes flags consistent with linear interpolation.

    Parameters
    ----------
    da: xr.DataArray
        Input data array of variable to be analysed.
    window: str
        Rolling window length (pandas offset string). For example, "7D".
    ref_freq: str
        Time unit used to normalize rates (e.g. "h", "D").
    min_periods: int
        Minimum samples required in the rolling window. For example, 10.
    factor: float
        Multiplier applied to the rolling 95th percentile threshold.
    tol: float
        Tolerance for linear interpolation unflagging.

    Returns
    -------
    flag_final: xr.DataArray
        Final flag after interpolation-based unflagging.
    fwd_full: xr.DataArray
        Forward rate-of-change flags on the full time axis.
    bwd_full: xr.DataArray
        Backward rate-of-change flags on the full time axis.
    flag_combined: xr.DataArray
        Combined logical flag before interpolation refinement.
    """
    da = da.dropna("time")
    fwd_full, bwd_full = rate_of_change_fwd_bwd_and_thresholds(da,
                                                               window,
                                                               ref_freq,
                                                               min_periods,
                                                               factor,
                                                               )
    y = da
    prev_missing = y.shift({"time": 1}).isnull()
    next_missing = y.shift({"time": -1}).isnull()

    tt = da["time"]
    dt_prev = tt - tt.shift({"time": 1})
    dt_next = tt.shift({"time": -1}) - tt
    uneven_dt = dt_prev != dt_next

    if da.sizes["time"] > 0:
        prev_missing.values[0] = True
        next_missing.values[-1] = True
        uneven_dt.values[0] = True
        uneven_dt.values[-1] = True

    # Combine multiple logical criteria
    flag_combined = (
        (fwd_full & bwd_full)
        | (fwd_full & prev_missing)
        | (fwd_full & next_missing)
        | (bwd_full & prev_missing)
        | (bwd_full & next_missing)
        | (fwd_full & uneven_dt)
        | (bwd_full & uneven_dt)
    )

    # Final refinement step
    if bool(flag_combined.any()):
        flag_final = unflag_if_linear_interp(da,
                                             flag_combined,
                                             tol)
    else:
        flag_final = flag_combined

    return flag_final, fwd_full, bwd_full, flag_combined


def rate_of_change_fwd_bwd_and_thresholds(da: xr.DataArray,
                                          window: str,
                                          ref_freq: str,
                                          min_periods: int,
                                          factor: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute forward/backward rate-of-change flags and rolling thresholds.
    Calculates absolute rates of change between consecutive samples of a
    time series, derives rolling 95th percentile thresholds, and returns
    forward- and backward-assigned exceedance flags aligned to the full
    time axis of the input DataArray.

    Parameters
    ----------
    da: xr.DataArray
        Input variable time series with NaNs already removed.
    window: str
        Rolling window length as a pandas offset string (e.g. "7D")
    ref_freq: str
        Time unit used to normalize rates (e.g. "h", "D").
    min_periods: int
        Minimum number of samples required in the rolling window
        (for example, 10).
    factor: float
        Multiplier applied to the rolling 95th percentile to form
        detection thresholds.

    Returns
    -------
    fwd_full: xr.DataArray
        Boolean DataArray of forward rate exceedances aligned
          to the full time axis.
    bwd_full: xr.DataArray
        Boolean DataArray of backward rate exceedances aligned
          to the full time axis.
    """
    t = da["time"].values
    v = da.values

    dt_ns = (t[1:] - t[:-1])
    dv = v[1:] - v[:-1]
    denom_ns = np.timedelta64(1, ref_freq)
    rate = np.abs(dv) / (dt_ns / denom_ns)

    idx_fwd = pd.to_datetime(t[1:])
    s_fwd = pd.Series(rate, index=idx_fwd)
    thr_fwd = factor * s_fwd.rolling(window=window,
                                     center=True,
                                     min_periods=min_periods).quantile(0.95)
    flag_fwd_s = s_fwd > thr_fwd

    idx_bwd = pd.to_datetime(t[:-1])
    s_bwd = pd.Series(rate, index=idx_bwd)
    thr_bwd = factor * s_bwd.rolling(window=window,
                                     center=True,
                                     min_periods=min_periods).quantile(0.95)
    flag_bwd_s = s_bwd > thr_bwd

    flag_fwd = xr.DataArray(
        flag_fwd_s.values.astype(bool),
        coords={"time": da["time"].values[1:]},
        dims=("time",),
    )
    flag_bwd = xr.DataArray(
        flag_bwd_s.values.astype(bool),
        coords={"time": da["time"].values[:-1]},
        dims=("time",),
    )

    tfull = da["time"].values
    fwd_full = xr.DataArray(np.zeros(tfull.shape, bool), coords={"time": tfull}, dims=("time",))
    bwd_full = xr.DataArray(np.zeros(tfull.shape, bool), coords={"time": tfull}, dims=("time",))
    fwd_full.loc[{"time": flag_fwd["time"]}] = flag_fwd
    bwd_full.loc[{"time": flag_bwd["time"]}] = flag_bwd

    return fwd_full, bwd_full


def unflag_if_linear_interp(da: xr.DataArray,
                            flag: xr.DataArray,
                            tol: float
) -> xr.DataArray:
    """Remove flags where values follow linear interpolation within tolerance.

    Parameters
    ----------
    da : xr.DataArray
        Dataset containing the variable.
    flag : xr.DataArray (bool)
        Initial flag array on the same time axis as the evaluated data.
    tol : float
        Absolute tolerance for deviation from linear interpolation.

    Returns
    -------
    xr.DataArray
        Updated flag array where linearly interpolated points are unflagged.
    """
    # Align variable to flag time axis
    v = da.sel({"time": flag["time"]}).astype("float64")
    t = v["time"].values.astype("datetime64[ns]").astype("int64")

    n = v.sizes["time"]
    idx = np.arange(n, dtype=np.int64)

    # Valid (non-flagged and finite) samples
    okv = ((~flag.values) & np.isfinite(v.values))
    base = np.where(okv, idx.astype("float64"), np.nan)
    pi = _ffill_idx(base)
    ni = _bfill_idx(base)

    # Finding points that have two valid neigbours
    has_both = np.isfinite(pi) & np.isfinite(ni) & (pi != ni)

    pi_i = np.zeros(n, dtype=np.int64)
    ni_i = np.zeros(n, dtype=np.int64)
    pi_i[has_both] = pi[has_both].astype(np.int64)
    ni_i[has_both] = ni[has_both].astype(np.int64)

    t_prev = np.zeros(n, dtype=np.int64)
    t_next = np.zeros(n, dtype=np.int64)
    v_prev = np.zeros(n, dtype=float)
    v_next = np.zeros(n, dtype=float)

    t_prev[has_both] = t[pi_i[has_both]]
    t_next[has_both] = t[ni_i[has_both]]
    v_prev[has_both] = v.values[pi_i[has_both]]
    v_next[has_both] = v.values[ni_i[has_both]]

    # Linear interpolation weights
    denom = (t_next - t_prev).astype("float64")
    w = np.full(n, np.nan, dtype="float64")
    w[has_both] = (t[has_both] - t_prev[has_both]) / denom[has_both]

    # Interpolated estimate
    v_hat = v_prev + w * (v_next - v_prev)

    # Unflag if close to linear interpolation
    unflag = (
        flag.values
        & has_both
        & np.isfinite(v_hat)
        & (np.abs(v.values - v_hat) <= tol)
    )

    return xr.DataArray(flag.values & ~unflag,
                        coords=flag.coords,
                        dims=flag.dims)


def _get_params(var, tol=None, factor=None):
    if tol is not None and factor is not None:
        logger.debug(f"{var}: using user tol={tol}, factor={factor}")
        return float(tol), float(factor)

    for pat, cfg in DEFAULT_VARIABLE_THRESHOLDS.items():
        if re.match(pat, var):
            vt = cfg["tol"] if tol is None else tol
            vf = cfg["factor"] if factor is None else factor
            return vt, vf

    vt = 0.1 if tol is None else float(tol)
    vf = 2.0 if factor is None else float(factor)
    return vt, vf


# avoid xarray ffill/bfill (slow) -> numpy forward/backward fill
def _ffill_idx(a):
    a = a.copy()
    m = np.isnan(a)
    idx = np.where(~m, np.arange(a.size), 0)
    np.maximum.accumulate(idx, out=idx)
    a[m] = a[idx[m]]
    return a


def _bfill_idx(a):
    a = a.copy()
    m = np.isnan(a)
    idx = np.where(~m, np.arange(a.size), a.size-1)
    idx = idx[::-1]
    np.minimum.accumulate(idx, out=idx)
    idx = idx[::-1]
    a[m] = a[idx[m]]
    return a
