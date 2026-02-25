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

# factor is the multiplicative factor applied
DEFAULT_VARIABLE_THRESHOLDS = {
    r"^t_[iul]$": {"tol": 3, "factor": 2.2},
    r"^p_[iul]$": {"tol": 1.0, "factor": 2.0},
    r"^rh_[iul]$": {"tol": 2.0, "factor": 3.5},
    r"^t_i_(?:10|[1-9])$": {"tol": 0.5, "factor": 1.5},
}

def set_and_remove_flagged_data(ds: xr.Dataset,
             v: str,
             flag: str,
             index_slice=None,
             mask=None
) -> xr.Dataset:
    if v in NO_QC_VAR: return ds

    if index_slice is None:
        index_slice = {"time": slice(None, None)}

    qc_flags = xr.DataArray(
                            np.full(ds[v].shape, "OK", dtype=object),
                            coords=ds[v].coords,
                            dims=ds[v].dims,
                            )

    q = qc_flags.loc[index_slice]
    x = ds[v].loc[index_slice]
    if q.size == 0:
        return ds

    m = xr.ones_like(x, dtype=bool) if mask is None else (
        mask.loc[index_slice] if isinstance(mask, xr.DataArray) else mask
    )

    cond = m & x.notnull() & (q == "OK")

    qc_flags.loc[index_slice] = xr.where(cond, str(flag), q)
    bad = qc_flags.astype(str) != "OK"
    ds[v] = ds[v].where(~bad)
    return ds

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

def unflag_if_linear_interp(ds, var, flag, tol=0.1):
    """
    Remove flags where values follow linear interpolation within tolerance.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable.
    var : str
        Variable name in ds to evaluate.
    flag : xarray.DataArray (bool)
        Initial flag array on the same time axis as the evaluated data.
    tol : float, optional
        Absolute tolerance for deviation from linear interpolation.

    Returns
    -------
    xarray.DataArray
        Updated flag array where linearly interpolated points are unflagged.
    """
    # Align variable to flag time axis
    v = ds[var].sel({"time": flag["time"]}).astype("float64")
    t = v["time"].values.astype("datetime64[ns]").astype("int64")

    n = v.sizes["time"]
    idx = np.arange(n, dtype=np.int64)

    # Valid (non-flagged and finite) samples
    okv = ((~flag.values) & np.isfinite(v.values))
    base = np.where(okv, idx.astype("float64"), np.nan)
    pi = _ffill_idx(base)
    ni = _bfill_idx(base)
    # finding points that has two valid neigbors
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
                        coords=flag.coords, dims=flag.dims,
                        name=f"{flag.name}_final")

def rate_of_change_fwd_bwd_and_thresholds(da, var, window="7D", ref_freq="h",
                              min_periods=10, factor=None, tol=None):
    """Compute forward/backward rate-of-change flags and rolling thresholds.

    Calculates absolute rates of change between consecutive samples of a
    time series, derives rolling 95th percentile thresholds, and returns
    forward- and backward-assigned exceedance flags aligned to the full
    time axis of the input DataArray.

    Args:
        da (xr.DataArray): Input variable time series with NaNs already removed.
        var (str): Name of the variable (used for logging and metadata).
        window (str, optional): Rolling window length as a pandas offset
            string (e.g. "7D"). Defaults to "7D".
        ref_freq (str, optional): Time unit used to normalize rates (e.g. "h", "D").
            Defaults to "h".
        min_periods (int, optional): Minimum number of samples required in the
            rolling window. Defaults to 10.
        factor (float, optional): Multiplier applied to the rolling 95th
            percentile to form detection thresholds. If None, a
            variable-specific default is used.
        tol (float, optional): Interpolation tolerance passed through for
            metadata consistency. If None, a variable-specific default is used.

    Returns:
        Tuple[xr.Dataset, xr.DataArray, xr.DataArray]:
            - roc_ds: Dataset containing raw rates and rolling thresholds.
            - fwd_full: Boolean DataArray of forward rate exceedances aligned
              to the full time axis.
            - bwd_full: Boolean DataArray of backward rate exceedances aligned
              to the full time axis.
    """
    if (tol is None) | (factor is None):
        tol, factor = _get_params(var, tol=tol, factor=factor)

    t = da["time"].values
    v = da.values

    dt_ns = (t[1:] - t[:-1])
    dv = v[1:] - v[:-1]
    denom_ns = np.timedelta64(1, ref_freq)
    rate = np.abs(dv) / (dt_ns / denom_ns)

    idx_fwd = pd.to_datetime(t[1:])
    s_fwd = pd.Series(rate, index=idx_fwd)
    thr_fwd = factor * s_fwd.rolling(window=window, center=True, min_periods=min_periods).quantile(0.95)
    flag_fwd_s = s_fwd > thr_fwd

    idx_bwd = pd.to_datetime(t[:-1])
    s_bwd = pd.Series(rate, index=idx_bwd)
    thr_bwd = factor * s_bwd.rolling(window=window, center=True, min_periods=min_periods).quantile(0.95)
    flag_bwd_s = s_bwd > thr_bwd

    flag_fwd = xr.DataArray(
        flag_fwd_s.values.astype(bool),
        coords={"time": da["time"].values[1:]},
        dims=("time",),
        name=f"{var}_high_var_flag_fwd",
    )
    flag_bwd = xr.DataArray(
        flag_bwd_s.values.astype(bool),
        coords={"time": da["time"].values[:-1]},
        dims=("time",),
        name=f"{var}_high_var_flag_bwd",
    )

    tfull = da["time"].values
    fwd_full = xr.DataArray(np.zeros(tfull.shape, bool), coords={"time": tfull}, dims=("time",))
    bwd_full = xr.DataArray(np.zeros(tfull.shape, bool), coords={"time": tfull}, dims=("time",))
    fwd_full.loc[{"time": flag_fwd["time"]}] = flag_fwd
    bwd_full.loc[{"time": flag_bwd["time"]}] = flag_bwd

    roc_ds = xr.Dataset(
        data_vars=dict(
            roc_rate=(("time_rate",), rate),
            roc_thr_fwd=(("time_fwd",), thr_fwd.values.astype("float64")),
            roc_thr_bwd=(("time_bwd",), thr_bwd.values.astype("float64")),
        ),
        coords=dict(
            time_rate=da["time"].values[1:],
            time_fwd=da["time"].values[1:],
            time_bwd=da["time"].values[:-1],
        ),
        attrs=dict(var=var, ref_freq=ref_freq, window=window, min_periods=min_periods, factor=factor, tol=tol),
    )

    return roc_ds, fwd_full, bwd_full


def flag_high_rate_of_change(ds, var, window="7D",
                         ref_freq="h", min_periods=10, factor=None, tol=None):
    """Flag anomalously high rates of change and refine using interpolation logic.

    Detects time steps where the rate of change exceeds a rolling percentile-
    based threshold (forward and backward differences), applies additional
    logical rules related to missing neighbors and uneven sampling, and
    finally removes flags consistent with linear interpolation.

    Args:
        ds (xr.Dataset): Dataset containing the variable.
        var (str): Name of the variable to analyze.
        window (str, optional): Rolling window length (pandas offset string).
            Defaults to "7D".
        ref_freq (str, optional): Time unit used to normalize rates (e.g. "h", "D").
            Defaults to "h".
        min_periods (int, optional): Minimum samples required in the rolling
            window. Defaults to 10.
        factor (float, optional): Multiplier applied to the rolling 95th
            percentile threshold. If None, a variable-specific default is used.
        tol (float, optional): Tolerance for linear interpolation unflagging.
            If None, a variable-specific default is used.

    Returns:
        Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
            - fwd_full: Forward rate-of-change flags on the full time axis.
            - bwd_full: Backward rate-of-change flags on the full time axis.
            - flag_combined: Combined logical flag before interpolation refinement.
            - flag_final: Final flag after interpolation-based unflagging.
    """
    tol, factor = _get_params(var, tol=tol, factor=factor)

    da = ds[var].dropna("time")

    roc_ds, fwd_full, bwd_full = rate_of_change_fwd_bwd_and_thresholds(
        da, var, window=window, ref_freq=ref_freq, min_periods=min_periods, factor=factor, tol=tol
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
    ).rename(f"{var}_high_var_flag_combined")

    # Final refinement step
    if flag_combined.any():
        flag_final = unflag_if_linear_interp(ds, var, flag_combined, tol=tol)
    else:
        flag_final = flag_combined
    logger.info(f"ROC filter on {var} (tol={tol}, factor={factor}): filtering {flag_final.sum().item()}/{len(ds.time)}")

    return fwd_full, bwd_full, flag_combined, flag_final

def rate_of_change_filter(ds):
    """Apply the rate-of-change outlier filter to all matching variables in a dataset.

    Selects variables in `ds.data_vars` whose names match any regex pattern in
    `DEFAULT_VARIABLE_THRESHOLDS`, then runs `flag_high_rate_of_change` to
    identify outliers. The filter is applied in up to two passes: after the
    first pass, flagged samples are temporarily set to NaN and the filter is
    rerun to catch additional outliers. Final flags are the logical OR of both
    passes.

    Args:
        ds (xr.Dataset): Input dataset containing time series variables.

    Returns:
        xr.Dataset: Dataset (same object) with the rate-of-change filter applied.
    """

    patterns = [re.compile(p) for p in DEFAULT_VARIABLE_THRESHOLDS]

    vars_with_thresholds = [
        v for v in ds.data_vars
        if any(p.match(v) for p in patterns)
    ]

    for var in vars_with_thresholds:
        _, _, flag_combined, flag_final = flag_high_rate_of_change(ds, var, window="7D")

        tmp = ds.copy(deep=True)
        tmp[var].loc[{"time": flag_final.time[flag_final]}] = np.nan  # apply first pass to temporary object

        if flag_final.any():
            _, _, flag_combined2, flag2 = flag_high_rate_of_change(tmp, var, window="7D")
            flag_combined2 = flag_combined2.reindex_like(flag_combined, fill_value=False)
            flag2 = flag2.reindex_like(flag_final, fill_value=False)

            flag_combined = (flag_combined | flag_combined2)
            flag_final = (flag_final | flag2)
        else:
            flag2 = None
        flag_final = flag_final.reindex_like(ds.time, fill_value=False)

        ds = set_and_remove_flagged_data(ds, var, flag='ROC', mask=flag_final)

    return ds
