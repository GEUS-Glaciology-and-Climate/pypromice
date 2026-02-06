import logging
import numpy as np
import pandas as pd
import xarray as xr
import re
from pypromice.core.qc.common import set_flag

logger = logging.getLogger(__name__)


DEFAULT_VARIABLE_THRESHOLDS = {
    r"^t_[iul]$": {"tol": 3, "factor": 2.2},
    r"^p_[iul]$": {"tol": 1.0, "factor": 2.0},
    r"^rh_[iul]$": {"tol": 2.0, "factor": 3.5},
    r"^wspd_[iul]$": {"tol": 0.5, "factor": 2.0},
    r"^t_i_(?:10|[1-9])$": {"tol": 0.5, "factor": 1.5},
}

def _get_params(var, tol=None, factor=None):
    if tol is not None and factor is not None:
        logger.debug(f"{var}: using user tol={tol}, factor={factor}")
        return float(tol), float(factor)

    for pat, cfg in DEFAULT_VARIABLE_THRESHOLDS.items():
        if re.match(pat, var):
            vt = float(cfg["tol"]) if tol is None else float(tol)
            vf = float(cfg["factor"]) if factor is None else float(factor)
            return vt, vf

    vt = 0.1 if tol is None else float(tol)
    vf = 2.0 if factor is None else float(factor)
    return vt, vf

def unflag_if_linear_interp(ds, var, flag, tol=0.1, time="time"):
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
    time : str, optional
        Name of the time dimension.

    Returns
    -------
    xarray.DataArray
        Updated flag array where linearly interpolated points are unflagged.
    """
    # Align variable to flag time axis
    v = ds[var].sel({time: flag[time]}).astype("float64")
    t = v[time].values.astype("datetime64[ns]").astype("int64")

    n = v.sizes[time]
    idx = np.arange(n, dtype=np.int64)

    # Valid (non-flagged and finite) samples
    ok = (~flag) & np.isfinite(v)

    # Index of previous valid sample
    prev_idx = xr.DataArray(np.where(ok.values, idx, np.nan),
                            coords={time: v[time].values}, dims=(time,)).ffill(time)

    # Index of next valid sample
    next_idx = xr.DataArray(np.where(ok.values, idx, np.nan),
                            coords={time: v[time].values}, dims=(time,)).bfill(time)

    pi = prev_idx.values
    ni = next_idx.values

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

    # import pdb
    # import matplotlib.pyplot as plt

    # resid = np.abs(v.values - v_hat)

    # # breakpoint: flagged points that fail tol but have both neighbors
    # bad = flag.values & has_both & np.isfinite(v_hat) & (resid > tol)
    # if np.any(bad):
    #     i = int(np.flatnonzero(bad)[0])
    #     pdb.set_trace()

    #     # plot local neighborhood + interp line + tol band
    #     i0 = max(i - 50, 0)
    #     i1 = min(i + 51, n)

    #     tt = pd.to_datetime(v[time].values[i0:i1])
    #     yy = v.values[i0:i1]
    #     yh = v_hat[i0:i1]

    #     fig, ax = plt.subplots(figsize=(12, 4))
    #     ax.plot(tt, yy, marker=".", linestyle="None", markersize=3, alpha=0.7, label="data")
    #     ax.plot(tt, yh, linestyle="-", alpha=0.8, label="v_hat (interp)")
    #     ax.fill_between(tt, yh - tol, yh + tol, alpha=0.25, label="±tol")

    #     ax.axvline(pd.to_datetime(v[time].values[i]), linestyle="--")
    #     ax.scatter([pd.to_datetime(v[time].values[i])], [v.values[i]], s=120, label="bad sample")

    #     ax.grid(True, alpha=0.3)
    #     ax.legend()
    #     ax.set_title(f"{var}: residual={resid[i]:.3g} > tol={tol} at i={i}")
    #     plt.show()


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

def rate_of_change_fwd_bwd_and_thresholds(da, var, window="7D", time="time", per="h",
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
        time (str, optional): Name of the time dimension. Defaults to "time".
        per (str, optional): Time unit used to normalize rates (e.g. "h", "D").
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

    t = da[time].values.astype("datetime64[ns]")
    v = da.values.astype("float64")

    dt_ns = (t[1:] - t[:-1]).astype("timedelta64[ns]").astype("int64")
    dv = v[1:] - v[:-1]
    denom_ns = np.timedelta64(1, per).astype("timedelta64[ns]").astype("int64")
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
        coords={time: da[time].values[1:]},
        dims=(time,),
        name=f"{var}_high_var_flag_fwd",
    )
    flag_bwd = xr.DataArray(
        flag_bwd_s.values.astype(bool),
        coords={time: da[time].values[:-1]},
        dims=(time,),
        name=f"{var}_high_var_flag_bwd",
    )

    tfull = da[time].values
    fwd_full = xr.DataArray(np.zeros(tfull.shape, bool), coords={time: tfull}, dims=(time,))
    bwd_full = xr.DataArray(np.zeros(tfull.shape, bool), coords={time: tfull}, dims=(time,))
    fwd_full.loc[{time: flag_fwd[time]}] = flag_fwd
    bwd_full.loc[{time: flag_bwd[time]}] = flag_bwd

    roc_ds = xr.Dataset(
        data_vars=dict(
            roc_rate=(("time_rate",), rate),
            roc_thr_fwd=(("time_fwd",), thr_fwd.values.astype("float64")),
            roc_thr_bwd=(("time_bwd",), thr_bwd.values.astype("float64")),
        ),
        coords=dict(
            time_rate=da[time].values[1:],
            time_fwd=da[time].values[1:],
            time_bwd=da[time].values[:-1],
        ),
        attrs=dict(var=var, per=per, window=window, min_periods=min_periods, factor=factor, tol=tol),
    )

    return roc_ds, fwd_full, bwd_full


def flag_high_rate_of_change(ds, var, window="7D", time="time",
                                       per="h", min_periods=10, factor=None, tol=None):
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
        time (str, optional): Name of the time dimension. Defaults to "time".
        per (str, optional): Time unit used to normalize rates (e.g. "h", "D").
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

    da = ds[var].dropna(time)

    roc_ds, fwd_full, bwd_full = rate_of_change_fwd_bwd_and_thresholds(
        da, var, window=window, time=time, per=per, min_periods=min_periods, factor=factor, tol=tol
    )

    y = da
    prev_missing = y.shift({time: 1}).isnull()
    next_missing = y.shift({time: -1}).isnull()

    tt = da[time]
    dt_prev = tt - tt.shift({time: 1})
    dt_next = tt.shift({time: -1}) - tt
    uneven_dt = dt_prev != dt_next

    is_first = xr.DataArray(np.zeros(tt.shape, dtype=bool), coords={time: tt.values}, dims=(time,))
    is_last  = xr.DataArray(np.zeros(tt.shape, dtype=bool), coords={time: tt.values}, dims=(time,))
    if (len(is_first)>0) & (len(is_last)>0):
        is_first[0] = True
        is_last[-1] = True

        prev_missing = prev_missing | is_first
        next_missing = next_missing | is_last
        uneven_dt = uneven_dt | is_first | is_last

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
        flag_final = unflag_if_linear_interp(ds, var, flag_combined, tol=tol, time=time)
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

        ds = set_flag(ds, var, flag='ROC', mask=flag_final)

    return ds
