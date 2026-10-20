"""QC flag infrastructure.

QC filters flag rejected samples on a ``<var>_qc`` companion variable instead
of overwriting ``<var>``: ``flag_qc`` never touches the data.

- ``<var>_qc`` is an ``int8`` CF ``status_flag``; 0 is "OK", other codes are
  listed in ``FLAG_MEANINGS``. The first flag a sample gets is kept.
- Read data through ``clean_view(ds)`` (flagged samples as NaN), and test for
  flags with ``is_ok(ds, var)``, not ``ds[var].isnull()``.
- ``finalize_qc`` drops the flags and removes flagged data by default, or
  keeps raw values and/or flags on request.
"""
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

# Variables that are never subject to QC flagging (coordinates / bookkeeping).
NO_QC_VARS = ("time", "rec")

# Canonical, ordered list of QC flag meanings. Index == the integer code
# stored in "<var>_qc". Code 0 is always "OK".
FLAG_MEANINGS = [
    "OK",
    "PERSISTENCE",
    "RATE_OF_CHANGE",
    "OUT_OF_LIMITS",
    "DEPENDENCY",
    "MANUAL",
    "GPS_BASELINE",
    "PRECIP_SENSOR_ERROR",
    "SUN_LOWER_DOME",
    "SR_ABOVE_TOA",
    "RIME",
]
FLAG_DTYPE = "int8"
_FLAG_CODE = {name: np.int8(code) for code, name in enumerate(FLAG_MEANINGS)}


def _qc_name(var: str) -> str:
    return f"{var}_qc"


def _ensure_qc_var(ds: xr.Dataset, var: str) -> xr.Dataset:
    qc_name = _qc_name(var)
    if qc_name not in ds:
        da = ds[var]
        ds[qc_name] = xr.Variable(
            da.dims,
            np.zeros(da.shape, dtype=FLAG_DTYPE),
            {
                "long_name": f"quality control flag for {var}",
                "standard_name": "status_flag",
                "flag_values": np.arange(len(FLAG_MEANINGS), dtype=FLAG_DTYPE),
                "flag_meanings": " ".join(FLAG_MEANINGS),
            },
        )
    elif ds[qc_name].dtype != FLAG_DTYPE:
        ds[qc_name] = ds[qc_name].astype(FLAG_DTYPE)
    return ds


def is_ok(ds: xr.Dataset, var: str) -> xr.DataArray:
    """True wherever ``var`` carries no QC flag (or has no ``_qc`` yet).

    Note this says nothing about whether ``ds[var]`` is NaN -- under this
    module's "never touch the data" design, a flagged sample keeps its
    original (possibly bad) value in ``ds``. Use this function, not
    ``ds[var].isnull()``, to test whether a sample has been QC-flagged.
    """
    qc_name = _qc_name(var)
    if qc_name not in ds:
        return xr.ones_like(ds[var], dtype=bool)
    return ds[qc_name] == _FLAG_CODE["OK"]


def _flag_positions(da, index_slice):
    """Positional indexer for a ``.loc``-style ``index_slice`` on a 1-D
    DataArray, or None if the request needs the generic xarray path."""
    if not index_slice:
        return slice(None)
    if len(index_slice) != 1:
        return None
    (dim, key), = index_slice.items()
    if dim != da.dims[0]:
        return None
    index = da.indexes[dim]
    if isinstance(key, slice):
        return index.slice_indexer(key.start, key.stop, key.step)
    positions = index.get_indexer(np.atleast_1d(key))
    if (positions < 0).any():
        raise KeyError(f"labels not found in {dim!r}: {np.atleast_1d(key)[positions < 0]}")
    return positions


def _flag_qc_numpy(ds, var, code, mask, index_slice):
    """numpy implementation of flag_qc for 1-D variables. Returns False if the
    request is not supported (the caller then uses the xarray implementation)."""
    da = ds[var]
    if da.ndim != 1:
        return False
    positions = _flag_positions(da, index_slice)
    if positions is None:
        return False

    n = da.size
    if isinstance(mask, xr.DataArray):
        if mask.dims != da.dims:
            return False
        dim = da.dims[0]
        if not mask.indexes[dim].equals(da.indexes[dim]):
            mask = mask.reindex({dim: da[dim]}, fill_value=False)
        mask_values = np.asarray(mask.values, dtype=bool)
    else:
        mask_values = bool(mask)

    values = da.values
    notnull = ~np.isnan(values) if values.dtype.kind in "fc" else ~pd.isnull(values)
    qc_values = ds[_qc_name(var)].values

    cond = np.zeros(n, dtype=bool)
    cond[positions] = mask_values[positions] if isinstance(mask_values, np.ndarray) else mask_values
    cond &= notnull
    cond &= qc_values == _FLAG_CODE["OK"]
    if cond.any():
        new_qc = qc_values.copy()
        new_qc[cond] = code
        ds[_qc_name(var)].values = new_qc
    return True


def _flag_qc_xarray(ds, var, code, mask, index_slice):
    """Generic (slower) xarray implementation of flag_qc."""
    qc_name = _qc_name(var)
    qc = ds[qc_name].loc[index_slice or {}]
    data = ds[var].loc[index_slice or {}]
    if qc.size == 0:
        return ds
    m = mask.loc[index_slice or {}] if isinstance(mask, xr.DataArray) else mask
    cond = m & data.notnull() & (qc == _FLAG_CODE["OK"])
    ds[qc_name].loc[index_slice or {}] = xr.where(cond, code, qc).astype(FLAG_DTYPE)
    return ds


def flag_qc(
    ds: xr.Dataset,
    var: str,
    flag_name: str,
    mask,
    index_slice: Optional[dict] = None,
) -> xr.Dataset:
    """Flag samples of ``var`` as failing the ``flag_name`` QC step.

    Only records the flag on "<var>_qc" -- ``ds[var]`` itself is never
    modified. Only samples that are still "OK" and non-null are flagged:
    the first QC step to reject a sample owns the reason recorded for it.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to update (a new dataset is returned; ``ds`` itself is not
        mutated in place).
    var : str
        Variable being flagged.
    flag_name : str
        One of ``FLAG_MEANINGS`` (other than "OK").
    mask : xr.DataArray or bool
        Boolean mask/scalar, True where this QC step rejects the sample.
    index_slice : dict, optional
        Optional ``.loc``-style slice (e.g. ``{"time": slice(t0, t1)}``)
        restricting where the flag can be applied. Defaults to the full
        variable.

    Returns
    -------
    xr.Dataset
    """
    if var in NO_QC_VARS or var not in ds:
        return ds
    if ds[var].ndim == 0:
        return ds
    if flag_name not in _FLAG_CODE:
        raise ValueError(
            f"Unknown QC flag {flag_name!r}, expected one of {FLAG_MEANINGS}"
        )

    ds = _ensure_qc_var(ds, var)
    code = _FLAG_CODE[flag_name]
    if _flag_qc_numpy(ds, var, code, mask, index_slice):
        return ds
    return _flag_qc_xarray(ds, var, code, mask, index_slice)


def clean_view(ds: xr.Dataset) -> xr.Dataset:
    """Return a disposable copy of ``ds`` with every flagged sample set to
    NaN, for QC filters and derived-variable calculations to safely read
    from. ``ds`` itself is never modified.
    """
    ds_clean = ds.copy(deep=True)
    for var in list(ds_clean.data_vars):
        if var.endswith("_qc") or var in NO_QC_VARS:
            continue
        qc_name = _qc_name(var)
        if qc_name not in ds_clean:
            continue
        flagged = ds_clean[qc_name].values != _FLAG_CODE["OK"]
        if not flagged.any():
            continue
        da = ds_clean[var]
        ds_clean[var] = xr.Variable(
            da.dims, np.where(flagged, np.nan, da.values), da.attrs
        )
    return ds_clean


def has_qc_flags(ds: xr.Dataset) -> bool:
    """True if ``ds`` carries at least one "<var>_qc" flag variable."""
    return any(v.endswith("_qc") for v in ds.data_vars)


def combine_first_qc(ds1: xr.Dataset, ds2: xr.Dataset) -> xr.Dataset:
    """``ds1.combine_first(ds2)`` that keeps the QC flags consistent.

    ds1 is preferred, ds2 fills the gaps, where a flagged sample counts as a
    gap: an OK value of ds2 is preferred to a flagged raw value of ds1. Each
    "<var>_qc" follows the value that was chosen (a missing flag variable
    means "never flagged"), so the result is the same as combining the two
    ``clean_view`` datasets, except that flagged raw values are kept.
    """
    if not (has_qc_flags(ds1) or has_qc_flags(ds2)):
        return ds1.combine_first(ds2)

    qc_names = sorted({v for ds in (ds1, ds2) for v in ds.data_vars
                       if v.endswith("_qc")})
    out = ds1.drop_vars([q for q in qc_names if q in ds1]).combine_first(
        ds2.drop_vars([q for q in qc_names if q in ds2]))

    time = out.time
    for q in qc_names:
        var = q[:-3]
        if var not in out or out[var].dims != ("time",):
            continue

        def values_and_flags(ds):
            val = (ds[var].reindex(time=time).values if var in ds
                   else np.full(time.size, np.nan))
            flag = (ds[q].reindex(time=time, fill_value=0).values if q in ds
                    else np.zeros(time.size, dtype=FLAG_DTYPE))
            return val, flag

        v1, q1 = values_and_flags(ds1)
        v2, q2 = values_and_flags(ds2)
        ok1 = ~np.isnan(v1) & (q1 == 0)
        ok2 = ~np.isnan(v2) & (q2 == 0)

        out[var].values = np.where(ok1, v1, np.where(ok2, v2, out[var].values))
        flag = np.select(
            [ok1 | ok2, ~np.isnan(v1), ~np.isnan(v2)],
            [0, q1, q2],
            default=np.where(q1 != 0, q1, q2),
        ).astype(FLAG_DTYPE)
        out[q] = xr.Variable(("time",), flag, (ds1[q] if q in ds1 else ds2[q]).attrs)
    return out


def finalize_qc(
    ds: xr.Dataset,
    keep_flagged_data: bool = False,
    keep_qc_flags: bool = False,
) -> xr.Dataset:
    """Finalize accumulated QC flags at the end of a pipeline stage.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``<var>_qc`` companions produced by one or more QC
        filters. Every variable still holds its true original reading,
        whether or not it ended up flagged.
    keep_flagged_data : bool, optional
        If False (default), returns ``clean_view(ds)`` with every
        ``<var>_qc`` variable dropped -- flagged samples become NaN, so the
        result matches a non-flag-based pipeline's output exactly.
        If True, ``ds`` is returned unchanged: every variable keeps its
        true reading (flagged or not) with its ``<var>_qc`` companion
        attached, for diagnostics.
    keep_qc_flags : bool, optional
        Only used when ``keep_flagged_data`` is False. If True, flagged
        samples are still set to NaN but the ``<var>_qc`` variables are kept,
        so the file says *why* each sample is missing. Defaults to False.

    Returns
    -------
    xr.Dataset
    """
    if keep_flagged_data:
        return ds

    ds = clean_view(ds)
    if keep_qc_flags:
        return ds
    qc_vars = [v for v in ds.data_vars if v.endswith("_qc")]
    return ds.drop_vars(qc_vars)
