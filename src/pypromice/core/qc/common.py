"""Shared quality-control (QC) flag infrastructure.

Every QC filter in this package (persistence, rate-of-change, value clipping,
manual GitHub-issue flags) marks samples it rejects on a companion
``<var>_qc`` integer variable instead of overwriting ``<var>`` with no
record of why. This keeps every variable's true, original reading in ``ds``
for as long as the pipeline runs, so a station file can eventually carry
both the raw values and the full set of flags explaining which ones were
excluded and why.

Encoding
--------
``<var>_qc`` follows the CF "status_flag" convention: a small integer with
mutually-exclusive codes and ``flag_values``/``flag_meanings`` attributes.
Code 0 is always "OK"; every other code names the QC step that rejected the
sample. See ``FLAG_MEANINGS`` for the canonical, ordered list -- add new QC
steps there, not as ad-hoc string literals in individual filter modules.

First flag wins
----------------
A sample keeps the *first* flag it receives. Once a sample is flagged,
``flag_qc`` will not let a later QC step overwrite the reason (it also won't
flag an already-NaN sample -- there's nothing to explain there). QC filters
therefore effectively run in a fixed priority order: whichever filter runs
first "claims" a bad sample.

Reading and flagging the correct image of the dataset
-------------------------------------------------------
``flag_qc`` only ever records the flag -- it never touches ``ds[var]``
itself. This means the true, original reading of every variable stays
available in ``ds`` no matter how many QC steps have flagged it.

The corollary: anything that *consumes* a variable's value -- another QC
filter detecting a new issue, or a derived-variable calculation further
down the pipeline -- must not read straight from ``ds``, since a flagged
sample sitting there is still its raw, potentially bad, value. Use
``clean_view(ds)`` to get a disposable copy with every already-flagged
sample replaced by NaN, and read from *that* instead. Every QC filter in
this package builds its own ``clean_view`` before evaluating new samples,
and the pipeline stage that calls them builds one before computing any
derived variable from raw sensor readings.

Dependency checks (e.g. "is this variable's parent currently bad?") should
use ``~is_ok(ds, var)`` rather than ``ds[var].isnull()`` -- a flagged
sample is not NaN in ``ds`` itself, only in a ``clean_view`` of it.

Finalizing
----------
``finalize_qc`` is called once, at the end of the pipeline stage that ran
these filters. By default (``keep_flagged_data=False``) it applies
``clean_view`` and drops every ``<var>_qc`` variable, matching exactly what
a non-flag-based pipeline produces. Pass ``keep_flagged_data=True`` to
instead get everything back untouched: every variable's true original
reading, raw sensor readings and derived variables alike, with its
``<var>_qc`` companion attached.
"""
from typing import Optional

import numpy as np
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
]
FLAG_DTYPE = "int8"
_FLAG_CODE = {name: np.int8(code) for code, name in enumerate(FLAG_MEANINGS)}


def _qc_name(var: str) -> str:
    return f"{var}_qc"


def _ensure_qc_var(ds: xr.Dataset, var: str) -> xr.Dataset:
    qc_name = _qc_name(var)
    if qc_name not in ds:
        ds[qc_name] = xr.DataArray(
            np.zeros(ds[var].shape, dtype=FLAG_DTYPE),
            coords=ds[var].coords,
            dims=ds[var].dims,
            attrs={
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

    if index_slice is None:
        index_slice = {}

    qc_name = _qc_name(var)
    qc = ds[qc_name].loc[index_slice]
    data = ds[var].loc[index_slice]
    if qc.size == 0:
        return ds

    m = mask.loc[index_slice] if isinstance(mask, xr.DataArray) else mask

    cond = m & data.notnull() & (qc == _FLAG_CODE["OK"])
    ds[qc_name].loc[index_slice] = xr.where(cond, _FLAG_CODE[flag_name], qc).astype(
        FLAG_DTYPE
    )
    return ds


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
        if qc_name in ds_clean:
            ds_clean[var] = ds_clean[var].where(is_ok(ds_clean, var))
    return ds_clean


def finalize_qc(ds: xr.Dataset, keep_flagged_data: bool = False) -> xr.Dataset:
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

    Returns
    -------
    xr.Dataset
    """
    if keep_flagged_data:
        return ds

    ds = clean_view(ds)
    qc_vars = [v for v in ds.data_vars if v.endswith("_qc")]
    return ds.drop_vars(qc_vars)
