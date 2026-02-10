import xarray as xr
import numpy as np
NO_QC_VAR = ['time','rec']


def set_flag(ds: xr.Dataset, v: str, flag: str, index_slice=None, mask=None) -> xr.Dataset:
    if v in NO_QC_VAR: return ds

    if index_slice is None:
        index_slice = {"time": slice(None, None)}

    vqc = f"{v}_qc"
    if vqc not in ds:
        ds[vqc] = xr.DataArray(
            np.full(ds[v].shape, "OK", dtype=object),
            coords=ds[v].coords,
            dims=ds[v].dims,
        )
    else:
        if ds[vqc].dtype.kind in ("U", "S"):
            ds[vqc] = ds[vqc].astype(object)

    q = ds[vqc].loc[index_slice]
    x = ds[v].loc[index_slice]
    if q.size == 0:
        return ds

    m = xr.ones_like(x, dtype=bool) if mask is None else (
        mask.loc[index_slice] if isinstance(mask, xr.DataArray) else mask
    )

    cond = m & x.notnull() & (q == "OK")

    ds[vqc].loc[index_slice] = xr.where(cond, str(flag), q)
    return ds


def remove_flagged_data(ds):
    ds_out = ds.copy(deep=True)
    for v in list(ds_out.data_vars):
        if v.endswith("_qc"):
            continue
        qc = f"{v}_qc"
        if qc in ds_out and "time" in ds_out[v].dims:
            bad = ds_out[qc].astype(str) != "OK"
            ds_out[v] = ds_out[v].where(~bad)
    return ds_out
