import xarray as xr
import numpy as np
NO_QC_VAR = ['time','rec']


def set_flag(da: xr.DataArray,
             flag: str,
             index_slice=None,
             mask=None,
             qc=None
) -> xr.DataArray:
    """Set flag on data array

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    flag : str
        String descriptor for flag
    index_slice : xr.DataArray
        Index slice to apply flagging to. Default is None.
    mask : xr.DataArray
        Boolean data array for flagging. Default is None.
    qc : xr.DataArray
        QC data array. Default is None.

    Returns
    -------
    qc : xr.DataArray
        QC data array
    """
    if index_slice is None:
        index_slice = {"time": slice(None, None)}

    if qc is None:
        qc = xr.DataArray(
            np.full(da.shape, "OK", dtype=object),
            coords=da.coords,
            dims=da.dims,
        )
    elif qc.dtype.kind in ("U", "S"):
        qc = qc.astype(object)

    q = qc.loc[index_slice]
    x = da.loc[index_slice]

    if q.size == 0:
        return qc

    m = xr.ones_like(x, dtype=bool) if mask is None else (
        mask.loc[index_slice] if isinstance(mask, xr.DataArray) else mask
    )

    cond = m & x.notnull() & (q == "OK")

    qc.loc[index_slice] = xr.where(cond, str(flag), q)

    return qc


def apply_flag(da: xr.DataArray,
               qc: xr.DataArray,
) -> xr.DataArray:
    bad = qc.astype(str) != "OK"
    return da.where(~bad)


def remove_flagged_data(ds: xr.Dataset) -> xr.Dataset:
    ds_out = ds.copy(deep=True)
    for v in list(ds_out.data_vars):
        if v.endswith("_qc"):
            continue
        qc = f"{v}_qc"
        if qc in ds_out and "time" in ds_out[v].dims:
#            bad = ds_out[qc].astype(str) != "OK"
#            ds_out[v] = ds_out[v].where(~bad)
            ds_out[v] = apply_flag(ds_out[v], ds_out[qc])
    return ds_out
