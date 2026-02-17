import numpy as np
import xarray as xr

def set_flag(
    qc: xr.DataArray,
    mask: xr.DataArray,
    flag: str,
    delimiter=", "
) -> xr.DataArray:
    """Set or append a QC flag on a QC DataArray.

    If qc == "OK", replaces with `flag`.
    If qc != "OK", appends `flag` unless already present.

    Parameters
    ----------
    qc : xr.DataArray
        QC data array
    mask : xr.DataArray
        Boolean data array for flagging
    flag : str
        String descriptor for flag
    da : xr.DataArray
        Input data array (used to avoid flagging NaNs)
    index_slice : dict, optional
        Index slice to apply flagging to
    delimiter : str
        Separator used when appending flags

    Returns
    -------
    qc : xr.DataArray
        Updated QC data array
    """
    if qc is None:
        raise ValueError("QC array is required (cannot be None)")

    if qc.shape != mask.shape:
        raise ValueError(f"QC shape {qc.shape} != mask shape {mask.shape}")

    # Ensure object dtype for string operations
    if qc.dtype.kind in ("U", "S"):
        qc = qc.astype(object)

    def _append_flag(qval, flag=flag):
        # Logic handling for flag construction
        if qval is None:
            return flag
        qval = str(qval)
        if qval == "OK":
            return flag
        # Prevent duplicates (simple containment check)
        if flag in [s.strip() for s in qval.split(",")]:
            return qval
        return f"{qval}{delimiter}{flag}"

    updated = xr.apply_ufunc(
        _append_flag,
        q,
        vectorize=True,
        dask="allowed",
        output_dtypes=[object],
    )

    qc = xr.where(mask, updated, qc)
    return qc



def apply_flag(da: xr.DataArray,
               qc: xr.DataArray,
) -> xr.DataArray:
    """Set flag on data array

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    qc : xr.DataArray
        Quality control flag array

    Returns
    -------
    xr.DataArray
        Filtered data array
    """
    bad = qc.astype(str) != "OK"
    return da.where(~bad)


def remove_flagged_data(ds: xr.Dataset) -> xr.Dataset:
    """Set all flags on dataset

    Parameters
    ----------
    ds: xr.Dataset
        Input dataset

    Returns
    -------
    ds_out: xr.Dataset
        Filtered dataset
    """
    ds_out = ds.copy(deep=True)
    for v in list(ds_out.data_vars):
        if hasattr(ds_out[v], "ancillary_variables"):
            qc = ds_out[v].attrs["ancillary_variables"]
            if qc in ds_out and "time" in ds_out[v].dims:
                ds_out[v] = apply_flag(ds_out[v], ds_out[qc])
    return ds_out
