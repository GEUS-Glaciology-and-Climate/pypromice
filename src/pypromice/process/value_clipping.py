from typing import Sequence

import numpy as np
import pandas
import xarray


def clip_values(
    ds: xarray.Dataset,
    df: pandas.DataFrame,
    cols: Sequence[str] = ("lo", "hi", "OOL"),
):
    """
    Clip values in dataset to defined "hi" and "lo" variables from dataframe.
    There is a special treatment here for rh_u and rh_l variables, where values
    are clipped and not assigned to NaN. This is for replication purposes

    Parameters
    ----------
    ds : `xarray.Dataset`
        Dataset to clip hi-lo range to
    df : `pandas.DataFrame`
        Dataframe to retrieve attribute hi-lo values from

    Returns
    -------
    ds : `xarray.Dataset`
        Dataset with clipped data
    """

    df = df[cols]
    df = df.dropna(how="all")
    lo = cols[0]
    hi = cols[1]
    ool = cols[2]
    for var in df.index:
        if var not in list(ds.variables):
            continue

        if var in ["rh_u_cor", "rh_l_cor"]:
            ds[var] = ds[var].where(ds[var] >= df.loc[var, lo], other=0)
            ds[var] = ds[var].where(ds[var] <= df.loc[var, hi], other=100)

            # Mask out invalid corrections based on uncorrected var
            var_uncor = var.split("_cor")[0]
            ds[var] = ds[var].where(~np.isnan(ds[var_uncor]), other=np.nan)

        else:
            if ~np.isnan(df.loc[var, lo]):
                ds[var] = ds[var].where(ds[var] >= df.loc[var, lo])
            if ~np.isnan(df.loc[var, hi]):
                ds[var] = ds[var].where(ds[var] <= df.loc[var, hi])

        other_vars = df.loc[var][ool]
        if isinstance(other_vars, str) and ~ds[var].isnull().all():
            for o in other_vars.split():
                if o not in list(ds.variables):
                    continue
                else:
                    if ~np.isnan(df.loc[var, lo]):
                        ds[o] = ds[o].where(ds[var] >= df.loc[var, lo])
                    if ~np.isnan(df.loc[var, hi]):
                        ds[o] = ds[o].where(ds[var] <= df.loc[var, hi])
    return ds
