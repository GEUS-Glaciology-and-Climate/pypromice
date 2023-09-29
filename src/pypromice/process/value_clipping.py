import numpy as np
import pandas
import xarray


def clip_values(
    ds: xarray.Dataset,
    var_configurations: pandas.DataFrame,
):
    """
    Clip values in dataset to defined "hi" and "lo" variables from dataframe.
    There is a special treatment here for rh_u and rh_l variables, where values
    are clipped and not assigned to NaN. This is for replication purposes

    Parameters
    ----------
    ds : `xarray.Dataset`
        Dataset to clip hi-lo range to
    var_configurations : `pandas.DataFrame`
        Dataframe to retrieve attribute hi-lo values from

    Returns
    -------
    ds : `xarray.Dataset`
        Dataset with clipped data
    """
    cols = ["lo", "hi", "OOL"]
    assert set(cols) <= set(var_configurations.columns)

    variable_limits = var_configurations[cols].dropna(how="all")
    for var, row in variable_limits.iterrows():
        if var not in list(ds.variables):
            continue

        if var in ["rh_u_cor", "rh_l_cor"]:
            ds[var] = ds[var].where(ds[var] >= row.lo, other=0)
            ds[var] = ds[var].where(ds[var] <= row.hi, other=100)

            # Mask out invalid corrections based on uncorrected var
            var_uncor = var.split("_cor")[0]
            ds[var] = ds[var].where(~np.isnan(ds[var_uncor]), other=np.nan)

        else:
            if ~np.isnan(row.lo):
                ds[var] = ds[var].where(ds[var] >= row.lo)
            if ~np.isnan(row.hi):
                ds[var] = ds[var].where(ds[var] <= row.hi)

        other_vars = row.OOL
        if isinstance(other_vars, str) and ~ds[var].isnull().all():
            for o in other_vars.split():
                if o not in list(ds.variables):
                    continue
                else:
                    if ~np.isnan(row.lo):
                        ds[var] = ds[var].where(ds[var] >= row.lo)
                    if ~np.isnan(row.hi):
                        ds[var] = ds[var].where(ds[var] <= row.hi)
    return ds
