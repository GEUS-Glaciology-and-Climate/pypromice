import numpy as np
import pandas
import xarray

from pypromice.utilities.dependency_graph import DependencyGraph


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

    variable_limits = var_configurations[cols].assign(
        dependents=lambda df: df.OOL.fillna("").str.split(),
        # Find the closure of dependents using the DependencyGraph class
        dependents_closure=lambda df: DependencyGraph.from_child_mapping(
            df.dependents
        ).child_closure_mapping(),
    )

    for var, row in variable_limits.iterrows():
        if var not in list(ds.variables):
            continue

        # This is a special case for rh_u_cor and rh_l_cor where values are clipped to 0 and 100.
        if var in ["rh_u_cor", "rh_l_cor"]:
            # Nan inputs should stay nan
            was_nan = ds[var].isnull()
            if ~np.isnan(row.lo):
                ds[var] = ds[var].where(ds[var] >= row.lo, other=0)
            if ~np.isnan(row.hi):
                ds[var] = ds[var].where( ds[var] <= row.hi, other=100)
            ds[var] = ds[var].where(~was_nan)
        else:
            if ~np.isnan(row.lo):
                ds[var] = ds[var].where(ds[var] >= row.lo)
            if ~np.isnan(row.hi):
                ds[var] = ds[var].where(ds[var] <= row.hi)

        # Flag dependents as NaN if parent is NaN
        for o in row.dependents_closure:
            if o not in list(ds.variables):
                continue
            ds[o] = ds[o].where(ds[var].notnull())

    return ds
