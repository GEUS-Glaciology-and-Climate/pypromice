from typing import Dict, Set, Mapping

import numpy as np
import pandas
import pandas as pd
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
    # TODO: Check if this is necessary
    # variable_limits = var_configurations[cols].dropna(how="all")

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
        # TODO: Check if this is necessary
        # I guess the nan flagging is already handled below
        # What if rh_u_cor is nan?
        # What if row.lo/hi is nan?

        if var in ["rh_u_cor", "rh_l_cor"]:
            ds[var] = ds[var].where(ds[var] >= row.lo, other=0)
            ds[var] = ds[var].where(ds[var] <= row.hi, other=100)

            # Mask out invalid corrections based on uncorrected var
            var_uncor = var.rstrip("_cor")
            ds[var] = ds[var].where(~np.isnan(ds[var_uncor]), other=np.nan)

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
