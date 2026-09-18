import numpy as np
import pandas
import xarray

from pypromice.core.dependency_graph import DependencyGraph
from pypromice.core.qc.common import flag_qc, is_ok


def clip_values(
    ds: xarray.Dataset,
    var_configurations: pandas.DataFrame,
):
    """
    Flag values outside the defined "hi"/"lo" range from dataframe, and
    propagate the flag to dependent variables.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Dataset to clip hi-lo range to
    var_configurations : `pandas.DataFrame`
        Dataframe to retrieve attribute hi-lo values from

    Returns
    -------
    ds : `xarray.Dataset`
        Dataset with out-of-limits data flagged "OUT_OF_LIMITS" (and their
        dependents flagged "DEPENDENCY") in "<var>_qc" (data itself is
        unchanged; use finalize_qc to remove it)
    """
    cols = ["lo", "hi", "dependent_variables"]
    assert set(cols) <= set(var_configurations.columns)

    variable_limits = var_configurations[cols].assign(
        dependents=lambda df: df.dependent_variables.fillna("").str.split(),
        # Find the closure of dependents using the DependencyGraph class
        dependents_closure=lambda df: DependencyGraph.from_child_mapping(
            df.dependents
        ).child_closure_mapping(),
    )

    for var, row in variable_limits.iterrows():
        if var not in list(ds.variables):
            continue

        bad = xarray.zeros_like(ds[var], dtype=bool)
        if ~np.isnan(row.lo):
            bad = bad | (ds[var] < row.lo)
        if ~np.isnan(row.hi):
            bad = bad | (ds[var] > row.hi)
        ds = flag_qc(ds, var, "OUT_OF_LIMITS", mask=bad)

        # Flag dependents as bad if parent is NaN (genuinely missing) or
        # flagged (above, or by an earlier QC step) -- a flagged parent is
        # NOT NaN in `ds` itself under this module's "never touch the data"
        # design, so this must check the qc flag, not just isnull().
        parent_bad = ds[var].isnull() | ~is_ok(ds, var)
        for o in row.dependents_closure:
            if o not in list(ds.variables):
                continue
            ds = flag_qc(ds, o, "DEPENDENCY", mask=parent_bad)

    return ds
