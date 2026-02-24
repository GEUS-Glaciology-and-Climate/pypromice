import numpy as np
import pandas
import xarray

from pypromice.core.dependency_graph import DependencyGraph
from pypromice.core.qc.common import set_flag


def clip_values(
    ds: xarray.Dataset,
    var_configurations: pandas.DataFrame,
):
    """Apply physical limits and propagate quality-control flags.

    Checks each variable against configured lower/upper bounds and updates
    associated quality-control (QC) variables instead of silently discarding
    data. Values outside limits are flagged as "OOL" (out of limits). Variables
    that are flagged because of their dependency on an invalid parent variable are
    flagged as "DEP".

    Args:
        ds (xr.Dataset): Dataset containing measurement variables and their
            corresponding "<var>_qc" quality flag variables.
        var_configurations (pd.DataFrame): Table containing per-variable
            configuration with columns:
            - "lo": Lower physical limit (NaN if not used)
            - "hi": Upper physical limit (NaN if not used)
            - "dependent_variables": Space-separated list of variables that
              depend on the parent variable.

    Returns:
        xr.Dataset: Dataset with:
            - Data outside limits replaced by NaN
            - QC flags set to:
                * "OK"  – value valid
                * "OOL" – value outside configured limits
                * "DEPENDENCY" – value invalid due to dependency on a flagged parent
    """
    cols = ["lo", "hi", "dependent_variables"]
    assert set(cols) <= set(var_configurations.columns)

    variable_limits = var_configurations[cols].assign(
        dependents=lambda df: df.dependent_variables.fillna("").str.split(),
        dependents_closure=lambda df: DependencyGraph.from_child_mapping(
            df.dependents
        ).child_closure_mapping(),
    )

    for var, row in variable_limits.iterrows():
        if var not in ds.variables: continue
        if np.isnan(row.lo) and np.isnan(row.hi): continue
        bad = xarray.zeros_like(ds[var], dtype=bool)
        if ~np.isnan(row.lo):
            bad = bad | (ds[var] < row.lo)
        if ~np.isnan(row.hi):
            bad = bad | (ds[var] > row.hi)

        var_qc = ds[var].attrs.get["ancillary_variables"]
        ds[var_qc] = set_flag(ds[var_qc],
                              bad,
                              "OOL")

        for dep_var in row.dependents_closure:
            if dep_var not in ds.variables: continue
            dep_var_qc = ds[dep_var].attrs.get["ancillary_variables"]
            dep_bad = ds[var].isnull() |  (ds[var_qc] != "OK")
            ds[dep_var_qc] = set_flag(ds[dep_var_qc],
                                      dep_bad,
                                      "DEPENDENCY")

    return ds