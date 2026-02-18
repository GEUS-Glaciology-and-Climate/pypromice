import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
from pypromice.core.qc.common import remove_flagged_data, set_flag

__all__ = [
    'flagNAN',
    'adjustTime',
    'adjustData',
]

logger = logging.getLogger(__name__)

def flagNAN(ds, flag_dir):
    '''Apply manual flag intervals from CSV to QC variables and mask non-OK samples.

    Reads a station-specific CSV file containing [t0, t1, variable, flag] rows.
    For each row, assigns the provided flag value to "<var>_qc" over the given
    time interval (inclusive). Before applying manual flags, a working copy is
    created where any samples with "<var>_qc" != "OK" are set to NaN.

    Parameters
    ----------
    ds : xr.Dataset
        Level 0 dataset
    flag_dir : str
        File directory where .csv flag files can be found

    Returns
    -------
    ds : xr.Dataset
        Level 0 data with flagged data
    '''
    df = _getDF(os.path.join(flag_dir, ds.attrs["station_id"] + ".csv"), list(ds.keys()))

    if isinstance(df, pd.DataFrame):
        df.t0 = pd.to_datetime(df.t0).dt.tz_localize(None)
        df.t1 = pd.to_datetime(df.t1).dt.tz_localize(None)

        if df.shape[0] > 0:
            for i in df.index:
                t0, t1, avar, fval = df.loc[i, ["t0", "t1", "variable", "flag"]]

                if avar == "*":
                    varlist = list(ds.keys())
                elif "*" in avar:
                    varlist = pd.DataFrame(columns=list(ds.keys())).filter(regex=(avar)).columns
                else:
                    varlist = avar.split()

                varlist = [v for v in varlist if (v != "time") and ("_qc" not in v)]

                if pd.isnull(t0):
                    t0 = pd.to_datetime(ds["time"].values[0])
                if pd.isnull(t1):
                    t1 = pd.to_datetime(ds["time"].values[-1])

                # construct boolean mask once per interval
                mask = (ds["time"] >= np.datetime64(t0)) & (ds["time"] <= np.datetime64(t1))

                for v in varlist:
                    if v not in ds:
                        logger.debug(f"---> could not flag {v} not in dataset")
                        continue

                    v_qc = ds[v].attrs["ancillary_variables"]

                    ds[v_qc] = set_flag(ds[v_qc],
                                        mask,
                                        str(fval)
                                        )
                    logger.debug(
                        f"---> flagging {t0} {t1} {v} with {fval}, "
                        f"flagged {int(mask.sum())}/{ds.dims['time']}"
                    )

    return ds


def adjustTime(ds, adj_dir, var_list=[], skip_var=[]):
    '''Read adjustment data from .csv file. Only applies the "time_shift" adjustment

    Parameters
    ----------
    ds : xr.Dataset
        Level 0 dataset
    adj_dir : str
        File directory where .csv adjustment files can be found

    Returns
    -------
    ds : xr.Dataset
        Level 0 data with flagged data
    '''
    ds_out = ds.copy(deep=True)
    adj_info=None

    adj_info = _getDF(os.path.join(adj_dir, ds.attrs["station_id"] + ".csv"), list(ds_out.keys()))

    if not isinstance(adj_info, pd.DataFrame): return ds_out
    if not "time_shift" in adj_info.adjust_function.values: return ds_out

    time_shifts = adj_info.loc[adj_info.adjust_function == "time_shift", :]

    for t0, t1, val in zip(
        time_shifts.t0,
        time_shifts.t1,
        time_shifts.adjust_value,
    ):
        ds_shifted = ds_out.sel(time=slice(t0,t1))
        ds_shifted['time'] = ds_shifted.time.values + pd.Timedelta(days = val)

        # here we concatenate what was before the shifted part, the shifted
        # part and what was after the shifted part
        # note that if any data was already present in the target period
        # (where the data lands after the shift), it is overwritten

        ds_out = xr.concat(
                                (
                                    ds_out.sel(time=slice(pd.to_datetime(ds_out.time.values[0]),
                                                          t0 + pd.Timedelta(days = val))),
                                    ds_shifted,
                                    ds_out.sel(time=slice(t1 + pd.Timedelta(days = val),
                                                          pd.to_datetime(ds_out.time.values[-1])))
                                ),
                                dim = 'time',
                               )
        if t0 > pd.Timestamp.now():
            ds_out = ds_out.sel(time=slice(pd.to_datetime(ds_out.time.values[0]),
                                           t0))
    return ds_out


def adjustData(ds, adj_dir, var_list=None, skip_var=None):
    '''Read adjustment data from .csv file. For each variable, and downstream
    dependents, adjust data accordingly if set in the adjustment .csv

    Parameters
    ----------
    ds : xr.Dataset
        Level 0 dataset
    adj_dir : str
        File directory where .csv adjustment files can be found

    Returns
    -------
    ds : xr.Dataset
        Level 0 data with flagged data
    '''

    ds_work = remove_flagged_data(ds)

    adj_info = _getDF(os.path.join(adj_dir, ds.attrs["station_id"] + ".csv"), list(ds_work.keys()))

    if not isinstance(adj_info, pd.DataFrame): return ds

    # removing potential time shifts from the adjustment list
    adj_info = adj_info.loc[adj_info.adjust_function != "time_shift", :]

    if var_list is None:
        var_list = np.unique(adj_info.variable)
    else:
        adj_info = adj_info.loc[np.isin(adj_info.variable, var_list), :]
        var_list = np.unique(adj_info.variable)

    if not skip_var is None:
        adj_info = adj_info.loc[~np.isin(adj_info.variable, skip_var), :]
        var_list = np.unique(adj_info.variable)
    var_list = [v for v in var_list if (v != "time") and ("_qc" not in v)]

    for var in var_list:
        if var not in list(ds_work.keys()):
            logger.info(f'could not adjust {var } not in dataset')
            continue
        for t0, t1, func, val in zip(
            adj_info.loc[var].t0,
            adj_info.loc[var].t1,
            adj_info.loc[var].adjust_function,
            adj_info.loc[var].adjust_value,
        ):

            index_slice = dict(time=slice(t0, t1))
            if len(ds_work[var].loc[index_slice].time.time) == 0:
                logger.info(f'---> {t0} {t1} {var} {func} {val}')
                logger.info("Time range does not intersect with dataset")
                continue

            if func in ["add", "multiply", "rotate"]:
                ds = dispatch_adjustment(ds, var, {"time": slice(t0, t1)}, func, val)
            else:
                ds_work = dispatch_adjustment(ds_work, var, {"time": slice(t0, t1)}, func, val)
                ds[var+'_qc'] = ds_work[var+'_qc']
    return ds

# here is a list of simple modifying functions
def _h_add(ds, var, sl, val):
    ds[var].loc[sl] = ds[var].loc[sl].values + val
    return ds, None

def _h_multiply(ds, var, sl, val):
    ds[var].loc[sl] = ds[var].loc[sl].values * val
    if ("DW" in var) or ("wspd" in var):
        ds[var].loc[sl] = ds[var].loc[sl] % 360
    return ds, None

def _h_rotate(ds_work, var, sl, val):
    ds_work[var].loc[sl] = (ds_work[var].loc[sl].values + val) % 360
    return ds_work, None

def _h_min_filter(ds_work, var, sl, val):
    tmp = ds_work[var].loc[sl]
    bad = tmp < val
    ds_work = set_flag(ds_work, var, index_slice=sl, flag="ADJ_MIN", mask=bad)
    return ds_work, bad.sum().item()

def _h_max_filter(ds_work, var, sl, val):
    tmp = ds_work[var].loc[sl]
    bad = tmp > val
    ds_work = set_flag(ds_work, var, index_slice=sl, flag="ADJ_MAX", mask=bad)
    return ds_work, bad.sum().item()

def _h_hampel_filter(ds_work, var, sl, val):
    tmp1 = _hampel(ds_work[var].loc[sl], k=7 * 24, t0=val)
    bad = ds_work[var].loc[sl].notnull() & tmp1.isnull()
    ds_work = set_flag(ds_work, var, index_slice=sl, flag="ADJ_HAMPEL", mask=bad)
    return ds_work, bad.sum().item()

def _h_grad_filter(ds_work, var, sl, val):
    tmp = ds_work[var].loc[sl].copy(deep=True)
    msk = tmp.diff()
    bad = np.roll(np.abs(msk).values > val, -1)
    bad = xr.DataArray(bad, coords_work=tmp.coords_work, dims=tmp.dims)
    ds_work = set_flag(ds_work, var, index_slice=sl, flag="ADJ_GRAD", mask=bad)
    return ds_work, bad.sum().item()

# object to link the keyword for the flag DB with the function to be used
HANDLERS = {
    "add": _h_add,
    "multiply": _h_multiply,
    "min_filter": _h_min_filter,
    "max_filter": _h_max_filter,
    "hampel_filter": _h_hampel_filter,
    "grad_filter": _h_grad_filter,
    "rotate": _h_rotate,
}

def dispatch_adjustment(ds_work, var, sl, func, val):
    """Apply a single adjustment operation to a variable using a dispatch pattern.

    Routes an adjustment instruction to the appropriate handler based on the
    `func` keyword. Some operations are parsed dynamically (e.g. "swap_with_*",
    "delete_when_same_as_*"), while others are looked up in the global
    `HANDLERS` mapping. Handlers modify the dataset in-place and may update
    associated QC flags via `set_flag`.

    Args:
        ds_work (xr.Dataset): Dataset being adjusted (modified in-place).
        var (str): Name of the primary variable to operate on.
        sl (dict): Indexing slice, typically {"time": slice(t0, t1)}.
        func (str): Adjustment function name from the configuration file.
        val (Any): Adjustment parameter value (e.g. offset, factor, threshold).

    Returns:
        xr.Dataset: The same dataset instance, after applying the adjustment.
    """
    # some keywords need to be parsed and are not in the HANDLERS dict
    if func.startswith("swap_with_"):
        logger.debug(f'---> {sl.get("time").start} {sl.get("time").stop} {var} {func} {val}')
        var2 = func[len("swap_with_"):]
        if var2 in ds_work:
            a = ds_work[var].loc[sl].values.copy(deep=True)
            b = ds_work[var2].loc[sl].values.copy(deep=True)
            ds_work[var].loc[sl] = b
            ds_work[var2].loc[sl] = a
        return ds_work

    if func.startswith("delete_when_same_as_"):
        var2 = func[len("delete_when_same_as_"):]
        if var2 in ds_work:
            tmp = ds_work[var].loc[sl]
            msk = np.abs(tmp - ds_work[var2].loc[sl]) < val
            tmp2 = tmp.where(~msk)
            m1 = tmp2.notnull() & tmp2.shift(time=1).isnull() & tmp2.shift(time=-1).isnull()
            m2_first  = tmp2.notnull() & tmp2.shift(time=1).isnull() & tmp2.shift(time=-1).notnull() & tmp2.shift(time=-2).isnull()
            m2_second = tmp2.notnull() & tmp2.shift(time=-1).isnull() & tmp2.shift(time=1).notnull() & tmp2.shift(time=2).isnull()
            bad = msk | m1 | m2_first | m2_second
            bad = msk | m1 | m2_first | m2_second
            ds_work = set_flag(ds_work, var, index_slice=sl, flag="ADJ_SAME_TO_OTHER_SENSOR", mask=bad)
            logger.debug(f'---> {sl.get("time").start} {sl.get("time").stop} {var} {func} {val} flagged {bad.sum().item()}/{len(ds_work.time)}')

        else:
            logger.warning(f"Tried to apply {func} but {var2} not found in data")

        return ds_work

    # if not in the special cases above, then func keyword is linked to its function in HANDLERS
    fn = HANDLERS.get(func)
    if fn is None:
        logger.warning(f"Unknown adjust_function={func} for {var}")
        return ds_work
    ds_work, count = fn(ds_work, var, sl, val)
    if count is None:
        logger.debug(f'---> {sl.get("time").start} {sl.get("time").stop} {var} {func} {val}')
    else:
        logger.debug(f'---> {sl.get("time").start} {sl.get("time").stop} {var} {func} {val} flagged {count}/{len(ds_work.time)}')

    return ds_work



def _getDF(flag_file, var_list):
    '''Get dataframe from flag or adjust file. First attempt to retrieve from
    URL. If this fails then attempt to retrieve from local file

    Parameters
    ----------
    flag_file : str
        Local path to file

    Returns
    -------
    df : pd.DataFrame
        Flag or adjustment dataframe
    '''

    logger.info(f"Using file: {flag_file}")

    if os.path.isfile(flag_file):
        adj_info = pd.read_csv(
                        flag_file,
                        comment="#",
                        skipinitialspace=True,
                        ).dropna(how='all', axis='rows')

        # making sure that t0 and t1 columns are object dtype then replaceing nan with None
        adj_info['t0'] = pd.to_datetime(adj_info['t0'], errors='coerce').dt.tz_localize(None)
        adj_info['t1'] = pd.to_datetime(adj_info['t1'], errors='coerce').dt.tz_localize(None)
        adj_info[['t0','t1']] = adj_info[['t0','t1']].astype(object)

        for t in ['t0','t1']:
            not_a_time = adj_info[t].isnull()
            adj_info.loc[not_a_time, t] = None

        # if "*" is in the variable name then we interpret it as regex
        selec = adj_info['variable'].str.contains(r'\*') & (adj_info['variable'] != "*")
        for ind in adj_info.loc[selec, :].index:
            line_template = adj_info.loc[ind, :].copy(deep=True)
            regex = adj_info.loc[ind, 'variable']
            for var in pd.DataFrame(columns = var_list).filter(regex=regex).columns:
                line_template.variable = var
                line_template.name = adj_info.index.max() + 1
                adj_info = pd.concat((adj_info, line_template.to_frame().transpose()),axis=0)
            adj_info = adj_info.drop(labels=ind, axis=0)

        adj_info = adj_info.sort_values(by=["variable", "t0"])
        adj_info.set_index(["variable", "t0"], drop=False, inplace=True)

    else:
        adj_info=None
        logger.info(f"No {flag_file} file to read.")
    return adj_info


def _hampel(vals_orig, k=7*24, t0=3):
    '''Hampel filter

    Parameters
    ----------
    vals : pd.DataSeries
        Series of values from which to remove outliers
    k : int
        Size of window, including the sample. For example, 7 is equal to 3 on
        either side of value. The default is 7*24.
    t0 : int
        Threshold value. The default is 3.
    '''
    #Make copy so original not edited
    vals=vals_orig.copy(deep=True)

    #Hampel Filter
    L= 1.4826
    rolling_median=vals.rolling(k).median()
    difference=np.abs(rolling_median-vals)
    median_abs_deviation=difference.rolling(k).median()
    threshold= t0 *L * median_abs_deviation
    outlier_idx=difference>threshold
    outlier_idx[0:round(k/2)]=False
    vals.loc[outlier_idx]=np.nan
    return(vals)
