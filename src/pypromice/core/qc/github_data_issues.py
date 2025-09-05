import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    'flagNAN',
    'adjustTime',
    'adjustData',
]

logger = logging.getLogger(__name__)


def flagNAN(ds_in, flag_dir):
    '''Read flagged data from .csv file. For each variable, and downstream
    dependents, flag as invalid (or other) if set in the flag .csv

    Parameters
    ----------
    ds_in : xr.Dataset
        Level 0 dataset
    flag_dir : str
        File directory where .csv flag files can be found

    Returns
    -------
    ds : xr.Dataset
        Level 0 data with flagged data
    '''
    ds = ds_in.copy(deep=True)
    df = None

    df = _getDF(os.path.join(flag_dir, ds.attrs["station_id"] + ".csv"))

    if isinstance(df, pd.DataFrame):
        df.t0 = pd.to_datetime(df.t0).dt.tz_localize(None)
        df.t1 = pd.to_datetime(df.t1).dt.tz_localize(None)

        if df.shape[0] > 0:
            for i in df.index:
                t0, t1, avar = df.loc[i,['t0','t1','variable']]

                if avar == '*':
                    # Set to all vars if var is "*"
                    varlist = list(ds.keys())
                elif '*' in avar:
                    # Reads as regex if contains "*" and other characters (e.g. 't_i_.*($)')
                    varlist = pd.DataFrame(columns = list(ds.keys())).filter(regex=(avar)).columns
                else:
                    varlist = avar.split()

                if 'time' in varlist: varlist.remove("time")

                # Set to all times if times are "n/a"
                if pd.isnull(t0):
                    t0 = ds['time'].values[0]
                if pd.isnull(t1):
                    t1 = ds['time'].values[-1]

                for v in varlist:
                    if v in list(ds.keys()):
                        logger.debug(f'---> flagging {t0} {t1} {v}')
                        ds[v] = ds[v].where((ds['time'] < t0) | (ds['time'] > t1))
                    else:
                        logger.debug(f'---> could not flag {v} not in dataset')

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

    adj_info = _getDF(os.path.join(adj_dir, ds.attrs["station_id"] + ".csv"))

    if isinstance(adj_info, pd.DataFrame):


        if "time_shift" in adj_info.adjust_function.values:
            time_shifts = adj_info.loc[adj_info.adjust_function == "time_shift", :]
            # if t1 is left empty, then adjustment is applied until the end of the file
            time_shifts.loc[time_shifts.t1.isnull(), "t1"] = pd.to_datetime(ds_out.time.values[-1]).isoformat()
            time_shifts.t0 = pd.to_datetime(time_shifts.t0).dt.tz_localize(None)
            time_shifts.t1 = pd.to_datetime(time_shifts.t1).dt.tz_localize(None)

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


def adjustData(ds, adj_dir, var_list=[], skip_var=[]):
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
    ds_out = ds.copy(deep=True)
    adj_info=None
    adj_info = _getDF(os.path.join(adj_dir, ds.attrs["station_id"] + ".csv"))

    if isinstance(adj_info, pd.DataFrame):
        # removing potential time shifts from the adjustment list
        adj_info = adj_info.loc[adj_info.adjust_function != "time_shift", :]

        # making sure that t0 and t1 columns are object dtype then replaceing nan with None
        adj_info[['t0','t1']] = adj_info[['t0','t1']].astype(object)
        adj_info.loc[adj_info.t1.isnull()|(adj_info.t1==''), "t1"] = None      
        adj_info.loc[adj_info.t0.isnull()|(adj_info.t0==''), "t0"] = None
		
        # if "*" is in the variable name then we interpret it as regex
        selec =  adj_info['variable'].str.contains('\*') & (adj_info['variable'] != "*")
        for ind in adj_info.loc[selec, :].index:
            line_template = adj_info.loc[ind, :].copy()
            regex = adj_info.loc[ind, 'variable']
            for var in pd.DataFrame(columns = list(ds.keys())).filter(regex=regex).columns:
                line_template.variable = var
                line_template.name = adj_info.index.max() + 1
                adj_info = pd.concat((adj_info, line_template.to_frame().transpose()),axis=0)
            adj_info = adj_info.drop(labels=ind, axis=0)

        adj_info = adj_info.sort_values(by=["variable", "t0"])
        adj_info.set_index(["variable", "t0"], drop=False, inplace=True)

        if len(var_list) == 0:
            var_list = np.unique(adj_info.variable)
        else:
            adj_info = adj_info.loc[np.isin(adj_info.variable, var_list), :]
            var_list = np.unique(adj_info.variable)

        if len(skip_var) > 0:
            adj_info = adj_info.loc[~np.isin(adj_info.variable, skip_var), :]
            var_list = np.unique(adj_info.variable)

        for var in var_list:
            if var not in list(ds_out.keys()):
                logger.info(f'could not adjust {var } not in dataset')
                continue
            for t0, t1, func, val in zip(
                adj_info.loc[var].t0,
                adj_info.loc[var].t1,
                adj_info.loc[var].adjust_function,
                adj_info.loc[var].adjust_value,
            ):
                # making all timestamps timezone naive (compatibility with xarray)
                if isinstance(t0, str):
                    t0 = pd.to_datetime(t0, utc=True).tz_localize(None)
                if isinstance(t1, str):
                    t1 = pd.to_datetime(t1, utc=True).tz_localize(None)

                index_slice = dict(time=slice(t0, t1))
                if len(ds_out[var].loc[index_slice].time.time) == 0:
                    logger.info(f'---> {t0} {t1} {var} {func} {val}')
                    logger.info("Time range does not intersect with dataset")
                    continue

                else:
                    logger.debug(f'---> {t0} {t1} {var} {func} {val}')

                if func == "add":
                    ds_out[var].loc[index_slice] = ds_out[var].loc[index_slice].values + val
                    # flagging adjusted values
                    # if var + "_adj_flag" not in ds_out.columns:
                    #     ds_out[var + "_adj_flag"] = 0
                    # msk = ds_out[var].loc[index_slice])].notnull()
                    # ind = ds_out[var].loc[index_slice])].loc[msk].time
                    # ds_out.loc[ind, var + "_adj_flag"] = 1

                if func == "multiply":
                    ds_out[var].loc[index_slice] = ds_out[var].loc[index_slice].values * val
                    if "DW" in var:
                        ds_out[var].loc[index_slice] = ds_out[var].loc[index_slice] % 360
                    # flagging adjusted values
                    # if var + "_adj_flag" not in ds_out.columns:
                    #     ds_out[var + "_adj_flag"] = 0
                    # msk = ds_out[var].loc[index_slice].notnull()
                    # ind = ds_out[var].loc[index_slice].loc[msk].time
                    # ds_out.loc[ind, var + "_adj_flag"] = 1

                if func == "min_filter":
                    tmp = ds_out[var].loc[index_slice].values
                    tmp[tmp < val] = np.nan

                if func == "max_filter":
                    tmp = ds_out[var].loc[index_slice].values
                    tmp[tmp > val] = np.nan
                    ds_out[var].loc[index_slice] = tmp

                if func == "upper_perc_filter":
                    tmp = ds_out[var].loc[index_slice].copy()
                    df_w = ds_out[var].loc[index_slice].resample(time="14D").quantile(1 - val / 100)
                    df_w = ds_out[var].loc[index_slice].resample(time="14D").var()
                    for m_start, m_end in zip(df_w.time[:-2], df_w.time[1:]):
                        msk = (tmp.time >= m_start) & (tmp.time < m_end)
                        values_month = tmp.loc[msk].values
                        values_month[values_month < df_w.loc[m_start]] = np.nan
                        tmp.loc[msk] = values_month

                    ds_out[var].loc[index_slice] = tmp.values

                if func == "biweekly_upper_range_filter":
                    df_max = (
                        ds_out[var].loc[index_slice].copy(deep=True)
                        .resample(time="14D",offset='7D').max()
                        .sel(time=ds_out[var].loc[index_slice].time.values, method='ffill')
                        )
                    df_max['time'] = ds_out[var].loc[index_slice].time
                    # updating original pandas                   
                    ds_out[var].loc[index_slice] = ds_out[var].loc[index_slice].where(ds_out[var].loc[index_slice] > df_max-val)
                                        

                if func == "hampel_filter":
                    tmp = ds_out[var].loc[index_slice]
                    tmp = _hampel(tmp, k=7 * 24, t0=val)
                    ds_out[var].loc[index_slice] = tmp.values

                if func == "grad_filter":
                    tmp = ds_out[var].loc[index_slice].copy()
                    msk = ds_out[var].loc[index_slice].copy().diff()
                    tmp[np.roll(msk.abs() > val, -1)] = np.nan
                    ds_out[var].loc[index_slice] = tmp

                if "swap_with_" in func:
                    var2 = func[10:]
                    val_var = ds_out[var].loc[index_slice].values.copy()
                    val_var2 = ds_out[var2].loc[index_slice].values.copy()
                    ds_out[var2].loc[index_slice] = val_var
                    ds_out[var].loc[index_slice] = val_var2

                if func == "rotate":
                    ds_out[var].loc[index_slice] = (ds_out[var].loc[index_slice].values + val) % 360

    return ds_out


def _getDF(flag_file):
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
        df = pd.read_csv(
                        flag_file,
                        comment="#",
                        skipinitialspace=True,
                        ).dropna(how='all', axis='rows')
    else:
        df=None
        logger.info(f"No {flag_file} file to read.")
    return df


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
    vals=vals_orig.copy()

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
