#!/usr/bin/env python
"""
Compute percentile distributions and write to local sqlite db
"""
import sqlite3
import os
import pandas as pd
import numpy as np
from datetime import timedelta
import argparse
# from IPython import embed

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--l3-filepath',
        #default='../../../../aws-l3/level_3/', # relative path to qc dir
        default ='/data/geusgk/awsl3-fileshare/aws-l3/level_3', # full path
        type=str,
        required=False,
        help='Path to read level 3 csv files.')

    args = parser.parse_args()
    return args

def make_db_connection():
    '''
    Make connection to on-disk sqlite database

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''
    print('Creating sqlite3 connection...') 
    con = sqlite3.connect( # will create db if does not exist
    'percentiles.db', # define path and filename
    isolation_level = None # autocommit mode
    )

    print('Creating sqlite3 cursor...') 
    cur = con.cursor()
    return con, cur

def create_tables(cur, var_list):
    '''
    Create variable-specific tables in the sqlite database

    Parameters
    ----------
    cur : sqlite3.Cursor
        cursor on the sqlite database connection
    var_list : list
        list of variable strings (e.g. 't_u') used to make percentiles

    Returns
    -------
    None
    '''
    for v in var_list:
        print(f'Creating {v} table...')
        if v not in ('t_u',):
            # No seasonality, create "flat" percentile distribution, single PRIMARY key
            try:
                cur.execute(
                    f'create table {v}('
                    'stid text PRIMARY KEY, '
                    'p0 float, p0p5 float, p1 float, p5 float, p10 float, '
                    # 'p25 float, p33 float, p50 float, p66 float, p75 float, '
                    'p90 float, p95 float, p99 float, p99p5 float, p100 float, '
                    'years smallint)'
                )
                # cur.execute("create unique index percid on percentiles (stid,season)")
            except sqlite3.OperationalError:
                print(f"----> {v} table already exists!")
        elif v=='t_u':
            # Here we add the 'season' column, and have a compound PRIMARY key
            # TODO: difference between using index or compound PRIMARY?
            try:
                cur.execute(
                    f'create table {v}('
                    'stid text, '
                    'season smallint, ' # only used for airtemp
                    'p0 float, p0p5 float, p1 float, p5 float, p10 float, '
                    # 'p25 float, p33 float, p50 float, p66 float, p75 float, ' # optional add'l percentiles
                    'p90 float, p95 float, p99 float, p99p5 float, p100 float, '
                    'years smallint, '
                    'PRIMARY KEY (stid, season))'
                )
                # cur.execute("create unique index percid on percentiles (stid,season)")
            except sqlite3.OperationalError:
                print(f"----> {v} table already exists!")

def clear_tables(cur, var_list):
    '''
    Clear all rows from all tables. We run this by default, assuming that anytime
    we are running this script we intend to overwrite all rows for all tables.

    Parameters
    ----------
    cur : sqlite3.Cursor
        cursor on the sqlite database connection
    var_list : list
        list of variable strings (e.g. 't_u') used to make percentiles

    Returns
    -------
    None
    '''
    for v in var_list:
        cur.execute(f'delete from {v}')
        print(f'Deleted {cur.rowcount} records from the {v} table.')

def write_percentiles(cur, var_list):
    '''
    Write percentile data to tables

    Parameters
    ----------
    cur : sqlite3.Cursor
        cursor on the sqlite database connection
    var_list : list
        list of variable strings (e.g. 't_u') used to make percentiles

    Returns
    -------
    None
    '''
    print(f'writing to tables...')
    for x in os.walk(args.l3_filepath):
        if (len(x[2]) > 0): # files are present
            stid = x[0].split('/')[-1]
            csv_file = [s for s in x[2] if '_hour.csv' in s]
            if (len(csv_file) > 0) and (stid not in disclude_stations): # csv file is present
                print(stid)
                csv_filepath = x[0] + '/' + csv_file[0]
                df = pd.read_csv(csv_filepath)
                timestamp = pd.to_datetime(df.time)
                years = round((timestamp.max()-timestamp.min()) / timedelta(days=365.25))
                quantiles = [0,0.005,0.01,0.05,0.10,0.90,0.95,0.99,0.995,1]
                for v in var_list:
                    if v not in ('t_u',):
                        exe_list = [stid] # initialize list with stid
                        for i in quantiles:
                            exe_list.append(df[f'{v}'].quantile(q=i)) # percentiles calculated here!
                        exe_list.append(years)
                        cur.execute(
                            f'insert into {v} '
                            '(stid,p0,p0p5,p1,p5,p10,p90,p95,p99,p99p5,p100,years) '
                            'values (?,?,?,?,?,?,?,?,?,?,?,?)',
                            exe_list
                            )
                    elif v=='t_u':
                        df.set_index(timestamp, inplace=True)
                        # data.drop(['time'], axis=1, inplace=True) # optionally drop original time column

                        winter = df.t_u[df.index.month.isin([12,1,2])]
                        spring = df.t_u[df.index.month.isin([3,4,5])]
                        summer = df.t_u[df.index.month.isin([6,7,8])]
                        fall = df.t_u[df.index.month.isin([9,10,11])]

                        # Equivalent to above
                        # winter = df.t_u[df.index.month.isin([12,1,2])]
                        # spring = df.t_u[(df.index.month >= 3) & (df.index.month <= 5)]
                        # summer = df.t_u[(df.index.month >= 6) & (df.index.month <= 8)]
                        # fall = df.t_u[(df.index.month >= 9) & (df.index.month <= 11)]

                        season_list = [winter,spring,summer,fall]
                        season_integers = [1,2,3,4]

                        for s, s_i in zip(season_list,season_integers):
                            exe_list = [stid,s_i] # initialize list
                            for i in quantiles:
                                exe_list.append(s.quantile(q=i))
                            exe_list.append(years)
                            cur.execute(
                                f'insert into {v} '
                                '(stid,season,p0,p0p5,p1,p5,p10,p90,p95,p99,p99p5,p100,years) '
                                'values (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                exe_list
                                )

def _analyze_percentiles():
    '''
    This is run ONLY to examine percentile thresholds with context to full datasets
    '''
    for x in os.walk(args.l3_filepath):
        if (len(x[2]) > 0): # files are present
            stid = x[0].split('/')[-1]
            csv_file = [s for s in x[2] if '_hour.csv' in s]
            if (len(csv_file) > 0) and (stid not in disclude_stations): # csv file is present
                print(stid)
                csv_filepath = x[0] + '/' + csv_file[0]
                df = pd.read_csv(csv_filepath)
                timestamp = pd.to_datetime(df.time)
                df.set_index(timestamp, inplace=True)
                _percentileQC(df, stid)

def _percentileQC(df, stid):
    '''
    This is the same function that is found in L1toL2.py
    Once thresholds are determined, they can be transferred to L1toL2.py
    '''
    # Define threshold dict to hold limit values, and 'hi' and 'lo' percentile.
    # Limit values indicate how far we will go beyond the hi and lo percentiles to flag outliers.
    # *_u are used to calculate and define all limits, which are then applied to *_u, *_l and *_i
    var_threshold = {
        't_u': {'limit': 9}, # 'hi' and 'lo' will be held in 'seasons' dict
        'p_u': {'limit': 15},
        'rh_u': {'limit': 12},
        'wspd_u': {'limit': 10}
        }

    # Query from the on-disk sqlite db for specified percentiles
    con = sqlite3.connect('percentiles.db')
    cur = con.cursor()
    for k in var_threshold.keys():
        if k == 't_u':
            # Different pattern for t_u, which considers seasons
            # 1: winter (DecJanFeb), 2: spring (MarAprMay), 3: summer (JunJulAug), 4: fall (SepOctNov)
            seasons = {1: {}, 2: {}, 3: {}, 4: {}}
            sql = f"SELECT p0p5,p99p5,season FROM {k} WHERE season in (1,2,3,4) and stid = ?"
            cur.execute(sql, [stid])
            result = cur.fetchall()
            for row in result:
                # row[0] is p0p5, row[1] is p99p5, row[2] is the season integer
                seasons[row[2]]['lo'] = row[0] # 0.005
                seasons[row[2]]['hi'] = row[1] # 0.995
                var_threshold[k]['seasons'] = seasons
        else:
            sql = f"SELECT p0p5,p99p5 FROM {k} WHERE stid = ?"
            cur.execute(sql, [stid])
            result = cur.fetchone() # we only expect one row back per station
            var_threshold[k]['lo'] = result[0] # 0.005
            var_threshold[k]['hi'] = result[1] # 0.995

    con.close() # close the database connection (and cursor)

    # Set flagged data to NaN
    for k in var_threshold.keys():
        if k == 't_u':
            # use t_u thresholds to flag t_u, t_l, t_i
            base_var = k.split('_')[0]
            vars_all = [k, base_var+'_l', base_var+'_i']
            for t in vars_all:
                if t in df:
                    winter = df[t][df.index.month.isin([12,1,2])]
                    spring = df[t][df.index.month.isin([3,4,5])]
                    summer = df[t][df.index.month.isin([6,7,8])]
                    fall = df[t][df.index.month.isin([9,10,11])]
                    season_dfs = [winter,spring,summer,fall]

                    _plot_percentiles_t(k,t,df,season_dfs,var_threshold,stid) # BEFORE OUTLIER REMOVAL
                    for x1,x2 in zip([1,2,3,4], season_dfs):
                        print(f'percentile flagging {t} {x1}')
                        lower_thresh = var_threshold[k]['seasons'][x1]['lo'] - var_threshold[k]['limit']
                        upper_thresh = var_threshold[k]['seasons'][x1]['hi'] + var_threshold[k]['limit']
                        outliers_upper = x2[x2.values > upper_thresh]
                        outliers_lower = x2[x2.values < lower_thresh]
                        outliers = pd.concat([outliers_upper,outliers_lower])
                        df.loc[outliers.index,t] = np.nan
                        df.loc[outliers.index,t] = np.nan

                    # _plot_percentiles_t(k,t,df,season_dfs,var_threshold,stid) # AFTER OUTLIER REMOVAL
        else:
            # use *_u thresholds to flag *_u, *_l, *_i
            base_var = k.split('_')[0]
            vars_all = [k, base_var+'_l', base_var+'_i']
            for t in vars_all:
                if t in df:
                    print(f'percentile flagging {t}')
                    upper_thresh = var_threshold[k]['hi'] + var_threshold[k]['limit']
                    lower_thresh = var_threshold[k]['lo'] - var_threshold[k]['limit']
                    _plot_percentiles(k,t,df,var_threshold,upper_thresh,lower_thresh,stid) # BEFORE OUTLIER REMOVAL
                    if t == 'p_i':
                        # shift p_i so we can use the p_u thresholds
                        shift_p = df[t]+1000.
                        outliers_upper = shift_p[shift_p.values > upper_thresh]
                        outliers_lower = shift_p[shift_p.values < lower_thresh]
                    else:
                        outliers_upper = df[t][df[t].values > upper_thresh]
                        outliers_lower = df[t][df[t].values < lower_thresh]
                    outliers = pd.concat([outliers_upper,outliers_lower])
                    df.loc[outliers.index,t] = np.nan
                    df.loc[outliers.index,t] = np.nan

                    # _plot_percentiles(k,t,df,var_threshold,upper_thresh,lower_thresh,stid) # AFTER OUTLIER REMOVAL
    return None

def _plot_percentiles_t(k, t, df, season_dfs, var_threshold, stid):
    '''Plot data and percentile thresholds for air temp (seasonal)'''
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,12))
    inst_var = t.split('_')[0] + '_i'
    if inst_var in df:
        i_plot = df[inst_var]
        plt.scatter(df.index,i_plot, color='orange', s=3, label='t_i instantaneuous')
    if t in ('t_u','t_l'):
        plt.scatter(df.index,df[t], color='b', s=3, label=f'{t} hourly ave')
    for x1,x2 in zip([1,2,3,4], season_dfs):
        y1 = np.full(len(x2.index), (var_threshold[k]['seasons'][x1]['lo'] - var_threshold[k]['limit']))
        y2 = np.full(len(x2.index), (var_threshold[k]['seasons'][x1]['hi'] + var_threshold[k]['limit']))
        y11 = np.full(len(x2.index), (var_threshold[k]['seasons'][x1]['lo'] ))
        y22 = np.full(len(x2.index), (var_threshold[k]['seasons'][x1]['hi'] ))
        plt.scatter(x2.index, y1, color='r',s=1)
        plt.scatter(x2.index, y2, color='r', s=1)
        plt.scatter(x2.index, y11, color='k', s=1)
        plt.scatter(x2.index, y22, color='k', s=1)
    plt.title('{} {}'.format(stid, t))
    plt.legend(loc="lower left")
    plt.show()

def _plot_percentiles(k, t, df, var_threshold, upper_thresh, lower_thresh, stid):
    '''Plot data and percentile thresholds'''
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,12))
    inst_var = t.split('_')[0] + '_i'
    if inst_var in df:
        if k == 'p_u':
            i_plot = (df[inst_var]+1000.)
        else:
            i_plot = df[inst_var]
        plt.scatter(df.index,i_plot, color='orange', s=3, label='instantaneuous')
    if t != inst_var:
        plt.scatter(df.index,df[t], color='b', s=3, label=f' {t} hourly ave')
    plt.axhline(y=upper_thresh, color='r', linestyle='-')
    plt.axhline(y=lower_thresh, color='r', linestyle='-')
    plt.axhline(y=var_threshold[k]['hi'], color='k', linestyle='--')
    plt.axhline(y=var_threshold[k]['lo'], color='k', linestyle='--')
    plt.title('{} {}'.format(stid, t))
    plt.legend(loc="lower left")
    plt.show()

if __name__ == '__main__':
    """Executed from the command line"""
    args = parse_arguments()

    var_list = ['t_u','rh_u','p_u','wspd_u'] # one table per var
    disclude_stations = ('XXX',)

    # THE FOLLOWING WILL WRITE A NEW SQLITE DB
    # Intended to be run on an (e.g.) monthly cron
    # Turn this off if you want to only run _analyze_percentiles()
    # ========================================
    con, cur = make_db_connection()
    create_tables(cur,var_list)
    clear_tables(cur,var_list)
    write_percentiles(cur,var_list)
    # ========================================

    # Turn this on to make full station plots
    # _analyze_percentiles()