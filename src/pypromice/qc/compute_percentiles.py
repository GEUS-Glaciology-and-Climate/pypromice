#!/usr/bin/env python
"""
Compute percentile distributions and write to local sqlite db
"""
import sqlite3
import os
import pandas as pd
from datetime import timedelta
import argparse
# from IPython import embed

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--l3-filepath',
        default='../../../../aws-l3/level_3/', # relative to qc dir
        # default='/data/pypromice_aws/aws-l3/level_3/' # full
        type=str,
        required=False,
        help='Path to read level 3 csv files.')

    args = parser.parse_args()
    return args

def make_db_connection():
    print('Creating sqlite3 connection...') 
    con = sqlite3.connect( # will create db if does not exist
    'percentiles.db', # write to on-disk file at current directory location
    isolation_level=None # autocommit mode
    )

    print('Creating sqlite3 cursor...') 
    cur = con.cursor()
    return con, cur

def create_tables(cur, var_list):
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
                    # 'p25 float, p33 float, p50 float, p66 float, p75 float, '
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
    '''
    for v in var_list:
        cur.execute(f'delete from {v}')
        print(f'Deleted {cur.rowcount} records from the {v} table.')

def write_percentiles(cur, var_list):
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
                        exe_list = [stid] # initialize list
                        for i in quantiles:
                            exe_list.append(df[f'{v}'].quantile(q=i))
                        exe_list.append(years)
                        # exe_list.insert(0,stid)
                        cur.execute(
                            f'insert into {v} '
                            '(stid,p0,p0p5,p1,p5,p10,p90,p95,p99,p99p5,p100,years) '
                            'values (?,?,?,?,?,?,?,?,?,?,?,?)',
                            exe_list
                            )
                    elif v=='t_u':
                        df.set_index(timestamp, inplace=True)
                        # data.drop(['time'], axis=1, inplace=True) # drop original time column
                        winter = df.t_u[(df.index.month >= 1) & (df.index.month <= 3)]
                        spring = df.t_u[(df.index.month >= 4) & (df.index.month <= 6)]
                        summer = df.t_u[(df.index.month >= 7) & (df.index.month <= 9)]
                        fall = df.t_u[(df.index.month >= 10) & (df.index.month <= 12)]

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

if __name__ == '__main__':
    """Executed from the command line"""
    args = parse_arguments()

    var_list = ['t_u','rh_u','p_u','wspd_u'] # one table per var
    disclude_stations = ('XXX',)

    con, cur = make_db_connection()
    create_tables(cur,var_list)
    clear_tables(cur,var_list)
    write_percentiles(cur,var_list)
    # embed()