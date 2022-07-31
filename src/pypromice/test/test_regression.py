import numpy as np
import pandas as pd
from promiceAWS import promiceAWS
import os
import pytest

def load_variables_csv():
    data_path = os.path.join('src','promiceAWS','variables.csv')
    print(data_path)
    return pd.read_csv(data_path)

def load_old(filename):
    ## Parse IDL/GDL date time columns
    def mydf(y,m,d,h): return pd.to_datetime(y+'-'+m+'-'+d+':'+h, format='%Y-%m-%d:%H')
    gdl2py_col = load_variables_csv()[['field','IDL']].set_index('IDL').dropna().to_dict()['field']
    gdl = pd.read_csv(filename, delimiter=r"\s+", parse_dates={'time':[0,1,2,3]},
                      infer_datetime_format=True, date_parser=mydf, index_col=0)\
            .apply(pd.to_numeric, errors='coerce')\
            .rename(columns=gdl2py_col)
    return gdl


def process_a_file():
    # pAWS = promiceAWS(config_file='./test_data/conf/'+station+'.toml', inpath='./test_data/input')
    # pAWS.process()
    # pAWS.write(outpath='./test_data/out_python') # Saves L3 data 4x: Daily and hourly in both CSV and NetCDF format
    new = pd.read_csv('./test_data/output_py/KPC_L/KPC_L_hour.csv', index_col=0, parse_dates=True)
    return new

def setup():
    old = load_old('./test_data/output/KPC_L_hour_v03.txt')
    new = process_a_file()
    # subset = np.intersect1d(new.columns, old.columns)
    # print('Common columns: ', sorted(subset), '\n')
    # print('OLD columns dropped:', sorted([_ for _ in old.columns if _ not in subset]), '\n')
    # print('Python columns dropped:', sorted([_ for _ in new.columns if _ not in subset]), '\n')
    # new = new[subset]
    # old = old[subset]
    return old, new


def test_percentage_difference_under_1_percent():
    old, new = setup()

    # import pdb; pdb.set_trace()
    err = new - old.replace(-999,np.nan) # abs error
    err_pct = (err / old.replace(-999,np.nan).mean(axis='rows'))*100 # % err; should work as long as mean != 0

    for var in ['fan_dc','gps_alt','p','qh','rh_cor',
                't_1','t_2','t_i_1','t_i_8',
                't_log','usr','wdir','wspd',
                'z_boom','z_pt','z_pt_cor','z_stake']:
        print(var)
        arr = err_pct[var]
        assert(np.all(abs(arr.dropna()) < 1))

