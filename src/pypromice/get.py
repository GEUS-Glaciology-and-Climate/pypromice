#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AWS data retrieval module
"""
from pyDataverse.api import NativeApi
import pandas as pd
import xarray as xr
import unittest, pkg_resources
from datetime import datetime
        

def lookup_table(base_dois,
                 server='https://dataverse.geus.dk'):
    # Prime API
    dataverse_server = server.strip("/")
    api = NativeApi(dataverse_server)
    
    # Look through DOI entries
    lookup_list = {}
    for d in base_dois:
        dataset = api.get_dataset(d)
        
        # Get file names and DOIs
        f_list = dataset.json()['data']['latestVersion']['files']
        for f in f_list:
            fname = f['dataFile']['filename'].lower()
            if '.csv' in fname or '.nc' in fname or '.txt' in fname:
                link = _getURL(f['dataFile']['persistentId'])
                lookup_list[fname] = link
    return lookup_list       
   
def aws_names():
    '''Return PROMICE and GC-Net AWS names that can be used in get.aws_data() 
    fetching'''
    lookup = lookup_table(['doi:10.22008/FK2/IW73UU', 'doi:10.22008/FK2/GNYFUK'])
    print(f'Available dataset keywords: {list(lookup.keys())}')
    return list(lookup.keys())

def aws_data(aws_name):
    '''Return PROMICE and GC-Net AWS L3 v3 hourly observations
    
    Returns
    -------
    df : pandas.DataFrame
        AWS observations dataframe
    '''
    lookup = lookup_table(['doi:10.22008/FK2/IW73UU', 'doi:10.22008/FK2/GNYFUK'])
    assert aws_name.lower() in list(lookup.keys())
    if '.csv' in aws_name.lower():
        data = pd.read_csv(lookup[aws_name], index_col=0, parse_dates=True)
    elif '.nc' in aws_name.lower():
        data = xr.open_dataset(lookup[aws_name])
    return data        
  
def watson_discharge(t='hour'):
    '''Return PROMICE hourly Watson river discharge
    
    Parameters
    ----------
    t : str
        Temporal resolution of the data - "hour", "day" or "month"

    Returns
    -------
    df : pandas.DataFrame
        Watson river discharge dataframe    
    '''
    lookup = lookup_table(['doi:10.22008/FK2/XEHYCM'])
    if 'month' in t.lower():
        link = lookup['watson river discharge (2006-2021) monthly.txt']
    elif 'daily' in t.lower() or 'day' in t.lower():
        link = lookup['watson river discharge (2006-2021) daily.txt']     
    else:
        link = lookup['watson river discharge (2006-2021) hourly.txt']  
        
    df = pd.read_csv(link, sep="\s+", parse_dates=[[0,1,2,3]])\
                    .rename({"WaterFluxDiversOnly(m3/s)"         : "divers",
                            "Uncertainty(m3/s)"                 : "divers_err",
                            "WaterFluxDivers&Temperature(m3/s)" : "divers_t",
                            "Uncertainty(m3/s).1"               : "divers_t_err",
                            "WaterFluxCumulative(km3)"          : "Q",
                            "Uncertainty(km3)"                  : "err"}, 
                            axis='columns')
    df = _getDFdatetime(df, list(df.iloc[:,0]))
    return df

def _getURL(persistentId, 
            base_link='https://dataverse.geus.dk/api/access/datafile/:persistentId?persistentId='):
    '''Return URL download link from persistentId attached to DOI'''
    return base_link+persistentId


def _getDFdatetime(df, dt_str, dt_format='%Y %m %d %H'):
    '''Format dataframe with datetime (year, month, day, hour) index column
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    dt_str : list
        List of datetime strings to format and add
    dt_format : str
        Datetime string format. Default is "%Y %m %d %H".
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with added datetime as index
    '''
    dates = [datetime.strptime(str(d), '%Y %m %d %H') for d in dt_str]
    df['Datetime'] = dates
    df = df.set_index('Datetime')
    df.drop(columns=df.columns[0], axis=1, inplace=True)  
    return df

#------------------------------------------------------------------------------
        
class TestGet(unittest.TestCase): 
    def testURL(self):
        '''Test URL retrieval'''
        l = lookup_table(['doi:10.22008/FK2/IW73UU', 'doi:10.22008/FK2/GNYFUK'])
        self.assertTrue('10.22008/FK2' in list(l.values())[0])
    
    def testAWSname(self):  
        '''Test AWS names retrieval'''
        n = aws_names()
        self.assertIsInstance(n, list)
        self.assertTrue('nuk_k_hour.csv' in n)
    
    def testAWScsv(self):
        '''Test AWS data retrieval'''
        kan_b = aws_data('kan_b_hour.csv')
        self.assertIsInstance(kan_b, pd.DataFrame)

    # def testAWSnc(self):
    #     '''Test AWS data retrieval'''
    #     kan_b = aws_data('nuk_k_hour.nc')
    #     self.assertIsInstance(kan_b, xr.DataArray)
        
    def testWatsonHour(self):
        '''Test Wason River discharge hourly data retrieval'''
        wh = watson_discharge()
        self.assertTrue(wh['Q']['2021-10-27 23:00:00']==5.48)
        
    # def testWatsonDaily(self):
    #     '''Test Wason River discharge daily data retrieval'''
    #     wd = watson_discharge(t='day')
    #     print(wd)
        # self.assertTrue(wd['Q']['2021-10-27']==5.48)
            
if __name__ == "__main__": 
    unittest.main()
