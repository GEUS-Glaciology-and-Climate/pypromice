#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pypromice data retrieval module
"""
import pandas as pd
import unittest
from datetime import datetime

        
def aws_names(url_index='data_urls.csv'):
    '''Return PROMICE AWS names that can be used in get.aws_data() fetching'''
    with open(url_index, 'r') as f:
        lines = f.readlines()
    names = [l.split(',')[0] for l in lines]
    print(f'Available dataset keywords: {names}')
    return names    
    
def aws_data(aws_name):                                                        #TODO add daily and monthly datasets
    '''Return PROMICE AWS L3 v3 hourly observations
    
    Returns
    -------
    df : pandas.DataFrame
        AWS observations dataframe
    '''
    URL = _getURL(aws_name.lower()+'_hour', url_index='data_urls.csv')
    df = pd.read_csv(URL, delimiter='\s+', parse_dates=[[0,1,2,3]], header=0)
    df = _getDFdatetime(df, list(df.iloc[:,0]))
    return df
    
def watson_discharge_hourly():
    '''Return PROMICE hourly Watson river discharge
    
    Returns
    -------
    df : pandas.DataFrame
        Watson river discharge dataframe    
    '''
    URL = _getURL('watson_discharge_hourly', 'data_urls.csv')
    df = pd.read_csv(URL, sep="\s+", parse_dates=[[0,1,2,3]])\
            .rename({"WaterFluxDiversOnly(m3/s)"         : "divers",
                    "Uncertainty(m3/s)"                 : "divers_err",
                    "WaterFluxDivers&Temperature(m3/s)" : "divers_t",
                    "Uncertainty(m3/s).1"               : "divers_t_err",
                    "WaterFluxCumulative(km3)"          : "Q",
                    "Uncertainty(km3)"                  : "err"}, 
                    axis='columns')
    df = _getDFdatetime(df, list(df.iloc[:,0]))
    return df

def watson_discharge_daily():
    '''Return PROMICE daily Watson river discharge
    
    Returns
    -------
    df : pandas.DataFrame        # self.assertEquals(a, 36820)
        # self.assertTrue(e.imei in '300234061165160')
        # self.assertFalse(e.mtmsn)
        Watson river discharge dataframe    
    '''
    URL = _getURL('watson_discharge_daily', 'data_urls.csv')
    df = pd.read_csv(URL, sep="\s+", parse_dates=[[0,1,2]], index_col=0)\
            .rename({"WaterFluxDiversOnly(m3/s)"         : "divers",
                    "Uncertainty(m3/s)"                 : "divers_err",
                    "WaterFluxDivers&Temperature(m3/s)" : "divers_t",
                    "Uncertainty(m3/s).1"               : "divers_t_err",
                    "WaterFluxCumulative(km3)"          : "Q",
                    "Uncertainty(km3)"                  : "err"}, 
                    axis='columns')
    df.index.name = "Date"
    return df   

# def basal_melt():                                                              #TODO fix this
#     '''Return PROMICE basal melt data
#     )
#     Returns
#     -------
#     df : pandas.DataFrame
#         Watson river discharge dataframe    
#     '''    
#     URL1 = _getURL('basal_melt', 'data_urls.csv')
#     URL2 = _getURL('heat_sources', 'data_urls.csv')

#     with fsspec.open(URL1) as fobj:
#         basal = xr.open_dataset(fobj)
#     with fsspec.open(URL2) as fobj:
#         heat = xr.open_dataset(URL2)
        
#     return basal, heat

# # Use xarray with HTTP: https://github.com/pydata/xarray/issues/3653#issuecomment-570163736
# def ice_discharge(resolution="GIS"):
#     res_file = {"gate": "https://dataverse01.geus.dk/api/access/datafile/:persistentId?persistentId=doi:10.22008/promice/data/ice_discharge/d/v02/IRLTR2",
#                 "sector": "https://dataverse01.geus.dk/api/access/datafile/:persistentId?persistentId=doi:10.22008/promice/data/ice_discharge/d/v02/UXWVIF",
#                 "region": "https://dataverse01.geus.dk/api/access/datafile/:persistentId?persistentId=doi:10.22008/promice/data/ice_discharge/d/v02/B3PQEH",
#                 "GIS": "https://dataverse01.geus.dk/api/access/datafile/:persistentId?persistentId=doi:10.22008/promice/data/ice_discharge/d/v02/ANRF6L"}
    
#     assert(resolution in res_file.keys())
#     URL = res_file[resolution]
#     print(URL)
#     # download_warn(URL)
#     with fsspec.open(URL) as fobj:
#         ds = xr.open_dataset(fobj)

#     return ds
    
# def gates():
#     '''Return PROMICE ice discharge gates
    
#     Returns
#     -------
#     df : pandas.DataFrame
#         Ice discharge gates dataframe    
#     '''
#     URL = getURL('ice_discharge_gates')
#     df = pd.read_csv(URL, delimiter="\t", index_col=0, parse_dates=[[0,1,2]])
#     return df

# def ice_extent():
#     '''Return PROMICE ice extent
    
#     Returns
#     -------)
#     df : geopandas.GeoDataFrame
#         Ice extent geodataframe    
#     '''
#     URL = getURL('ice_extent')
#     df = gpd.read_file(URL)
#     return df

def _getURL(name, url_index, delimiter=','):
    '''Get Dataset URL from index file
    
    Parameters
    ----------
    name : str
        Dataset name
    url_index : str
        URL index .csv file
    delimiter : str
        String delimiter. Default is ","
    '''
    url = None
    with open(url_index, 'r') as f:
        lines = f.readlines()
    for l in lines:
        if name in l:
            url = l.split(',')[-1]
    return url

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
        u = _getURL('watson_discharge_hourly', 'data_urls.csv')
        self.assertTrue('doi:10.22008/FK2/XEHYCM/ODSEOV' in u)
    
    def testAWSname(self):  
        '''Test AWS names retrieval'''
        n = aws_names()
        self.assertIsInstance(n, list)
        self.assertTrue('kan_b_hour' in n)
    
    def testAWSdata(self):
        '''Test AWS data retrieval'''
        kan_b = aws_data('KAN_B')
        self.assertIsInstance(kan_b, pd.DataFrame)
    
    def testWatsonHour(self):
        '''Test Wason River discharge hourly data retrieval'''
        wh = watson_discharge_hourly()
        self.assertTrue(wh['Q']['2021-10-27 23:00:00']==5.48)
        
    def testWatsonDaily(self):
        '''Test Wason River discharge daily data retrieval'''
        wd = watson_discharge_daily()
        self.assertTrue(wd['Q']['2021-10-27']==5.48)
            
if __name__ == "__main__": 
    unittest.main()   
    print(aws_names())
