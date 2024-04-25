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
import warnings, os

def aws_names():
    '''Return PROMICE and GC-Net AWS names that can be used in get.aws_data() 
    fetching'''
    lookup = lookup_table(['doi:10.22008/FK2/IW73UU'])
    print(f'Available dataset keywords: {list(lookup.keys())}')
    return list(lookup.keys())

def aws_data(aws_name):
    '''Return PROMICE and GC-Net AWS L3 v3 hourly observations
    
    Returns
    -------
    df : pandas.DataFrame
        AWS observations dataframe
    '''
    lookup = lookup_table(['doi:10.22008/FK2/IW73UU'])
    assert aws_name.lower() in list(lookup.keys())
    data = pd.read_csv(lookup[aws_name], index_col=0, parse_dates=True)
    return data        
  
def watson_discharge(t='hour'):
    '''Return PROMICE hourly Watson river discharge
    
    Parameters
    ----------
    t : str
        Temporal resolution of the data - "hour", "day" or "year"

    Returns
    -------
    df : pandas.DataFrame
        Watson river discharge dataframe    
    '''
    lookup = lookup_table(['doi:10.22008/FK2/XEHYCM'])
    
    dict_keys = lookup.keys()
    
    if 'year' in t.lower():
        
        key = [k for k in dict_keys if 'year' in k]
        
        if not key: 
            warnings.warn('The yearly Watson River Discharge file does not exist, or has changed name, on GEUS Dataverse DOI, ' + \
                  'please check the dataset, and the naming of the txt files on Dataverse')
                
        if len(key) > 1: 
            warnings.warn('Warning, there exist multiple yearly txt files on dataverse, please check ' + \
                  'if the correct txt file is used')
        
        link = lookup[key[0]]
        df = pd.read_csv(link, sep="\s+", skiprows=9, index_col=0)
        
    elif 'daily' in t.lower() or 'day' in t.lower():
        
        key = [k for k in dict_keys if 'daily' in k]
        
        if not key: 
            warnings.warn('The daily Watson River Discharge file does not exist, or has changed name, on GEUS Dataverse DOI, ' + \
                  'please check the dataset, and the naming of the txt files on Dataverse')
                
        if len(key) > 1: 
            warnings.warn('Warning, there exist multiple daily txt files on dataverse, please check ' + \
                  'if the correct txt file is used')
        
        link = lookup[key[0]]   
        
        df = pd.read_csv(link, sep="\s+", parse_dates=[[0,1,2]])\
                        .rename({"WaterFluxDiversOnly(m3/s)"        : "divers",
                                "Uncertainty(m3/s)"                 : "divers_err",
                                "WaterFluxDivers&Temperature(m3/s)" : "divers_t",
                                "Uncertainty(m3/s).1"               : "divers_t_err",
                                "WaterFluxCumulative(km3)"          : "Q",
                                "Uncertainty(km3)"                  : "err"}, 
                                axis='columns')
        df['time'] = df.iloc[:,0]
        df = df.set_index('time')
        df.drop(columns=df.columns[0:1], axis=1, inplace=True) 
        
    else:
        
        key = [k for k in dict_keys if 'hourly' in k]
        
        if not key: 
            warnings.warn('The hourly Watson River Discharge file does not exist, or has changed name, on GEUS Dataverse DOI, ' + \
                  'please check the dataset, and the naming of the txt files on Dataverse')
                
        if len(key) > 1: 
            warnings.warn('Warning, there exist multiple Houlry txt files on dataverse, please check ' + \
                  'if the correct txt file is used')
        
        link = lookup[key[0]]   
        
        df = pd.read_csv(link, sep="\s+", parse_dates=[[0,1,2,3]])\
                        .rename({"WaterFluxDiversOnly(m3/s)"        : "divers",
                                "Uncertainty(m3/s)"                 : "divers_err",
                                "WaterFluxDivers&Temperature(m3/s)" : "divers_t",
                                "Uncertainty(m3/s).1"               : "divers_t_err",
                                "WaterFluxCumulative(km3)"          : "Q",
                                "Uncertainty(km3)"                  : "err"}, 
                                axis='columns')
        df = _getDFdatetime(df, list(df.iloc[:,0]))            

    return df

def lookup_table(base_dois,
                 server='https://dataverse.geus.dk'):
    '''Fetch dictionary of data files and download URLs from a DOI entry in the
    GEUS Dataverse
    
    Parameters
    ----------
    base_dois : list
        List of DOIs to search
    server : str, optional
        DOI server. The default is "https://dataverse.geus.dk"
    '''
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
            if '.csv' in fname or '.txt' in fname:
                link = _getURL(f['dataFile']['persistentId'])
                lookup_list[fname] = link
    return lookup_list 

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
    dates = [datetime.strptime(str(d), dt_format) for d in dt_str]
    df['time'] = dates
    df = df.set_index('time')
    df.drop(columns=df.columns[0], axis=1, inplace=True)  
    return df

#------------------------------------------------------------------------------
        
class TestGet(unittest.TestCase): 
    def testURL(self):
        '''Test URL retrieval'''
        l = lookup_table(['doi:10.22008/FK2/IW73UU'])
        self.assertTrue('10.22008/FK2' in list(l.values())[0])
    
#    def testAWSname(self):  
#        '''Test AWS names retrieval'''
#        n = aws_names()
#        self.assertIsInstance(n, list)
#        self.assertTrue('nuk_k_hour.csv' in n)
    
#    def testAWScsv(self):
#        '''Test AWS data retrieval'''
#        kan_b = aws_data('kan_b_hour.csv')
#        self.assertIsInstance(kan_b, pd.DataFrame)
        
#    def testWatsonHour(self):
#        '''Test Wason River discharge hourly data retrieval'''
#        wh = watson_discharge()
#        self.assertTrue(wh['Q']['2021-10-27 23:00:00']==5.48)
        
#    def testWatsonDaily(self):
#        '''Test Wason River discharge daily data retrieval'''
#        wd = watson_discharge(t='day')
#        self.assertTrue(wd['Q']['2009-09-04 00:00:00']==4.72)

    def testGetCLI(self):
        '''Test get_promice_data'''
        exit_status = os.system('get_promice_data -h')
        self.assertEqual(exit_status, 0)
            
if __name__ == "__main__": 
    unittest.main()
