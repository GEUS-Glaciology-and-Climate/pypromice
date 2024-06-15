#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processing test module
"""
from pypromice.process.aws import AWS
from pypromice.process.load import getVars, getMeta
from pypromice.process.utilities import addVars, addMeta
import xarray as xr
import pandas as pd
import unittest, datetime, os
from datetime import timedelta

class TestProcess(unittest.TestCase):

    def testgetVars(self):
        '''Test variable table lookup retrieval'''
        v = getVars()
        self.assertIsInstance(v, pd.DataFrame)
        self.assertTrue(v.columns[0] in 'standard_name')
        self.assertTrue(v.columns[2] in 'units')

    def testgetMeta(self):
        '''Test AWS names retrieval'''
        m = getMeta()
        self.assertIsInstance(m, dict)
        self.assertTrue('references' in m)

    def testAddAll(self):
        '''Test variable and metadata attributes added to Dataset'''
        d = xr.Dataset()
        v = getVars()
        att = list(v.index)
        att1 = ['gps_lon', 'gps_lat', 'gps_alt', 'albedo', 'p']
        for a in att:
            d[a]=[0,1]
        for a in att1:
            d[a]=[0,1]
        d['time'] = [datetime.datetime.now(),
                     datetime.datetime.now()-timedelta(days=365)]
        d.attrs['station_id']='TEST'
        meta = getMeta()
        d = addVars(d, v)
        d = addMeta(d, meta)
        self.assertTrue(d.attrs['station_id']=='TEST')
        self.assertIsInstance(d.attrs['references'], str)

    def testL0toL3(self):
        '''Test L0 to L3 processing'''
        try:
            import pypromice
            pAWS = AWS(os.path.join(os.path.dirname(pypromice.__file__),'test/test_config1.toml'),
                       os.path.join(os.path.dirname(pypromice.__file__),'test'))
        except:
            pAWS = AWS('../test/test_config1.toml', '../test/')
        pAWS.process()
        self.assertIsInstance(pAWS.L2, xr.Dataset)
        self.assertTrue(pAWS.L2.attrs['station_id']=='TEST1')

    def testCLIgetl2(self):
        '''Test get_l2 CLI'''
        exit_status = os.system('get_l2 -h')
        self.assertEqual(exit_status, 0)
        
    def testCLIjoinl2(self):
        '''Test join_l2 CLI'''
        exit_status = os.system('join_l2 -h')
        self.assertEqual(exit_status, 0)
        
    def testCLIgetl2tol3(self):
        '''Test get_l2tol3 CLI'''
        exit_status = os.system('get_l2tol3 -h')
        self.assertEqual(exit_status, 0)
        
    def testCLIjoinl3(self):
        '''Test join_l3 CLI'''
        exit_status = os.system('join_l3 -h')
        self.assertEqual(exit_status, 0)

if __name__ == "__main__":

    unittest.main()
