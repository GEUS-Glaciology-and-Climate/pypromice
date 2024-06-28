#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processing test module
"""
from pathlib import Path

from pypromice.process.aws import AWS
from pypromice.process.load import getVars, getMeta
from pypromice.process.write import addVars, addMeta
import xarray as xr
import pandas as pd
import unittest, datetime, os
from datetime import timedelta

TEST_ROOT = Path(__file__).parent.parent
TEST_DATA_ROOT_PATH = TEST_ROOT / "data"
TEST_CONFIG_PATH = TEST_DATA_ROOT_PATH / "test_config1.toml"


class TestProcess(unittest.TestCase):

    def get_vars(self):
        '''Test variable table lookup retrieval'''
        v = getVars()
        self.assertIsInstance(v, pd.DataFrame)
        self.assertTrue(v.columns[0] in 'standard_name')
        self.assertTrue(v.columns[2] in 'units')

    def get_meta(self):
        '''Test AWS names retrieval'''
        m = getMeta()
        self.assertIsInstance(m, dict)
        self.assertTrue('references' in m)

    def add_all(self):
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
        d.attrs['level']='L2_test'
        meta = getMeta()
        d = addVars(d, v)
        d = addMeta(d, meta)
        self.assertTrue(d.attrs['station_id']=='TEST')
        self.assertIsInstance(d.attrs['references'], str)

    def l0_to_l3(self):
        '''Test L0 to L3 processing'''
        pAWS = AWS(TEST_CONFIG_PATH.as_posix(), TEST_DATA_ROOT_PATH.as_posix())
        pAWS.process()
        self.assertIsInstance(pAWS.L2, xr.Dataset)
        self.assertTrue(pAWS.L2.attrs['station_id']=='TEST1')

    def get_l2_cli(self):
        '''Test get_l2 CLI'''
        exit_status = os.system('get_l2 -h')
        self.assertEqual(exit_status, 0)
        
    def join_l2_cli(self):
        '''Test join_l2 CLI'''
        exit_status = os.system('join_l2 -h')
        self.assertEqual(exit_status, 0)
        
    def l2_to_l3_cli(self):
        '''Test get_l2tol3 CLI'''
        exit_status = os.system('get_l2tol3 -h')
        self.assertEqual(exit_status, 0)
        
    def join_l3_cli(self):
        '''Test join_l3 CLI'''
        exit_status = os.system('join_l3 -h')
        self.assertEqual(exit_status, 0)

if __name__ == "__main__":
    unittest.main()
