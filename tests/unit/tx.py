#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing for pypromice.tx module for transmission fetching and decoding
"""
from pathlib import Path
import unittest, email, os
from pypromice.tx import L0tx, EmailMessage, PayloadFormat

TEST_ROOT = Path(__file__).parent.parent
TEST_DATA_ROOT_PATH = TEST_ROOT / "data"
TEST_EMAIL_PATH = TEST_DATA_ROOT_PATH / "test_email"

with TEST_EMAIL_PATH.open("r") as f:
    content = f.read()
    TEST_EMAIL = email.message_from_bytes(content)



class TestTX(unittest.TestCase):
    def test_payload_format(self):
        '''Test PayloadFormat object initialisation'''
        p = PayloadFormat()
        self.assertTrue(p.payload_format[30][0] == 12)

    def test_email_message(self):
        '''Test EmailMessage object initialisation from .msg file'''
        e = EmailMessage(TEST_EMAIL, sender_name='sbdservice')
        self.assertEqual(e.momsn, 36820)
        self.assertEqual(e.msg_size, 27)
        self.assertTrue(e.imei in '300234061165160')
        self.assertFalse(e.mtmsn)

    def test_l0_tx_object(self):
        '''Test L0tx object initialisation'''
        l0 = L0tx(TEST_EMAIL)
        self.assertTrue(l0.bin_valid)
        self.assertEqual(l0.bin_val, 12)
        self.assertTrue('tfffffffffff' in l0.bin_format)
        self.assertTrue('2022-07-25 10:00:00' in l0.msg)
        self.assertFalse('?' in l0.msg)

    def test_l0_tx_cli(self):
        '''Test get_l0tx CLI'''
        exit_status = os.system('get_l0tx -h')
        self.assertEqual(exit_status, 0)
