import unittest


class TestSDBDecode(unittest.TestCase):


    # PROMICEv3_Rewired_1_5.CR1X
    def test_sdb_decode(self):
        from tx.sdb import sdb_decode



        sdb_data = b'\x00\x01\x02\x03\x04\x05\x06\x07'

        # Decode the SDB data
        decoded_data = sdb_decode(sdb_data)

        # Check if the decoded data is as expected
        assert decoded_data == b'\x00\x01\x02\x03\x04\x05\x06\x07'  # Replace with expected output

