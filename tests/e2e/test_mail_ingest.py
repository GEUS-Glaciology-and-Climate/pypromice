import unittest
from pathlib import Path
from pypromice.tx.aws_mail_ingest import pipeline
from pypromice.tx.aws_mail_ingest.loging_conf import setup_logging
from pypromice.tx.tx_monolite import parse_envelope
import gzip

setup_logging(
    file_path='logs.jsonl',
    fmt='json',
)


class MailIngestTestCase(unittest.TestCase):

    def test_init(self):
        pipeline.init_db()

        pipeline.ingest(max_messages=5*2048, window=1500, forward=True)

        pipeline.stats()



    def test_parse_raw_files(self):
        mail_root = Path('./blobs/raw/gmail/all_mail/')
        output_root = Path('./blobs/gzip/gmail/all_mail/')
        output_root.mkdir(exist_ok=True, parents=True)

        size_total = 0
        i = 0
        for i, mail_path in enumerate(mail_root.glob('*.eml')):
            with mail_path.open('br') as f:
                data = f.read()

                parse_envelope(data)

                size_total += len(data)


            if not (i%100):
                print(f"{i:6n} - Total {size_total*2**-20:5.1f}MB")

        print(f"{i:6n} - Total {size_total * 2 ** -20:5.1f}MB")

    # 248841