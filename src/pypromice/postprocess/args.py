import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dev',
    action='store_true',
    help='If included (True), run in dev mode.')

parser.add_argument('--l3-path-dev',
    default='/home/pwright/GEUS/pypromice-dev/aws-l3/level_3',
    type=str,
    help='Path to the dev l3 directory.')

parser.add_argument('--l3-files-dev',
    default='/home/pwright/GEUS/pypromice-dev/aws-l3/level_3/*/*_hour.csv',
    type=str,
    help='Path to the string pattern for l3 instantaneous tx files.')

parser.add_argument('--bufr-out',
    default='./BUFR_out/',
    type=str,
    help='Path to the BUFR out directory.')

parser.add_argument('--time-limit',
    default='14D',
    type=str,
    help='Previous time to limit BUFR files.')

args = parser.parse_args()