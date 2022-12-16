import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dev',
    action='store_true',
    help='If included (True), run in dev mode. Useful for repeated runs of script between transmissions.')

parser.add_argument('--positions',
    action='store_true',
    help='If included (True), make a positions dict and output a positions.csv file.')

parser.add_argument('--l3-filepath',
    default='../../../../aws-l3/tx/*/*_hour.csv',
    type=str,
    help='Relative path to l3 tx .csv files.')

parser.add_argument('--bufr-out',
    default='./BUFR_out/',
    type=str,
    help='Path to the BUFR out directory.')

parser.add_argument('--time-limit',
    default='3M',
    type=str,
    help='Previous time to limit dataframe before applying linear regression.')

args = parser.parse_args()