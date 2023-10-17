#!/usr/bin/env python
from argparse import ArgumentParser
import os, unittest
from pypromice.get.get import aws_data


def parse_arguments_data():
	parser = ArgumentParser(description="PROMICE and GC-Net dataset fetcher")       
	parser.add_argument('-n', '--awsname', default=None, type=str, required=True, 
		        help='AWS name')
	parser.add_argument('-f', '--format', default='csv', type=str, required=False, 
		        help='File format to save data as')                      
	parser.add_argument('-o', '--outpath', default=os.getcwd(), type=str, required=False, 
		        help='Directory where file will be written to')          	        
	args = parser.parse_args()
	return args


def get_promice_data():
    '''Command line driver for fetching PROMICE and GC-Net datasets'''

    args = parse_arguments_data()
	
    # Construct AWS dataset name
#    n = aws_names()
#    assert(args.awsname in n) 

    # Check file format type
    f = args.format.lower()
    assert(args.format in ['csv', 'nc', '.csv', '.nc'])
	
    # Construct output file path
    assert(os.path.exists(args.outpath))
	
    # Remove pre-existing files of same name
    if os.path.isfile(f):
        os.remove(f)
	    	
    # Fetch data 
    print(f'Fetching {args.awsname.lower()}...')
    data = aws_data(args.awsname.lower())
	
    # Save to file
    if f in 'csv':
        outfile = os.path.join(args.outpath, args.awsname.lower()) 
        if outfile is not None:
            data.to_csv(outfile)
    elif f in 'nc': 
        data.to_netcdf(outfile, mode='w', format='NETCDF4', compute=True)
        if outfile is not None:
            outfile = os.path.join(args.outpath, args.awsname.lower().split('.csv')[0]+'.nc')
        	
    print(f'File saved to {outfile}')

if __name__ == "__main__":  
    get_promice_data()
