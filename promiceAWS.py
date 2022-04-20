from argparse import ArgumentParser
from promiceAWS import promiceAWS

def parse_arguments():
    parser = ArgumentParser(description="PROMICE AWS Processor")

    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to config (TOML) file')
    parser.add_argument('--data_dir', default='data', type=str, required=True, 
                        help='Path to data directory (containing L0 sub-folder)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """Executed from the command line"""
    args = parse_arguments()
    pAWS = promiceAWS(config_file=args.config_file, data_dir=args.data_dir)
    pAWS.process()
    pAWS.write(data_dir=pAWS.data_dir)

else:
    """Executed on import"""
    pass
        
