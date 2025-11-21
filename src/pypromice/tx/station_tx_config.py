from datetime import datetime, timedelta
from pathlib import Path
from typing import List, TypedDict

import pandas as pd
import toml

__all__ = ["TxConfig", "load_tx_configurations", "get_tx_configs"]


class TxConfig(TypedDict):
    start: datetime
    end: datetime
    imei: str
    name: str
    filename: str
    stid: str

def load_tx_configurations(*toml_list: Path) -> pd.DataFrame:
    """
    Loads transmission configurations from TOML files and checks for overlaps.
    """
    # Parse config files
    tx_configs = get_tx_configs(*toml_list)

    # Convert to dataframe
    tx_config_df = pd.DataFrame(tx_configs)

    # Sort by imei and start time for easier overlap detection
    tx_config_df.sort_values(by=['imei', 'start'], inplace=True)

    # Check for overlaps within each imei group
    for imei, imei_configs in tx_config_df.groupby('imei'):
        stids = ','.join(imei_configs['stid'].unique())
        prev_end = None
        for idx, row in imei_configs.iterrows():
            if prev_end and row['start'] < prev_end:
                raise ValueError(f"Overlapping transmission config for IMEI {imei}: "
                                 f"{row['start']} starts before {prev_end} ends."
                                 f" Station IDs: {stids}"
                                 )
            prev_end = row['end']


    return tx_config_df


def get_tx_configs(*toml_list: Path) -> List[TxConfig]:
    """
    Extracts tx modem configurations from TOML files.
    """
    tx_configs = []
    for t in toml_list:
        conf = toml.load(t)
        stid = conf["station_id"]
        count = 1
        for m in conf["modem"]:
            imei = m[0]
            start_time = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S")
            if len(m) > 2:
                end_time = datetime.strptime(m[2], "%Y-%m-%d %H:%M:%S")
            else:
                end_time = None
                # end_time = datetime.now() + timedelta(hours=3)
            name = str(conf["station_id"]) + "_" + m[0] + "_" + str(count)
            filename = f'{name}.txt'

            tx_configs.append(
                TxConfig(
                    start=start_time,
                    end=end_time,
                    imei=imei,
                    name=name,
                    filename=filename,
                    stid=stid
                )
            )
            count += 1
    return tx_configs
