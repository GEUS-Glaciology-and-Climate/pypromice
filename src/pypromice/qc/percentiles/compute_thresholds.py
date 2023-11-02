import sys
from datetime import datetime

import pandas as pd

from pypromice.process import AWS
from pathlib import Path
import logging
from pypromice.qc.github_data_issues import adjustTime, flagNAN, adjustData


# %%
logger = logging.getLogger("ComputeThreshold")


# %%
def compute_all_thresholds(
    station_thresholds_root: Path,
    thresholds_output_path: Path,
    aws_l0_repo_path: Path,
    start_time: datetime,
    end_time: datetime,
):
    logger.info("Computing all thresholds for stations available in the L0 repository")
    logger.info(f"station_thresholds_root: {station_thresholds_root}")
    logger.info(f"thresholds_output_path:  {thresholds_output_path}")
    logger.info(f"aws_l0_repo_path:        {aws_l0_repo_path}")
    logger.info(f"start_time:              {start_time}")
    logger.info(f"end_time:                {end_time}")

    station_thresholds_root.mkdir(parents=True, exist_ok=True)

    # %%
    output_paths = []
    for config_path in aws_l0_repo_path.glob("raw/config/*.toml"):
        stid = config_path.stem

        logger.info(f"Processing {stid}")
        data_path = aws_l0_repo_path.joinpath("raw", stid)
        output_path = station_thresholds_root.joinpath(f"{stid}.csv")
        try:
            if not output_path.exists():
                threshold = find_thresholds(
                    stid,
                    config_path,
                    data_path,
                    start_time,
                    end_time,
                )
                threshold.to_csv(
                    path_or_buf=output_path, index=False, float_format="{:.2f}".format
                )
            output_paths.append(output_path)
        except Exception:
            logger.exception(f"Failed processing {stid}")
            continue

    logger.info("Merge threshold files")
    pd.concat(pd.read_csv(p) for p in output_paths).to_csv(
        thresholds_output_path, index=False, float_format="{:.2f}".format
    )


def find_thresholds(
    stid: str,
    config_path: Path,
    data_path: Path,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    """
    Compute variable threshold for a station using historical distribution quantiles.

    Parameters
    ----------
    stid
    config_path
    data_path
    start_time
    end_time

    Returns
    -------
    Upper and lower thresholds for a set of variables and seasons


    """
    stid_logger = logger.getChild(stid)
    # %%

    stid_logger.info("Read AWS data and get L1")
    aws = AWS(config_file=config_path.as_posix(), inpath=data_path.as_posix())
    aws.getL1()

    # %%
    stid_logger.info("Apply QC filters on data")
    ds = aws.L1A.copy(deep=True)  # Reassign dataset
    ds = adjustTime(ds)  # Adjust time after a user-defined csv files
    ds = flagNAN(ds)  # Flag NaNs after a user-defined csv files
    ds = adjustData(ds)

    # %%
    stid_logger.info("Determine thresholds")
    df = (
        ds[["rh_u", "wspd_u", "p_u", "t_u"]]
        .to_pandas()
        .loc[start_time:end_time]
        .assign(season=lambda df: (df.index.month // 3) % 4)
    )

    threshold_rows = []

    # Pressure
    p_lo, p_hi = df["p_u"].quantile([0.005, 0.995]) + [-12, 12]
    threshold_rows.append(
        dict(
            stid=stid,
            variable_pattern="p_[ul]",
            lo=p_lo,
            hi=p_hi,
        )
    )
    threshold_rows.append(
        dict(
            stid=stid,
            variable_pattern="p_i",
            lo=p_lo - 1000,
            hi=p_hi - 1000,
        )
    )

    # Wind speed
    lo, hi = df["wspd_u"].quantile([0.005, 0.995]) + [-12, 12]
    threshold_rows.append(
        dict(
            stid=stid,
            variable_pattern="wspd_[uli]",
            lo=lo,
            hi=hi,
        )
    )

    # Temperature
    season_map = ["winter", "spring", "summer", "fall"]
    for season_index, season_df in df[["t_u", "season"]].groupby(
        (df.index.month // 3) % 4
    ):
        lo, hi = season_df.quantile([0.005, 0.995])["t_u"] + [-9, 9]

        threshold_rows.append(
            dict(
                stid=stid,
                variable_pattern="t_[uli]",
                season=season_map[season_index],
                lo=lo,
                hi=hi,
            )
        )

    threshold = pd.DataFrame(threshold_rows)
    stid_logger.info(threshold)
    return threshold
    # %%


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--l0",
        required=True,
        type=Path,
        help="L0 repository root path",
    )
    parser.add_argument(
        "--thresholds_output_path",
        "-o",
        default=Path(__file__).parent.joinpath("thresholds.csv"),
        type=Path,
        help="Output csv file with thresholds for all stations",
    )
    parser.add_argument(
        "--station_thresholds_root",
        "--str",
        default=Path(__file__).parent.joinpath("station_thresholds"),
        type=Path,
        help="Directory containing threshold files for the individual stations",
    )
    parser.add_argument(
        "--start_time",
        default="2000-01-01",
        help="Start time for data series. Format: %Y-%m-%d",
    )
    parser.add_argument(
        "--end_time",
        default="2023-10-01",
        help="End time for data series. Format: %Y-%m-%d",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s; %(levelname)s; %(name)s; %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    thresholds_output_path = args.thresholds_output_path
    station_thresholds_root = args.station_thresholds_root

    start_time = datetime.strptime(args.start_time, "%Y-%m-%d")
    end_time = datetime.strptime(args.end_time, "%Y-%m-%d")
    aws_l0_repo_path = args.l0

    compute_all_thresholds(
        station_thresholds_root=station_thresholds_root,
        thresholds_output_path=thresholds_output_path,
        aws_l0_repo_path=aws_l0_repo_path,
        start_time=start_time,
        end_time=end_time,
    )
