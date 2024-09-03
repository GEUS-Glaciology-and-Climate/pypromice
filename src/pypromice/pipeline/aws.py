#!/usr/bin/env python
"""
AWS data processing module
"""
import json
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging, os
from pathlib import Path
import pandas as pd
import xarray as xr
from functools import reduce
from importlib import metadata


import pypromice.resources
from pypromice.pipeline.L0toL1 import toL1
from pypromice.pipeline.L1toL2 import toL2
from pypromice.pipeline.L2toL3 import toL3
from pypromice.pipeline import utilities
from pypromice.io import write
from pypromice.io.ingest.l0 import (load_data_files, load_config)
from pypromice.io.ingest.git import get_commit_hash_and_check_dirty

pd.set_option("display.precision", 2)
xr.set_options(keep_attrs=True)
logger = logging.getLogger(__name__)


class AWS(object):
    """AWS object to load and process PROMICE AWS data"""

    def __init__(
        self,
        config_file,
        inpath,
        data_issues_repository: Path | str,
        var_file=None,
        meta_file=None,
    ):
        """Object initialisation

        Parameters
        ----------
        config_file : str
            Configuration file path
        inpath : str
            Input file path
        var_file: str, optional
            Variables look-up table file path. If not given then pypromice's
            variables file is used. The default is None.
        meta_file: str, optional
            Metadata info file path. If not given then pypromice's
            metadata file is used. The default is None.
        """
        assert os.path.isfile(config_file), "cannot find " + config_file
        assert os.path.isdir(inpath), "cannot find " + inpath

        logger.debug(
            "AWS("
            f"config_file={config_file},"
            f" inpath={inpath},"
            f" data_issues_repository={data_issues_repository},"
            f" var_file={var_file},"
            f" meta_file={meta_file}"
            ")"
        )

        # Load config, variables CSF standards, and L0 files
        self.vars = pypromice.resources.load_variables(var_file)
        self.meta = pypromice.resources.load_metadata(meta_file)
        self.data_issues_repository = Path(data_issues_repository)

        config_hash = get_commit_hash_and_check_dirty(Path(config_file))
        config_source_string = f"{Path(config_file).name}:{config_hash}"
        inpath_hash = get_commit_hash_and_check_dirty(Path(inpath))
        data_issues_hash = get_commit_hash_and_check_dirty(self.data_issues_repository)
        source_dict = dict(
            pypromice=metadata.version("pypromice"),
            l0_config_file=config_source_string,
            l0_data_root=inpath_hash,
            data_issues=data_issues_hash,
        )
        logger.debug('Source information: %s', source_dict)
        self.meta["source"] = json.dumps(source_dict)

        # Load config file
        config = load_config(config_file, inpath)
        L0 = load_data_files(config)

        self.L0 = []
        for l in L0:
            n = write.getColNames(self.vars, l)
            self.L0.append(utilities.popCols(l, n))

        formats = {dataset.attrs["format"].lower() for dataset in self.L0}
        if "raw" in formats:
            self.format = "raw"
        elif "stm" in formats:
            self.format = "STM"
        elif "tx" in formats:
            self.format = "tx"
        else:
            raise ValueError(f"Unknown formats from l0 datasets: {','.join(formats)}")

        self.L1 = None
        self.L1A = None
        self.L2 = None
        self.L3 = None

    def process(self):
        """Perform L0 to L3 data processing"""
        try:
            logger.info(
                f'Commencing {self.L0.attrs["number_of_booms"]}-boom processing...'
            )
            logger.info(
                f'Commencing {self.L0.attrs["number_of_booms"]}-boom processing...'
            )
        except:
            logger.info(
                f'Commencing {self.L0[0].attrs["number_of_booms"]}-boom processing...'
            )
        self.getL1()
        self.getL2()
        self.getL3()

    def getL1(self):
        """Perform L0 to L1 data processing"""
        logger.info("Level 1 processing...")
        self.L0 = [utilities.addBasicMeta(item, self.vars) for item in self.L0]
        self.L1 = [toL1(item, self.vars) for item in self.L0]
        self.L1A = reduce(xr.Dataset.combine_first, reversed(self.L1))
        self.L1A.attrs["format"] = self.format

    def getL2(self):
        """Perform L1 to L2 data processing"""
        logger.info("Level 2 processing...")

        self.L2 = toL2(
            self.L1A,
            vars_df=self.vars,
            data_flags_dir=self.data_issues_repository / "flags",
            data_adjustments_dir=self.data_issues_repository / "adjustments",
        )

    def getL3(self):
        """Perform L2 to L3 data processing, including resampling and metadata
        and attribute population"""
        logger.info("Level 3 processing...")
        self.L3 = toL3(self.L2, data_adjustments_dir=self.data_issues_repository / "adjustments")

