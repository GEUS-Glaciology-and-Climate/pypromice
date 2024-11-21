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
from pypromice.process.L0toL1 import toL1
from pypromice.process.L1toL2 import toL2
from pypromice.process.L2toL3 import toL3
from pypromice.process import write, load, utilities
from pypromice.utilities.git import get_commit_hash_and_check_dirty

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
        logger.info(
            "AWS("
            f"config_file={config_file},"
            f" inpath={inpath},"
            f" data_issues_repository={data_issues_repository},"
            f" var_file={var_file},"
            f" meta_file={meta_file}"
            ")"
        )

        # Load config, variables CSF standards, and L0 files
        self.config = self.loadConfig(config_file, inpath)
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
        L0 = self.loadL0()
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

    def loadConfig(self, config_file, inpath):
        """Load configuration from .toml file

        Parameters
        ----------
        config_file : str
            TOML file path
        inpath : str
            Input folder directory where L0 files can be found

        Returns
        -------
        conf : dict
            Configuration parameters
        """
        conf = load.getConfig(config_file, inpath)
        return conf

    def loadL0(self):
        """Load level 0 (L0) data from associated TOML-formatted
        config file and L0 data file

        Try readL0file() using the config with msg_lat & msg_lon appended. The
        specific ParserError except will occur when the number of columns in
        the tx file does not match the expected columns. In this case, remove
        msg_lat & msg_lon from the config and call readL0file() again. These
        station files either have no data after Nov 2022 (when msg_lat &
        msg_lon were added to processing), or for whatever reason these fields
        did not exist in the modem message and were not added.

        Returns
        -------
        ds_list : list
            List of L0 xr.Dataset objects
        """
        ds_list = []
        for k in self.config.keys():
            target = self.config[k]
            try:
                ds_list.append(self.readL0file(target))

            except pd.errors.ParserError as e:
                # ParserError: Too many columns specified: expected 40 and found 38
                # logger.info(f'-----> No msg_lat or msg_lon for {k}')
                for item in ["msg_lat", "msg_lon"]:
                    target["columns"].remove(item)  # Also removes from self.config
                ds_list.append(self.readL0file(target))
            logger.info(f"L0 data successfully loaded from {k}")
        return ds_list

    def readL0file(self, conf):
        """Read L0 .txt file to Dataset object using config dictionary and
        populate with initial metadata

        Parameters
        ----------
        conf : dict
            Configuration parameters

        Returns
        -------
        ds : xr.Dataset
            L0 data
        """
        file_version = conf.get("file_version", -1)
        ds = load.getL0(
            conf["file"],
            conf["nodata"],
            conf["columns"],
            conf["skiprows"],
            file_version,
            time_offset=conf.get("time_offset"),
        )
        ds = utilities.populateMeta(ds, conf, ["columns", "skiprows", "modem"])
        return ds
