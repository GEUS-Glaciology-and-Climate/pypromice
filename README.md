# pyPROMICE
[![](<https://img.shields.io/badge/Dataverse DOI-10.22008/FK2/IPOHT5-orange>)](https://www.doi.org/10.22008/FK2/IPOHT5)
 
The pyPROMICE toolbox is for retrieving, processing and handling PROMICE datasets. This is a development toolbox, compiled from several repositories:
- Receive and decode transmissions from PROMICE automatic weather stations - [awsrx](https://github.com/GEUS-Glaciology-and-Climate/awsrx)
- AWS L0 >> L3 processing - [PROMICE-AWS-processing](https://github.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-processing)
- Post-processing AWS L3 data, including flagging, filtering and fixing - [PROMICE-AWS-toolbox](https://github.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-toolbox), [GC-Net-level-1-data-processing](https://github.com/GEUS-Glaciology-and-Climate/GC-Net-level-1-data-processing)
- WMO data processing into BUFR formats for operational ingestion - [csv2bufr](https://github.com/GEUS-Glaciology-and-Climate/csv2bufr)
- Retrieving PROMICE/GC-Net datasets from Dataverse/online, with no downloading - [PROMICE](https://github.com/GEUS-Glaciology-and-Climate/PROMICE)


## Installation

pyPROMICE can either be cloned and ran in an environment with the needed dependencies:
```
conda create --name pypromice python=3.8
conda activate pypromice

conda install xarray pandas pathlib
pip install netCDF4
pip install scikit-learn # If you will be running `postprocess/csv2bufr.py`

git clone git@github.com:GEUS-Glaciology-and-Climate/pypromice.git
```

Or installed directly using pip:

```
pip install --upgrade git+http://github.com/GEUS-Glaciology-and-Climate/pypromice.git
```

Note that [eccodes](https://confluence.ecmwf.int/display/ECC/ecCodes+installation), the official package for BUFR encoding and decoding, is not included in this set-up given the problems that conda has with resolving environments. Try firstly to install with conda-forge like so:

```
conda install -c conda-forge eccodes
```

If the environment cannot resolve the eccodes installation then follow the steps documented [here](https://gist.github.com/MHBalsmeier/a01ad4e07ecf467c90fad2ac7719844a) to download eccodes, and then install eccodes' python bindings using pip:

```
pip3 install eccodes-python
```

Once the pyPROMICE toolbox is cloned/installed, the toolbox can be checked with its in-built unittesting:

```
python -m unittest tx.py aws.py get.py
```
 
## Design

![pypromice](https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/pypromice/main/fig/pypromice_prelim.png)

It is planned that the pyPROMICE toolbox will be the go-to tool for handling and processing PROMICE and GC-Net datasets, available through pip and conda-forge, perhaps even across platforms such as R and Matlab:

```
conda install -c conda-forge pypromice

pip install pypromice
```

And new releases of pyPROMICE will be uploaded to GEUS' Dataverse with a generated DOI - see instructions [here](https://github.com/marketplace/actions/dataverse-uploader-action). 


## tx

The `tx` module contains all objects and functions for processing transmissions to Level 0 (L0) data products. Credentials are needed to access emails - `accounts.ini`, `credentials.ini` - along with a list of modem numbers and names (`imei2name.ini`).

### Level 0 data products

![tx_workflow](https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/pypromice/main/fig/L00_to_L0.png)

Level 0 is generated from one of three methods:
- [ ] Copied from CF card in the field
- [ ] Downloaded from logger box in the field
- [X] Transmitted via satellite and decoded in this module

Where Level 0 files are collected via:
- `raw`: All 10-minute data stored on the CF-card (external module on CR logger)
- `SlimTableMem`: Hourly averaged 10-min data stored in the internal logger memory
- `transmitted`: Transmitted via satellite. Only a subset of data is transmitted, and only hourly or daily average depending on station and day of year

Level 0 files are stored in the `data/L0/<S>/` folder, where `<S>` is the station name. File names can be anything are are processed as per the =TOML= config files, but ideally they should encode the station, end-of-year of download, a version number if there are multiple files for a given year, and the format. Best practices would use the following conventions:  

- **Generic**: `=data/<L>/<S>/<S>_<Y>[.<n>]_<F>.txt=`
- **Example**: `=data/L0/QAS_L/QAS_L_2021_raw_transmitted.txt=`

Where 

- `<L>` is the processing level
  - `<L>` must be one of the following: `[L0, L1, L1A, L2, L3]`
- `<S>` is a station ID
  - `<S>` must be one of the following strings: `[CEN, EGP, KAN_B, KAN_L, KAN_M, KAN_U, KPC_L, KPC_U, MIT, NUK_K, NUK_L, NUK_N, NUK_U, QAS_A, QAS_L, QAS_M, QAS_U, SCO_L, SCO_U, TAS_A, TAS_L, TAS_U, THU_L, THU_U, UPE_L, UPE_U]`
- `<Y>` is a four-digit year with a value greater than =2008=
  - `<Y>` should represent the year at the last timestamp in the file
  - Optionally, `.<n>` is a version number if multiple files from the same year are present
- `<F>` is the format, one of `raw`, `TX`, or `STM`

Each L0 file that will be processed must have an entry in the TOML-formatted configuration file. The config file can be located anywhere, and the processing script receives the config file and the location of the L0 data (see example below).

```
station_id         = "EGP"
latitude           = 75.62
longitude          = -35.98
nodata             = ['-999', 'NAN'] # if one is a string, all must be strings
dsr_eng_coef       = 12.71  # from manufacturer to convert from eng units (1E-5 V) to  physical units (W m-2)
usr_eng_coef       = 12.71
dlr_eng_coef       = 12.71
ulr_eng_coef       = 12.71

columns = ["time", "rec", "min_y",
	"p", "t_1", "t_2", "rh", "wspd", "wdir", "wd_std",
	"dsr", "usr", "dlr", "ulr", "t_rad",
	"z_boom", "z_boom_q", "z_stake", "z_stake_q", "z_pt",
	"t_i_1", "t_i_2", "t_i_3", "t_i_4", "t_i_5", "t_i_6", "t_i_7", "t_i_8",
	"tilt_x", "tilt_y",
	"gps_time", "gps_lat", "gps_lon", "gps_alt", "gps_geoid",
	"SKIP", "SKIP", "gps_numsat", "gps_hdop",
	"t_log", "fan_dc", "SKIP", "batt_v_ss", "batt_v"]

# Parameters applied to all files are above.
# Define files for processing and
# override file-specific parameters below.

["EGP_2016_raw.txt"]
format    = "raw"
skiprows  = 3
hygroclip_t_offset = 0      # degrees C

["EGP_2019_raw_transmitted.txt"]
hygroclip_t_offset = 0
skiprows = 0
format   = "TX"
columns = ["time", "rec",
	"p", "t_1", "t_2", "rh", "wspd", "wdir",
	"dsr", "usr", "dlr", "ulr", "t_rad",
	"z_boom", "z_stake", "z_pt",
	"t_i_1", "t_i_2", "t_i_3", "t_i_4", "t_i_5", "t_i_6", "t_i_7", "t_i_8",
	"tilt_x", "tilt_y",
	"gps_time", "gps_lat", "gps_lon", "gps_alt", "gps_hdop",
	"fan_dc", "batt_v"]
```

The TOML config file has the following expectations and behaviors:
- Properties can be defined at the top level or under a section
- Each file that will be processed gets its own section
- Properties at the top level are copied to each section (assumed to apply to all files)
- Top-level properties are overridden by file-level properties if they exist in both locations

In the example above,
- The `station_id`, `latitude`, etc. properties are the same in both files (`EGP_2016_raw.txt` and `EGP_2019_raw_transmitted.txt`) and so they are defined once at the top of the file. They could have been defined in each of the sections similar to `hygroclip_t_offset`.
- The `format` and `skiprows` properties are different in each section and defined in each section
- The top-level defined `columns` is applied only to `EGP_2016_raw.txt` because it is defined differently in the `EGP_2019_raw_transmitted.txt` section.

Any files that do not have an associated section in the config file will be ignored. However, for cleanliness, L0 files that will not be processed should be placed in an `L0/<S>/archive` subfolder.

### tx workflow

The workflow in `tx.py` fetches messages over IMAP sent from the Iridium SBD service. These messages are decoded from the binary format transmitted by the AWS, and appends each dataline to the corresponding station that transmitted it (based on the modem number, `imei`). 

The workflow is object-oriented to handle each component needed to fetch and decode messages.

![tx_workflow](https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/pypromice/main/fig/tx_design.png)

1. `PayloadFormat` handles the message types and formatting templates. These can be imported from file, with the two .csv files in the `payload_formatter` currently used. These used to be hardcoded in the script, but it seemed more appropriate to store them in files

2. `SbdMessage` handles the SBD message, either taken from an `email.message.Message` object or a .sbd file (half completed, still being developed)

3. `EmailMessage` handles the email message (that the SBD message is attached to) to parse information such as sender, subject, date, and to check for attachments. The `EmailMessage` inherits from the `SbdMessage` object, as the SBD message is part of the email. Previously this was the opposite which, although followed the workflow steps, was unintuitive for the object-oriented workflow design

4. `L0tx` handles the processing and output of the L0 transmission dataline. This object inherits from `EmailMessage` and `PayloadFormat` to read and decode messages

To reprocess old messages, these can be retrieved from the mailbox by rolling back the counter in `last_aws_uid.ini` or by reading from .sbd file.

To fetch L0 TX messages from all valid stations:

```
getL0tx -a accounts.ini -p credentials.ini -c tx/config -f payload_formats.csv -t payload_types.csv -u last_aws_uid.ini -o tx
```


## get

The `get` module is for fetching PROMICE datasets from GEUS' Dataverse without downloads. Currently this is up and running for retrieving AWS datasets and hydrological observations from the Watson River. 

To retrieve an AWS dataset as a pandas DataFrame:

```
from get import aws_data, aws_names

# To check AWS names
aws_names()

# To retrieve AWS data from KAN_B
kan_b = aws_data('KAN_B')
```

To retrieve hydrological observations from the Watson River:
        
```
from get import watson_discharge_hourly, watson_discharge_daily

# Fetch daily observations
wd = watson_discharge_daily()
   
# Fetch hourly observations
wh = watson_discharge_hourly()
```


## process

The `process` module is for processing PROMICE AWS observations from Level 0 (L0) to Level 3 (L3, end-user product) data products. Currently, this code focuses on the **transmitted** data.

To process from L0>>L3, the following command can be used:

```
getL3 -v variables.csv -m metadata.csv -c config/KPC_L.toml -i . -o ../../aws-l3/tx"
```

And in parallel through all configuration .toml files `$imei_list`:

```
parallel --bar "getL3 -v variables.csv -m metadata.csv -c ./{} -i . -o ../../aws-l3/tx" ::: $(ls $imei_list)
```

This processes L0 files in the following manner:

- L0: Raw data in CSV file format in one of three formats:
  - [ ] =raw= (see below)
  - [ ] =STM= (Slim Table Memory; see below)
  - [X] =TX= (transmitted; see below)
  - Manually split so no file includes changed sensors, changed calibration parameters, etc.
  - [X] Manually created paired header files based on [[./example.toml]] or in the =data/L0/config= folder.
- L1:
  - [X] Engineering units (e.g. current or volts) converted to physical units (e.g. temperature or wind speed)
- L1A:
  - [ ] Invalid / bad / suspicious data flagged
  - [X] Files merged to one time series per station
- L2:
  - [X] Calibration using secondary sources (e.g. radiometric correction requires input of tilt sensor)
- L3:
  - [X] Derived products (e.g. SHF and LHF)
  - [ ] Merged, patched, and filled (raw > STM > TX) to one product
  
![process_workflow](https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/pypromice/main/fig/levels.png)  
  


