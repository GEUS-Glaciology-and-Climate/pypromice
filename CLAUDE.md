# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

pypromice is the PROMICE/GC-Net toolbox for processing automatic weather station (AWS) data from the Greenland Ice Sheet. It processes raw transmission data through a pipeline of increasing quality levels: L0 (raw) → L1 (calibrated) → L2 (quality controlled) → L3 (derived variables).

The repository is a monorepo split into three packages sharing the `pypromice.*` namespace:

| Package | PyPI | Module namespace | Responsibility |
|---|---|---|---|
| `packages/pypromice/` | `pypromice` | `pypromice.*` | Core: L0→L3 pipeline, QC, file IO, resources |
| `packages/pypromice-tx/` | `pypromice-tx` | `pypromice.tx` | Transmission: IMAP email, Iridium SBD payload decoding |
| `packages/pypromice-bufr/` | `pypromice-bufr` | `pypromice.bufr` | WMO BUFR export |

The namespace sharing uses pkgutil: every package's `src/pypromice/__init__.py` starts with:
```python
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
```

## Development install

```bash
pip install -e "packages/pypromice[all]"
pip install -e "packages/pypromice-tx[google]"
pip install -e "packages/pypromice-bufr"
```

`eccodes` (required by pypromice-bufr) must be installed via conda-forge, not pip:
```bash
conda install -c conda-forge eccodes
```

## Running tests

```bash
# Per package
pytest packages/pypromice/tests/
pytest packages/pypromice-tx/tests/
pytest packages/pypromice-bufr/tests/

# Single file or test
pytest packages/pypromice/tests/unit/variables/test_wind.py
pytest packages/pypromice/tests/unit/qc/test_persistence.py::ClassName::test_method

# E2E integration tests (require real L0 data — see process_test.yml)
pytest tests/e2e/
```

## Pipeline architecture

**Data levels** (defined in `packages/pypromice/src/pypromice/pipeline/`):
- **L0toL1**: unit conversion, variable extraction, GPS/wind/radiation/pressure calibration
- **L1toL2**: QC — persistence detection, percentile outlier detection, rate-of-change filters, manual flags from the [aws-data-issues](https://github.com/GEUS-Glaciology-and-Climate/aws-data-issues) GitHub repo
- **L2toL3**: derived variables — turbulent fluxes, ice/snow surface heights, thermistor depths, GPS smoothing

**Central class:** `pypromice.pipeline.aws.AWS` — initialises from a TOML config file + L0 data directory, exposes `getL1()`, `getL2()`, `getL3()` methods. L0 files are either TOA5 (Campbell Scientific datalogger), NetCDF, or plain CSV; each station requires a TOML sidecar config specifying variable names and metadata.

**IO:** `pypromice.io.write.prepare_and_write()` handles all output — resamples to 10 min, hourly, daily, monthly and writes parallel NetCDF and CSV files. Variable metadata (units, standard_name) is looked up from `pypromice.resources` (variables.csv).

**Ingest** (`pypromice.ingest`): file-based L0 loading lives in `packages/pypromice/src/pypromice/ingest/` — `l0.py` detects file type and parses, `toa5.py` handles Campbell Scientific format, `l0_repository.py` manages multi-station repositories, `git.py` retrieves commit hashes.

**Transmission** (`pypromice.tx`): `packages/pypromice-tx/src/pypromice/tx/` — `tx.py` contains `L0tx`/`EmailMessage`/`PayloadFormat` classes for decoding Iridium SBD binary payloads from email attachments. Payload format is defined by `payload_formats.csv` and `payload_types.csv` bundled with the package.

## CLI entry points

```
get_l2            Process L0 raw files → L2 (via AWS class)
get_l2tol3        Process L2 NetCDF → L3
join_l2 / join_l3 Merge two products, preferring file1, gap-filling from file2
make_metadata_csv Generate metadata CSV from xarray dataset
get_l0tx          Fetch L0 data via IMAP email (transmission stations)
get_msg           Download raw .msg email files
get_bufr          Export L3 data to WMO BUFR format
create_bufr_files Batch BUFR creation from L3 directory
bufr_to_csv       Convert BUFR → CSV
```

## Versioning and releases

Git tags use a package-prefix format: `pypromice/v1.10.1`, `pypromice-tx/v1.0.0`, `pypromice-bufr/v1.0.0`. Each tag triggers its own publish workflow in `.github/workflows/publish-pypromice*.yml`. Versions are set manually in each package's `pyproject.toml` — tags do not automatically update the Python package version.
