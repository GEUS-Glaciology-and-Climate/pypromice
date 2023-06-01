---
title: 'pypromice: A Python package for processing automated weather station data'
tags:
  - Python
  - glaciology
  - climate
  - promice
  - gc-net
  - geus
  - greenland
  - kalaallit-nunaat
authors:
  - name: Penelope R. How
    orcid: 0000-0002-8088-8497
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Patrick J. Wright
    orcid: 0000-0003-2999-9076
    affiliation: 1
  - name: Kenneth D. Mankoff
    orcid: 0000-0001-5453-2019
    affiliation: "1, 2, 3"
  - name: Baptiste Vandecrux
    orcid: 0000-0002-4169-8973
    affiliation: 1
  - name: Robert S. Fausto
    orcid: 0000-0003-1317-8185
    affiliation: 1
  - name: Andreas P. Ahlstrøm
    orcid: 0000-0001-8235-8070
    affiliation: 1
affiliations:
 - name: Department of Glaciology and Climate, Geological Survey of Denmark and Greenland (GEUS), Copenhagen, Denmark
   index: 1
 - name:  Autonomic Integra, New York, NY, USA
   index: 2
 - name: NASA Goddard Institute for Space Studies, New York, NY, USA
   index: 3

date: 03 March 2023
bibliography: paper.bib

---

# Summary

The `pypromice` Python package is for processing and handling observation datasets from automated weather stations (AWS). It is primarily aimed at users of AWS data from the Geological Survey of Denmark and Greenland (GEUS), which collects and distributes in situ weather station observations to the cryospheric science research community. Functionality in `pypromice` is primarily handled using two key open-source Python packages, `xarray` [@hoyer-xarray-2017] and `pandas` [@pandas-decpandas-2020].

A defined processing workflow is included in `pypromice` for transforming original AWS observations (Level 0, `L0`) to a usable, CF-convention-compliant dataset (Level 3, `L3`) (\autoref{fig:process}). Intermediary processing levels (`L1`,`L2`) refer to key stages in the workflow, namely the conversion of variables to physical measurements and variable filtering (`L1`), cross-variable corrections and user-defined data flagging and fixing (`L2`), and derived variables (`L3`). Information regarding the station configuration is needed to perform the processing, such as instrument calibration coefficients and station type (one-boom tripod or two-boom mast station design, for example), which are held in a `toml` configuration file. Two example configuration files are provided with `pypromice`, which are also used in the package's unit tests. More detailed documentation of the AWS design, instrumentation, and processing steps are described in @fausto-programme-2021.

![AWS data Level 0 (`L0`) to Level 3 (`L3`) processing steps, where `L0` refers to raw, original data and `L3` is usable data that has been transformed, corrected and filtered \label{fig:process}](https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/geus-glaciology-and-climate.github.io/master/assets/images/pypromice_process_design.png){ width=75% }

`L0` data is either collected from an AWS during a station visit or is transmitted in near-real-time from each AWS via the Iridium Short Burst Data (SBD) service. An object-oriented workflow for fetching and decoding SBD messages to Level 0 data (`L0 tx`) is included in `pypromice` (\autoref{fig:tx}). Alongside the processing module, this workflow can be deployed for operational uses to produce `L3` AWS data in near-real-time. A post-processing workflow is also included to demonstrate how near-real-time AWS data can be treated after `L3` for submission to global weather forecasting models under the World Meteorological Organisation ([WMO](https://public.wmo.int)).

![Object-oriented workflow in `pypromice.tx` for fetching and decoding AWS transmission messages to Level 0 (`L0 tx`) data \label{fig:tx}](https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/geus-glaciology-and-climate.github.io/master/assets/images/pypromice_tx_design.png){ width=75% }


# Statement of need

`pypromice` has four main research purposes:

1. Process and handle AWS observations
2. Document AWS data processing with transparency and reproducibility
3. Supply easy and accessible methods to handle AWS data
4. Provide opportunities to contribute to the processing and handling of AWS data in an open and collaborative manner

The `pypromice` software has been designed to handle and process data from AWSs located in Greenland. The compilation and processing of data from national AWS networks has historically been conducted through un-distributed, oftentimes proprietary software. Similar Python packages to `pypromice` have been developed to process data from historical AWS in Greenland [@vandecrux-gcnet-2020;@steffen-gcnet-2023], from commercial AWS (e.g. [pywws](https://pypi.org/project/pywws/), @easterbrook-pywws-2023), or to post-process and harmonize AWS data from different institutions (e.g. [JAWS](https://github.com/jaws/jaws), @zender-jaws-2019). Therefore, there was a key need for the development of `pypromice` in order to have a package with a complete and operational `L0` to `L3` workflow.


# Usage

The `pypromice` software handles data from 43 AWSs on hourly, daily and monthly time scales. The AWS data products have been used in high impact studies [@macguth-greenland-2016; @oehri_vegetation_2022; @box-greenland-2022], and have been crucial for evaluating the effect of climate change on land ice in annual reports such as the Arctic Report Card and "State of the Climate" [@moon-greenland-2022a; @moon-greenland-2022b]. The AWS data originates from three national monitoring programmes - the Programme for Monitoring of the Greenland Ice Sheet (PROMICE), the Greenland Climate Network (GC-Net) and the Greenland Ecosystem Monitoring programme (GEM). 

GEUS is responsible for the Programme for Monitoring of the Greenland Ice Sheet ([PROMICE](https://promice.org)), which is now a network of over 21 AWSs installed across the Greenland Ice Sheet [@ahlstrom-programme-2008]. Launched in 2007, these one-level tripod stations are designed to stand on ice and move with the ice flow close to the ice sheet periphery [@fausto-programme-2021; @how-one-boom-2022]. The PROMICE stations are designed to monitor the surface melt and its meteorological drivers in the ablation area of the ice sheet.

In 2021, GEUS assumed responsibility of the Greenland Climate Network (GC-Net) AWS locations [@steffen-greenland-1996], previously maintained by the United States National Science Foundation (NSF), National Aeronautics and Space Administration (NASA) and Swiss Federal Institute for Forest, Snow and Landscape Research (WSL). This expansion added 16 two-level mast stations to GEUS' sites. The data from these stations are intended to monitor conditions on the inner regions of the ice sheet, including snow accumulation and surface conditions [@how-one-boom-2022].

The Greenland Ecosystem Monitoring programme ([GEM](https://g-e-m.dk)) is an integrated, long-term monitoring effort that examines the effects of climate change on Arctic ecosystems. Established in 1995, GEM includes monitoring at Zackenberg, Kobbefjord, and Disko, Greenland. The program offers access to over 1000 freely-available environmental datasets, including data from 6 GEUS-designed AWS installations [@gem-glaciobasis-2020] which have been used in scientific publications [@messerli-snow-2022].

 
# Documentation

`pypromice` versions accompany releases of GEUS AWS data publications [@how-pypromice-2022].

Package documentation is available on the `pypromice` [readthedocs](https://pypromice.readthedocs.io/en/latest/). 

Guides for general GEUS AWS processing operations under PROMICE and GC-Net are included at the [GEUS Glaciology and Climate GitHub pages](https://geus-glaciology-and-climate.github.io/).


# Acknowledgements

This work is funded through the Danish Ministry of Climate, Energy and Utilities via The Programme for Monitoring of the Greenland Ice Sheet (PROMICE) and the Greenland Climate Network (GC-Net). 


# References

