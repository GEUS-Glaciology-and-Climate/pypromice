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
 - name: Department of Glaciology and Climate, Geological Survey of Denmark and Greenland (GEUS), Denmark
   index: 1
 - name:  Business Integra, New York, NY, USA
   index: 2
 - name: NASA Goddard Institute for Space Studies, New York, NY, USA
   index: 3

date: 01 March 2023
bibliography: paper.bib

---

# Summary

<!--   A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.	-->

The `pypromice` Python package is for processing and handling observation datasets from automated weather stations (AWS). It is primarily aimed at users of AWS data from the Geological Survey of Denmark and Greenland (GEUS), which is a valuable set of in situ observations to the cryospheric science research community. Functionality in `pypromice` is primarily handled using two key open-source Python packages, `xarray` [@hoyer-xarray-2017] and `pandas` [@pandas-decpandas-2020].

A defined processing workflow is included in `pypromice` for transforming original AWS observations (Level 0, `L0`) to a usable, CF-convention-compliant dataset (Level 3, `L3`) (\autoref{fig:process}). Intermidiary processing levels (`L1`,`L2`) refer to key stages in the workflow, namely the conversion of variables to physical measurements and variable filtering (`L1`), cross-variable corrections (`L2`), and derived variables (`L3`). Information regarding the station configuration is needed to perform the processing, such as instrument calibration coefficients and station type (one-boom or two-boom station design, for example), which are held in a `toml` configuration file. Two example configuration files are provided with `pypromice`, which are also used in the package's unit tests. More detailed documentation of the AWS design, instrumentation, and processing steps are described in [@fausto-programme-2021].

![AWS data Level 0 (`L0`) to Level 3 (`L3`) processing steps, where `L0` refers to raw, original data and `L3` is usable data that has been transformed, corrected and filtered \label{fig:process}](https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/geus-glaciology-and-climate.github.io/master/assets/images/pypromice_process_design.png){ width=75% }

`L0` data is either collected from an AWS during a station visit or is transmitted in near-real-time from each PROMICE AWS via the Iridium Short Burst Data (SBD) service. An object-oriented workflow for fetching and decoding SBD messages to Level 0 data (`L0 tx`) is included in `pypromice` (\autoref{fig:tx}). Alongside the processing module, this workflow has been deployed for operational uses in PROMICE to produce `L3` AWS data in near-real-time. A post-processing workflow is also included to demonstrate how near-real-time AWS data is treated after `L3` for submission to global weather forecasting models under the World Meteorological Organisation ([WMO](https://public.wmo.int)).

![Object-oriented workflow in `pypromice.tx` for fetching and decoding AWS transmission messages to Level 0 (`L0 tx`) data \label{fig:tx}](https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/geus-glaciology-and-climate.github.io/master/assets/images/pypromice_tx_design.png){ width=75% }


# Statement of need

<!--   A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work.  -->

`pypromice` has four main research purposes:

1. Process and handle AWS observations  
2. Document the PROMICE and GC-Net AWS processing with transparency and reproducibility
3. Supply easy and accessible methods to handle AWS data
4. Provide opportunities to contribute to the processing and handling of AWS data in an open and collaborative manner


# Usage

<!--    Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->

The `pypromice` software has been designed to handle and process data from AWSs located in Greenland. Similar functionality is available in the [JAWS](https://github.com/jaws/jaws) Python package [@zender-jaws-2019], however JAWS is not no longer maintained and is not compatible with the most recent AWS data format. Therefore, there was a key need for the development of `pypromice`. The following sections outline ongoing research projects and monitoring efforts which actively use `pypromice`. 

GEUS is responsible for the Programme for Monitoring of the Greenland Ice Sheet ([PROMICE](https://promice.org)), which is now a network of over 21 AWSs installed across the Greenland Ice Sheet [@ahlstrom-programme-2008]. Launched in 2007, these one-level tripod stations are designed to sit on ice and move with the ice flow close to the ice sheet periphery [@fausto-programme-2021].

In 2020, GEUS assumed responsibility of the American Greenland Climate Network (GC-Net) and expanded operations by including 16 two-level mast stations for monitoring conditions on the interior ice sheet [@steffen-greenland-1996]. The data from these stations are intended to monitor conditions on the inner regions of the ice sheet, including snow accumulation and surface conditions.

The Greenland Ecosystem Monitoring programme ([GEM](https://g-e-m.dk)) is an integrated, long-term monitoring effort that examines the effects of climate change on Arctic ecosystems. Established in 1995, GEM includes monitoring at Zackenberg, Kobbefjord, and Disko. The program offers access to over 1000 freely-available environmental datasets, including data from 6 GEUS-designed AWS installations [@gem-glaciobasis-2020] which have been used in scientific publications [@messerli-snow-2022].

The `pypromice` software thus handles data from 43 AWSs on hourly, daily and monthly time scales, producing standard output formats for further use in Greenland research [@how-two-boom-2022; @how-one-boom-2022]. The AWS data products have been used in high impact studies [@macguth-greenland-2016; @oehri_vegetation_2022; @box-greenland-2022], and have been crucial for evaluating the effect of climate change on land ice in annual reports such as the Arctic Report Card and "State of the Climate" [@moon-greenland-2022a; @moon-greenland-2022b]. 


# Documentation

`pypromice` versions accompany releases of PROMICE AWS one-boom and two-boom data publications [@how-pypromice-2022].

Package documentation is available on the `pypromice` [readthedocs](https://pypromice.readthedocs.io/en/latest/). 

Guides for general PROMICE AWS processing operations are included at the [GEUS Glaciology and Climate GitHub pages](https://geus-glaciology-and-climate.github.io/).


# Acknowledgements

This work is funded through the Danish Ministry of Climate, Energy and Utilities via The Programme for Monitoring of the Greenland Ice Sheet (PROMICE) and the Greenland Climate Network (GC-Net). 


# References

<!--  A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline.	-->
