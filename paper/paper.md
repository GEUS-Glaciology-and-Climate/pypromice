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
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Kenneth D. Mankoff
    orcid: 0000-0001-5453-2019
    equal-contrib: true 
    affiliation: "1, 2"
  - name: Patrick J. Wright
    orcid: 0000-0003-2999-9076
    affiliation: 1
    equal-contrib: true 
  - name: Baptiste Vandecrux
    orcid: 0000-0002-4169-8973
    affiliation: 1
    equal-contrib: true
  - name: Robert S. Fausto
    orcid: 0000-0003-1317-8185
    affiliation: 1
    equal-contrib: true
affiliations:
 - name: Department of Glaciology and Climate, Geological Survey of Denmark and Greenland (GEUS), Denmark
   index: 1
 - name: National Snow and Ice Data Center (NSIDC), USA
   index: 2

date: 01 March 2023
bibliography: paper.bib

---

# Summary

<!--   A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.	-->

The `pypromice` Python package is for processing and handling observation datasets from automated weather stations (AWS). It is primarily aimed at users of AWS data from PROMICE (The Programme for Monitoring of the Greenland Ice Sheet), which is a valuable set of in situ observations to the cryospheric science research community (REFS). Functionality is primarily handled in `pypromice` using two key open-source Python packages, `xarray` [@hoyer-xarray-2017] and `pandas` [@pandas-decpandas-2020].

A defined processing workflow is included in `pypromice` for transforming original AWS observations (Level 0, `L0`) to a usable, CF-convention-compliant dataset (Level 3, `L3`) (\autoref{fig:process}). Intermidiary processing levels (`L1`,`L2`) refer to key stages in the workflow, namely the conversion of variables to physical measurements and variable filtering (`L1`), cross-variable corrections (`L2`), and derived variables (`L3`). Information regarding the station configuration is needed to perform the processing, such as instrument calibration coefficients and station type (one-boom or two-boom station design, for example), which are held in a `toml` configuration file. Two example configuration files are provided with `pypromice`, which are also used in the package's unit tests. More detailed documentation of the AWS design, instrumentation, and processing steps are described in [@fausto-programme-2021].

![AWS data Level 0 (`L0`) to Level 3 (`L3`) processing steps, where `L0` refers to raw, original data and `L3` is usable data that has been transformed, corrected and filtered \label{fig:process}](https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/geus-glaciology-and-climate.github.io/master/assets/images/aws_workflow_raw.png){ width=75% }

`L0` data is either collected from an AWS during a station visit or is transmitted in near-real-time from each PROMICE AWS via the Iridium Short Burst Data (SBD) service. An object-oriented workflow for fetching and decoding SBD messages to Level 0 data (`L0 tx`) is included in `pypromice` (\autoref{fig:tx}). Alongside the processing module, this workflow has been deployed for operational uses in PROMICE to produce `L3` AWS data in near-real-time. A post-processing workflow is also included to demonstrate how near-real-time AWS data is treated after `L3` for submission to global weather forecasting models under the World Meteorological Organisation (WMO).

![Object-oriented workflow in `pypromice.tx` for fetching and decoding AWS transmission messages to Level 0 (`L0 tx`) data \label{fig:tx}](https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/geus-glaciology-and-climate.github.io/master/assets/images/aws_tx_design.png){ width=75% }

A minor component of `pypromice` is the `get` module, which is primarily for AWS data users to access PROMICE data in an easy manner. This functionality includes options to pipe AWS data directly into their Python console with no need for downloads. It is intended to further develop this part of `pypromice` to enable user-accessibility to PROMICE AWS datasets, subject to user demand. Aspects of this expansion could be data transformation and visualisation. PROMICE and GC-Net AWS dataset handling is also available as part of the [JAWS](https://github.com/jaws/jaws) Python package [@zender-jaws-2019], however it does not appear to be maintained routinely and therefore the demand for expansion of `pypromice.get` is expected.


# Statement of need

<!--   A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work.  -->

`pypromice` has four main research purposes:

1. Process and handle AWS observations  
2. Document the PROMICE and GC-Net AWS processing with transparency and reproducibility
3. Supply easy and accessible methods to handle AWS data
4. Provide opportunities to contribute to the processing and handling of AWS data in an open and collaborative manner


# Usage

<!--    Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it. -->

The following sections outline ongoing research projects which actively use `pypromice`.

## The Programme for the Monitoring of the Greenland Ice Sheet (PROMICE)

The Programme for Monitoring of the Greenland Ice Sheet ([PROMICE](https://promice.org)) is a network of over 30 weather stations installed across the Greenland Ice Sheet. Launched in 2007, these weather stations were a one-boom tripod fixture, designed to sit on top of the ice and move with the ice flow at the ice sheet periphery. `pypromice` is used operationally for its processing of one-boom AWS observations [@how-one-boom-2022].

## Greenland Climate Network (GC-Net)

In 2020, PROMICE assumed responsibility of the Greenland Climate Network ([GC-Net]) and expanded their operations with the installation of a new two-boom mast station design intended for monitoring conditions on the interior ice sheet. `pypromice` is used operationally for its processing of two-boom AWS observations [@how-two-boom-2022].

## Greenland Ecosystem Monitoring (GEM)

The Greenland Ecosystem Monitoring programme ([GEM](https://g-e-m.dk/)) is an integrated, long-term monitoring effort on ecosystems and the effects of climate change in the Arctic, which was established in 1995 and includes monitoring around Zackenberg (1995-), Kobbefjord in Nuuk (2007-) and Disko (2017-). It is host to over 1000 freely-available environmental datasets, including AWS data from four stations. One of these stations is located at a small mountain glacier in Kobbefjord, and three are located in a tight network on the A. P. Olsen Ice Cap in Zackenberg. `pypromice` is used to process observations from these weather stations routinely [@abermann-strong-2019; @gem-glaciobasis-2020; @messerli-snow-2022].


# Documentation

`pypromice` versions accompany releases of PROMICE AWS one-boom and two-boom data publications [@how-pypromice-2022].

Package documentation is available on the `pypromice` [readthedocs](https://pypromice.readthedocs.io/en/latest/). 

Guides for general PROMICE AWS processing operations are included at the [GEUS Glaciology and Climate GitHub pages](https://geus-glaciology-and-climate.github.io/).


# Acknowledgements

This work is funded through the Danish Ministry of Climate, Energy and Utilities via The Programme for Monitoring of the Greenland Ice Sheet (PROMICE) and the Greenland Climate Network (GC-Net). 


# References

<!--  A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline.	-->
