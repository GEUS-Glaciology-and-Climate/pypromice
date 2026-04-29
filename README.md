# pypromice
[![PyPI version](https://badge.fury.io/py/pypromice.svg)](https://badge.fury.io/py/pypromice) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/pypromice/badges/version.svg)](https://anaconda.org/conda-forge/pypromice) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/pypromice/badges/platforms.svg)](https://anaconda.org/conda-forge/pypromice) [![](<https://img.shields.io/badge/Dataverse DOI-10.22008/FK2/3TSBF0-orange>)](https://www.doi.org/10.22008/FK2/3TSBF0) [![DOI](https://joss.theoj.org/papers/10.21105/joss.05298/status.svg)](https://doi.org/10.21105/joss.05298) [![Documentation Status](https://readthedocs.org/projects/pypromice/badge/?version=latest)](https://pypromice.readthedocs.io/en/latest/?badge=latest)
 
Pypromice is designed for processing and handling [PROMICE|GC-Net](https://promice.dk) automated weather station (AWS) data.

Both the source code of pypromice and the PROMICE|GC-Net AWS data it produces are released on [Dataverse](https://dataverse.geus.dk/dataverse/PROMICE). Please visit the pypromice [readthedocs](https://pypromice.readthedocs.io/en/latest/?badge=latest) and the following publications for more information. 

If you intend to use PROMICE|GC-Net AWS data and/or pypromice in your work, please cite these publications below, along with any other applicable PROMICE publications where possible:

**Fausto, R. S., How, P., Vandecrux, B., Lund, M. C., Box, J. E., Mankoff, K. D., Andersen, S. B., van As, D., Bahbah, R., Citterio, M., Colgan, W., Jakobsgaard, H. T., Karlsson, N. B., Kjeldsen, K. K., Larsen, S. H., Olsen, C., Oraschewski, F. M., Rutishauser, A., Shields, C. L., Solgaard, A. M., Stevens, I. T., Svendsen, S. H., Langley, K., Messerli, A., Bjørk, A. A., Andersen, J. K., Abermann, J., Steiner, J., Prinz, R., Hynek, B., Lea, J. M., Brough, S., and Ahlstrøm, A. P.: PROMICE | GC-NET automatic weather station data, Earth Syst. Sci. Data, 18, 2829–2873, https://doi.org/10.5194/essd-18-2829-2026, 2026. **

**Fausto, R.S., van As, D., Mankoff, K.D., Vandecrux, B., Citterio, M., Ahlstrøm, A.P., Andersen, S.B., Colgan, W., Karlsson, N.B., Kjeldsen, K.K., Korsgaard, N.J., Larsen, S.H., Nielsen, S., Pedersen, A.Ø., Shields, C.L., Solgaard, A.M., and Box, J.E. (2021) Programme for Monitoring of the Greenland Ice Sheet (PROMICE) automatic weather station data, Earth Syst. Sci. Data, 13, 3819–3845, [https://doi.org/10.5194/essd-13-3819-2021](https://doi.org/10.5194/essd-13-3819-2021)**

**How, P., Wright, P.J., Mankoff, K., Vandecrux, B., Fausto, R.S. and Ahlstrøm, A.P. (2023) pypromice: A Python package for processing automated weather station data, Journal of Open Source Software, 8(86), 5298, [https://doi.org/10.21105/joss.05298](https://doi.org/10.21105/joss.05298)** 

**How, P., Lund, M.C., Nielsen, R.B., Ahlstrøm, A.P., Fausto, R.S., Larsen, S.H., Mankoff, K.D., Vandecrux, B., Wright, P.J. (2023) pypromice, GEUS Dataverse, [https://doi.org/10.22008/FK2/3TSBF0](https://doi.org/10.22008/FK2/3TSBF0)** 

## Installation

### Quick install

The latest release of pypromice can installed using conda or pip:

```
$ conda install pypromice -c conda-forge
```

```
$ pip install pypromice
```

The [eccodes](https://confluence.ecmwf.int/display/ECC/ecCodes+installation) package for pypromice's post-processing functionality needs to be installed specifically in the pip distribution:

```
$ conda install eccodes -c conda-forge
$ pip install pypromice
```

And for the most up-to-date version of pypromice, the package can be cloned and installed directly from the repo: 

```
$ pip install --upgrade git+http://github.com/GEUS-Glaciology-and-Climate/pypromice.git
```

### Developer install
	
pypromice can be ran in an environment with the pypromice repo:

```
$ conda create --name pypromice python=3.8
$ conda activate pypromice
$ git clone git@github.com:GEUS-Glaciology-and-Climate/pypromice.git
$ cd pypromice/
$ pip install .
```

