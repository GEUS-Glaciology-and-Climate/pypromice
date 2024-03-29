pypromice
=========

`pypromice <https://github.com/GEUS-Glaciology-and-Climate/pypromice>`_ is designed for processing and handling `PROMICE <https://promice.dk>`_ and `GC-Net <http://cires1.colorado.edu/steffen/gcnet/>`_ automated weather station (AWS) data. The PROMICE (Programme for Monitoring of the Greenland Ice Sheet) weather station network monitors glacier mass balance in the melt zone of the Greenland Ice Sheet, providing ground truth data to calibrate mass budget models. GC-Net (Greenland Climate Network) weather stations measure snowfall and surface properties in the accumulation zone, providing valuable knowledge on the Greenland Ice Sheet's mass gain and climatology.

The PROMICE and GC-Net monitoring networks have unique AWS configurations and provide specialized data, therefore a toolbox is needed to handle and process their data. pypromice is the go-to toolbox for handling and processing climate and glacier datasets from the PROMICE and GC-Net monitoring networks. New releases of pypromice are uploaded alongside PROMICE AWS data releases to the `GEUS Dataverse <https://dataverse.geus.dk/dataverse/AWS>`_ for transparency purposes and to encourage collaboration on improving our data.

If you intend to use PROMICE and GC-Net AWS data and/or pypromice in your work, please cite these publications below, along with any other applicable PROMICE publications where possible:

Fausto, R.S., van As, D., Mankoff, K.D., Vandecrux, B., Citterio, M., Ahlstrøm, A.P., Andersen, S.B., Colgan, W., Karlsson, N.B., Kjeldsen, K.K., Korsgaard, N.J., Larsen, S.H., Nielsen, S., Pedersen, A.Ø., Shields, C.L., Solgaard, A.M., and Box, J.E. (2021) Programme for Monitoring of the Greenland Ice Sheet (PROMICE) automatic weather station data, Earth Syst. Sci. Data, 13, 3819–3845, `https://doi.org/10.5194/essd-13-3819-2021 <https://doi.org/10.5194/essd-13-3819-2021>`_

How, P., Wright, P.J., Mankoff, K., Vandecrux, B., Fausto, R.S. and Ahlstrøm, A.P. (2023) pypromice: A Python package for processing automated weather station data, Journal of Open Source Software, 8(86), 5298, `https://doi.org/10.21105/joss.05298 <https://doi.org/10.21105/joss.05298>`_

How, P., Lund, M.C., Nielsen, R.B., Ahlstrøm, A.P., Fausto, R.S., Larsen, S.H., Mankoff, K.D., Vandecrux, B., Wright, P.J. (2023) pypromice, GEUS Dataverse, `https://doi.org/10.22008/FK2/3TSBF0 <https://doi.org/10.22008/FK2/3TSBF0>`_


.. toctree::
   :maxdepth: 2
   :caption: Getting started
   
   install

.. toctree::
   :maxdepth: 2
   :caption: Guides
   
   guide_user
   guide_developer

.. toctree::
   :maxdepth: 2
   :caption: Technical info
   
   technical_data
   technical_process
   
.. toctree::
   :maxdepth: 1
   :caption: Modules
   
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

