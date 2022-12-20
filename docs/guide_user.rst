**********
User guide
**********

Loading PROMICE data
====================

Import from Dataverse (no downloads!)
-------------------------------------
The automated weather station (AWS) datasets are openly available on our Dataverse_. These can be imported directly with pypromice, with no downloading required.

.. code:: python

	import pypromice.get as pget

	# Import AWS data from station KPC_U
	df = pget.aws_data("KPC_U")

All available AWS datasets are retrieved by station name. Use ``aws_names()`` to list all station names which can be used as an input to ``aws_data()``.

.. code:: python

	n = pget.aws_names()
	print(n)

.. _Dataverse: https://dataverse.geus.dk/dataverse/AWS


Download with pypromice
-----------------------
AWS data can be downloaded to file with pypromice. Open up a CLI and use the ``getData`` command.

.. code:: console

	$ getData -n KPC_U

Files are downloaded to the current directory as a CSV formatted file. Use the ``-h`` help flag to explore further input variables.
 
.. code:: console

	$ getData -h

.. note::

	Currently, this functionality within pypromice is only for our hourly AWS data. For daily and monthly AWS data, please download these from the Dataverse_.
	
	
Load from NetCDF file
---------------------
AWS data can be loaded from a local NetCDF file with ``xarray``.

.. code:: python

	import xarray as xr
	ds = xr.open_dataset("KPC_U_hour.nc")


Load from CSV file
------------------

AWS data can be loaded from a local CSV file and handled as a ``pandas.DataFrame``.

.. code:: python

	import pandas as pd
	df = pd.read_csv("KPC_U_hour.csv", index_col=0, parse_dates=True)

If you would rather handle the AWS data as an ``xarray.Dataset`` object then the ``pandas.DataFrame`` can be converted.

.. code:: python

	ds = xr.Dataset.from_dataframe(df) 

