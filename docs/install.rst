*************
Quick install
*************
pypromice can installed using pip:

.. code:: console

	$ pip install --upgrade git+http://github.com/GEUS-Glaciology-and-Climate/pypromice.git

*****************
Developer install
*****************

pypromice can be ran in an environment with the specified dependencies.

.. code:: console

	$ conda create --name pypromice python=3.8
	$ conda activate pypromice
	$ conda install xarray pandas pathlib
	$ conda install -c conda-forge netCDF4

With the pypromice package cloned from GitHub_. 

.. code:: console

	$ git clone git@github.com:GEUS-Glaciology-and-Climate/pypromice.git

pypromice is also provided with a conda environment configuration environment.yml_ for a more straightforward set-up, if needed:

.. code:: console

	$ conda env create --file environment.yml -n pypromice
	
.. _GitHub: https://github.com/GEUS-Glaciology-and-Climate/pypromice
.. _environment.yml: https://github.com/GEUS-Glaciology-and-Climate/pypromice/blob/main/environment.yml


***********************
Additional dependencies
***********************

Additional packages are required if you wish to use pypromice's post-processing functionality. 


scikit-learn
------------
scikit-learn_ can installed with conda-forge.

.. code:: console

	$ conda install -c conda-forge scikit-learn

Or with pip. 

.. code:: console

	$ pip install scikit-learn 

.. _scikit-learn: https://scikit-learn.org/stable/


eccodes
-------
eccodes_ is the official package for BUFR encoding and decoding. Try firstly to install with conda-forge like so:

.. code:: console

	$ conda install -c conda-forge eccodes

.. note::

	If the environment cannot resolve the eccodes installation then follow the steps documented here_ to download eccodes and then install eccodes' python bindings using pip: ``pip3 install eccodes-python``

.. _eccodes: https://confluence.ecmwf.int/display/ECC/ecCodes+installation
.. _here: https://gist.github.com/MHBalsmeier/a01ad4e07ecf467c90fad2ac7719844a
