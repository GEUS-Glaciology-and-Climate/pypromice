*************
Quick install
*************
The latest release of pypromice can installed using conda or pip:

.. code:: console

	$ conda install pypromice -c conda-forge
	
	
.. code:: console

	$ pip install pypromice


The `eccodes <https://confluence.ecmwf.int/display/ECC/ecCodes+installation>`_ package for pypromice's post-processing functionality needs to be installed specifically in the pip distribution:

.. code:: console

	$ conda install eccodes -c conda-forge
	$ pip install pypromice


And for the most up-to-date version of pypromice, the package can be cloned and installed directly from the repo: 

.. code:: console

	$ pip install --upgrade git+http://github.com/GEUS-Glaciology-and-Climate/pypromice.git

*****************
Developer install
*****************

pypromice can be ran in an environment with the pypromice package forked or cloned from the `GitHub repo <https://github.com/GEUS-Glaciology-and-Climate/pypromice>`_. 

.. code:: console

	$ conda create --name pypromice python=3.11
	$ conda activate pypromice
	$ git clone git@github.com:GEUS-Glaciology-and-Climate/pypromice.git
	$ cd pypromice/
	$ pip install .

pypromice is also provided with a `conda environment configuration file <https://github.com/GEUS-Glaciology-and-Climate/pypromice/blob/main/environment.yml>`_ for a more straightforward set-up, if needed:

.. code:: console

	$ conda env create --file environment.yml -n pypromice

The package has inbuilt unit tests, which can be run to test the package installation:

.. code:: console

	$ python -m unittest discover pypromice
        
.. note::

	This command line unit testing only works if pypromice is installed in the active Python environment. Unit testing can be run directly from the cloned pypromice top directory also either by running each script or from the command line as so: ``$ python -m unittest discover pypromice``

