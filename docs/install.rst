*************
Quick install
*************
The latest release of pypromice can installed using pip or conda-forge:

.. code:: console

	$ pip install pypromice


.. code:: console

	$ conda install pypromice -c conda-forge


For the most up-to-date version, pypromice can be installed directly from the repo: 

.. code:: console

	$ pip install --upgrade git+http://github.com/GEUS-Glaciology-and-Climate/pypromice.git

*****************
Developer install
*****************

pypromice can be ran in an environment with the pypromice package forked or cloned from the `GitHub repo <https://github.com/GEUS-Glaciology-and-Climate/pypromice>`_. 

.. code:: console

	$ conda create --name pypromice python=3.8
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

	This command line unit testing only works if pypromice is installed in the active Python environment. Unit testing can be run directly from the cloned pypromice top directory also either by running each script or from the command line as so: ``$ python -m unittest discover src/pypromice``

***********************
Additional dependencies
***********************

Additional packages are required if you wish to use pypromice's post-processing functionality. 

`eccodes <https://confluence.ecmwf.int/display/ECC/ecCodes+installation>`_ is the official package for BUFR encoding and decoding, which pypromice uses for post-process formatting. Try firstly to install with conda-forge like so:

.. code:: console

	$ conda install -c conda-forge eccodes

.. note::

	If the environment cannot resolve the eccodes installation then follow the steps documented `here <https://gist.github.com/MHBalsmeier/a01ad4e07ecf467c90fad2ac7719844a>`_ to download eccodes and then install eccodes' python bindings using pip: ``pip3 install eccodes-python``
