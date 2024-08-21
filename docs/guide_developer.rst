***************
Developer guide
***************

Contributions, bug reports and fixes, documentation improvements, enhancements and ideas are welcome. A good starting place to look at is:

1. `PROMICE-AWS-data-issues <https://github.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-data-issues>_`, where we report suspicious or incorrect data
2. pypromice's `GitHub Issues <https://github.com/GEUS-Glaciology-and-Climate/pypromice/issues>`_, for an overview of known bugs, developments and ideas


Data reports
============

Automatic weather station (AWS) data from the Greenland Ice Sheet are often imperfect due to the complexity and conditions involved in installing and maintaining the AWS. 

If you are using our AWS data and something seems suspicious or erroneous, you can check the `PROMICE-AWS-data-issues <https://github.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-data-issues>`_ space to see if has previously been flagged and/or fixed. If not, then please follow the conventions stated in the repository and open an issue.

.. note::

	Data visualisations best demonstrate data problems and are greatly appreciated in solving data issues. If you are unsure, see examples of our closed issues in `PROMICE-AWS-data-issues <https://github.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-data-issues>`_ 


Bug reports and enhancement requests
====================================

Bug reports are essential to improving the stability and usability of pypromice. These should be raised on pypromice's `GitHub Issues <https://github.com/GEUS-Glaciology-and-Climate/pypromice/issues>`_. A complete and reproducible report is essential for bugs to be resolved easily, therefore bug reports must:

1. Include a concise and self-contained Python snippet reproducing the problem. For example:

.. code:: python

	df = pget.aws_data(...)


2. Include a description of how your pypromice configuration is set up, such as from pip install and repository cloning/forking. If installed from pip or locally built, you can find the version of pypromice you are using with the following function.

.. code:: python
	
	from importlib import metadata
	print(metadata.version('pypromice'))


3. Explain why the current behaviour is wrong or not desired, and what you expect instead

.. note:: 

	Before submitting an issue, please make sure that your installation is correct and working from either the pip installation or the `main <https://github.com/GEUS-Glaciology-and-Climate/pypromice/tree/main>`_ branch of the pypromice repository.


Contributing to pypromice
=========================

You can work directly with pypromice's development if you have a contribution, such as a solution to an issue or a suggestion for an enhancment. 


Forking 
-------

In order to contribute, you will need your own fork of the pypromice GitHub repository to work on the code. Go to the `repo <https://github.com/GEUS-Glaciology-and-Climate/pypromice>`_ and choose the ``Fork`` option. This now creates a copy in your own GitHub space, which is connected to the upstream pypromice repository.


Creating a development branch
-----------------------------

From your forked space, make sure you have a Python Environment for running pypromice, as described in :ref:`Developer install`. Then create and checkout a branch to make your developments on.

.. code:: console

	$ git checkout -b my-dev-branch

Keep changes in this branch specific to one bug or enhancement, so it is clear how this branch contributes to pypromice. 


Creating a pull request
-----------------------

To contribute your changes to pypromice, you need to make a pull request from your forked development branch to pypromice's develop branch. The develop branch is our staging for operational testing before it is deployed to our live processing in the main branch. Before doing so, retrieve the most recent version to keep your branch up to date with pypromice's develop branch.

.. code:: console

	$ git fetch
	$ git merge upstream/develop

And then open a pull request as documented `here <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_. Make sure to include the following in your pull request description:

1. The aim of your changes
2. Details of what these changes are
3. Any limitations or further development needed

Your pull request will be reviewed and, if valid and suitable, will be accepted. Following this, you will be listed as a contributor to pypromice!
