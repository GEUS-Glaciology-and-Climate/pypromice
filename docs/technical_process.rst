**********************************  
Transmission to Level 0 processing
**********************************

pypromice's ``tx`` module contains functionality for processing data transmissions from an AWS to a Level 0 data product:

- Fetching payload messages from the Iridium SBD service
- Payload message decoding from binary
- Level 0 data compiling 

  
Payload handling
================

``SbdMessage`` handles the SBD message, either taken from an ``email.message.Message`` object or a .sbd file.

``EmailMessage`` handles the email message (that the SBD message is attached to) to parse information such as sender, subject, date, and to check for attachments.


Payload decoder
===============

``PayloadFormat`` handles the message types and decoding templates. These can be imported from file, with two default CSV files provided with pypromice - payload_formatter.csv_ and payload_type.csv_.

.. _payload_formatter.csv: https://github.com/GEUS-Glaciology-and-Climate/pypromice/blob/main/src/pypromice/tx/payload_formats.csv
.. _payload_type.csv: https://github.com/GEUS-Glaciology-and-Climate/pypromice/blob/main/src/pypromice/tx/payload_types.csv


Payload processing
==================

``L0tx`` handles the processing and output of the L0 transmission dataline. This object inherits from ``EmailMessage`` and ``PayloadFormat`` to read and decode messages

To reprocess old messages, these can be retrieved from the mailbox by rolling back the ``uid`` counter or by reading from .sbd file.

The following function can be executed from a CLI to fetch ``L0`` transmission messages from all valid stations:

.. code:: console
	
	$ getL0tx -a accounts.ini -p credentials.ini -c tx/config 
	-u last_aws_uid.ini -o tx

.. note::

	Credentials are needed to access our AWS transmissions. We do not provide these to external users, so purely include this workflow in pypromice to demonstrate transparency in our processing.

*****************************
Level 0 to Level 3 processing
*****************************

The ``process`` module is for processing AWS observations from Level 0 to Level 3 (end-user product) data products.

To process from L0>>L3, the following function can be executed in a CLI.

.. code:: console
	
	$ getL3 -c config/KPC_L.toml -i . -o ../../aws-l3/tx"

And in parallel through all configuration .toml files ``$imei_list``

.. code:: console

	$ parallel --bar "getL3 -c ./{} -i . -o ../../aws-l3/tx" ::: $(ls $imei_list)


Station configuration
=====================

Each Level 0 file that will be processed must have an entry in the TOML-formatted configuration file. The config file can be located anywhere, and the processing script receives the config file and the location of the Level 0 data.

.. code:: python

	# Station configuration example, NSE
	
	station_id = 'NSE'
	logger_type = 'CR1000X'
	number_of_booms = 2
	nodata = ['-999', 'NAN'] 
	modem = [['300534062416350', '2021-06-19 12:00:00']]  #Formatting [[modem,start,end],[modem,start,end]]

	['NSE_300534062416350_1.txt']
	format     = 'TX'
	skiprows = 0
	latitude =  66.48
	longitude = 42.49

	dsr_eng_coef = 12.66
	usr_eng_coef = 13.90
	dlr_eng_coef = 8.55
	ulr_eng_coef = 11.26
	tilt_y_factor = -1 

	columns = ['time','rec','p_l','p_u','t_l','t_u','rh_l','rh_u',
		  'wspd_l','wdir_l','wspd_u','wdir_u','dsr',
 		  'usr','dlr','ulr','t_rad','z_boom_l','z_boom_u',
 		  't_i_1','t_i_2','t_i_3','t_i_4','t_i_5','t_i_6','t_i_7',
 	 	  't_i_8','t_i_9','t_i_10','t_i_11','tilt_x','tilt_y','rot',
 		  'precip_l','precip_u','gps_time','gps_lat','gps_lon',
 		  'gps_alt','gps_hdop','fan_dc_l','fan_dc_u','batt_v', 'p_i',
 		  't_i','rh_i','wspd_i','wdir_i','msg_i']


The TOML config file has the following expectations and behaviors:

- Properties can be defined at the top level or under a section
- Each file that will be processed gets its own section
- Properties at the top level are copied to each section (assumed to apply to all files)
- Top-level properties are overridden by file-level properties if they exist in both locations

.. note::

	Be aware the column names should follow those defined in the variables look-up table found here_. Any column names provided that are not in this look-up table will be passed through the processing untouched.

.. _here: https://github.com/GEUS-Glaciology-and-Climate/pypromice/blob/main/src/pypromice/process/variables.csv
