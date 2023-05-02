# Snowpark Python - TPC DS  - Customer Lifetime Value

This demo utilizes the [TPC DS sample](https://docs.snowflake.com/en/user-guide/sample-data-tpcds.html) dataset that is made available via  Snowflake share. It can be configured to run on either the 10 TB or the 100 TB version of the dataset. 

This illustrates how to utilize Snowpark for feature engineering, training, and inference to answer a common question for retailers: What is the value of a customer across all sales channels? 

### Setup 

The TPC DS data is available already to you in your Snowflake account as shared database utlizing Snowflake's data sharing. This means you as the user will never incur the costs of storing this large dataset. 

 1. Edit the *creds.json* file to with your Snowfalke account name, user name, and password to connect to your account. 
 2. Run all the script in the 01_environmental_setup.sql
 3. Run python 02_feature_engineering.py in terminal

### Cost Performance

Below is a table of some observed performance stats I have observed in AWS US East Ohio. All times reported in seconds and assuming enterprise edition list pricing. 

| Dataset       	| Eng/Prep Warehouse 	| Training (Opt) Warehouse 	| Time for feature eng/prep 	| Cost for feature eng/prep 	| Time for training 	| Cost for training 	| Time for inference 	| Cost for inference 	|
|---------------	|----------------------	|----------------------------	|---------------------------	|---------------------------	|-------------------	|-------------------	|--------------------	|--------------------	|
| TPC-DS 10 TB  	| 3XL                   | Medium                      | 60                        	| $3.20                     	| 1400.4            	| $7.07             	| 9.8                	| $0.52              	|
| TPC-DS 100 TB 	| 3XL                   | Medium                      | 311.6                     	| $16.51                    	| 2210              	| $11.05            	| 24.6               	| $1.30              	|
