# IMPORT
from snowflake.snowpark.session import Session
import pandas as pd
import json

# LOAD CREDENTIALS.JSON FILE
connection_parameters = json.load(open('credentials.json'))

# CONNECT TO SNOWFLAKE
session = Session.builder.configs(connection_parameters).create()

# CREATE COMPUTE WAREHOUSE
session.sql('create or replace warehouse snowpark_wh with warehouse_size = x3large;').collect()

# CREATE SNOWPARK OPTIMIZED WAREHOUSE
session.sql('create or replace warehouse snowpark_opt_wh with warehouse_size = medium warehouse_type = "snowpark-optimized" max_concurrency_level = 1;').collect()

# CREATE TPC DATABASE
session.sql('create database if not exists tpc_db from share sfc_samples.sample_data;').collect()

# CREATE DEMO DATABASE
session.sql('create or replace database ltv_db;').collect()

# SET SCHEMA CONTEXT
session.use_schema('ltv_db.public')

# LOAD TEST DATASET
session.sql('create or replace stage ml_models;').collect()

# SUSPEND WAREHOUSES
session.sql('alter warehouse snowpark_wh suspend;').collect()
session.sql('alter warehouse snowpark_opt_wh suspend;').collect()