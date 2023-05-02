# IMPORT
from snowflake.snowpark.session import Session
import pandas as pd
import json

# LOAD CREDENTIALS.JSON FILE
connection_parameters = json.load(open('credentials.json'))

# CONNECT TO SNOWFLAKE
session = Session.builder.configs(connection_parameters).create()

# CREATE COMPUTE WAREHOUSE
session.sql('create or replace warehouse nlp_wh with warehouse_size = small;').collect()

# SET WAREHOUSE CONTEXT
session.use_warehouse('nlp_wh')

# CREATE DATABASE
session.sql('create or replace database nlp_db;').collect()

# SET SCHEMA CONTEXT
session.use_schema('nlp_db.public')

# LOAD TEST DATASET
test_dataset = pd.DataFrame(pd.read_csv('Dataset/test_dataset.csv'))

# CREATE SNOWFLAKE TEST TABLE
session.write_pandas(test_dataset, 'test_dataset', auto_create_table=True, quote_identifiers=False)

# LOAD TEST DATASET
train_dataset = pd.DataFrame(pd.read_csv('Dataset/train_dataset.csv'))

# CREATE SNOWFLAKE TEST TABLE
session.write_pandas(train_dataset, 'train_dataset', auto_create_table=True, quote_identifiers=False)