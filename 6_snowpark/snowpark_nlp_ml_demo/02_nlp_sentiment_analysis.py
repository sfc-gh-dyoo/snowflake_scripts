# IMPORT
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import udf, sproc, col
from snowflake.snowpark.types import IntegerType, FloatType, StringType, BooleanType, Variant
from snowflake.snowpark import functions as fn
from snowflake.snowpark import version
import numpy as np
import pandas as pd
import json

# LOAD CREDENTIALS.JSON FILE
connection_parameters = json.load(open('credentials.json'))

# CONNECT TO SNOWFLAKE
session = Session.builder.configs(connection_parameters).create()

# Import the needed packageinto the Snowpark session
session.add_packages("snowflake-snowpark-python")
session.add_packages("scikit-learn", "pandas", "numpy", "nltk", "joblib", "cachetools")

