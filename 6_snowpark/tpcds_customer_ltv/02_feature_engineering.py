# “I want to be able to predict the life-time value of my customers across all sales channels”.

# IMPORT
import snowflake.snowpark
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark import version as v
import json

# LOAD CREDENTIALS.JSON FILE
connection_parameters = json.load(open('credentials.json'))

# CONNECT TO SNOWFLAKE
session = Session.builder.configs(connection_parameters).create()

# SET WAREHOUSE CONTEXT
session.use_warehouse('snowpark_wh')

# SELECT EITHER 100 or 10 FOR THE TPC.DS DATASET 
TPCDS_SIZE_PARAM = 10
TPC_DB = 'TPC_DB' # Name of Snowflake TPC Database might be different...

if TPCDS_SIZE_PARAM == 10: 
    TPCDS_SCHEMA = 'TPCDS_SF10TCL'
elif TPCDS_SIZE_PARAM == 100:
    TPCDS_SCHEMA = 'TPCDS_SF100TCL'
else:
    raise ValueError("Invalid TPCDS_SIZE_PARAM selection")
    
# CREATE DATAFRAMES
store_sales = session.table(f'{TPC_DB}.{TPCDS_SCHEMA}.store_sales')
catalog_sales = session.table(f'{TPC_DB}.{TPCDS_SCHEMA}.catalog_sales') 
web_sales = session.table(f'{TPC_DB}.{TPCDS_SCHEMA}.web_sales') 
date = session.table(f'{TPC_DB}.{TPCDS_SCHEMA}.date_dim')
dim_stores = session.table(f'{TPC_DB}.{TPCDS_SCHEMA}.store')
customer = session.table(f'{TPC_DB}.{TPCDS_SCHEMA}.customer')
address = session.table(f'{TPC_DB}.{TPCDS_SCHEMA}.customer_address')
demo = session.table(f'{TPC_DB}.{TPCDS_SCHEMA}.customer_demographics')

# AGGREGATE THE THREE SALES TABLES
store_sales_agged = store_sales.group_by('ss_customer_sk').agg(F.sum('ss_sales_price').as_('total_sales'))
web_sales_agged = web_sales.group_by('ws_bill_customer_sk').agg(F.sum('ws_sales_price').as_('total_sales'))
catalog_sales_agged = catalog_sales.group_by('cs_bill_customer_sk').agg(F.sum('cs_sales_price').as_('total_sales'))
store_sales_agged = store_sales_agged.rename('ss_customer_sk', 'customer_sk')
web_sales_agged = web_sales_agged.rename('ws_bill_customer_sk', 'customer_sk')
catalog_sales_agged = catalog_sales_agged.rename('cs_bill_customer_sk', 'customer_sk')

total_sales = store_sales_agged.union_all(web_sales_agged)
total_sales = total_sales.union_all(catalog_sales_agged)

total_sales = total_sales.group_by('customer_sk').agg(F.sum('total_sales').as_('total_sales'))

customer = customer.select('c_customer_sk','c_current_hdemo_sk', 'c_current_addr_sk', 'c_customer_id', 'c_birth_year')

customer = customer.join(address.select('ca_address_sk', 'ca_zip'), customer['c_current_addr_sk'] == address['ca_address_sk'] )
customer = customer.join(demo.select('cd_demo_sk', 'cd_gender', 'cd_marital_status', 'cd_credit_rating', 'cd_education_status', 'cd_dep_count'),
                                customer['c_current_hdemo_sk'] == demo['cd_demo_sk'] )
customer = customer.rename('c_customer_sk', 'customer_sk')
customer.show()

final_df = total_sales.join(customer, 'customer_sk')

session.use_schema('ltv_db.public')
final_df.write.mode('overwrite').save_as_table('feature_store')

session.add_packages('snowflake-snowpark-python', 'scikit-learn', 'pandas', 'numpy', 'joblib', 'cachetools', 'xgboost', 'joblib')

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import joblib
import os

def train_model(session: snowflake.snowpark.Session) -> float:
    snowdf = session.table("feature_store")
    snowdf = snowdf.drop(['CUSTOMER_SK', 'C_CURRENT_HDEMO_SK', 'C_CURRENT_ADDR_SK', 'C_CUSTOMER_ID', 'CA_ADDRESS_SK', 'CD_DEMO_SK'])
    snowdf_train, snowdf_test = snowdf.random_split([0.8, 0.2], seed=82) 

    # save the train and test sets as time stamped tables in Snowflake 
    snowdf_train.write.mode("overwrite").save_as_table("snowpark_db.public.tpc_TRAIN")
    snowdf_test.write.mode("overwrite").save_as_table("snowpark_db.public.tpc_TEST")
    train_x = snowdf_train.drop("TOTAL_SALES").to_pandas() # drop labels for training set
    train_y = snowdf_train.select("TOTAL_SALES").to_pandas()
    test_x = snowdf_test.drop("TOTAL_SALES").to_pandas()
    test_y = snowdf_test.select("TOTAL_SALES").to_pandas()
    cat_cols = ['CA_ZIP', 'CD_GENDER', 'CD_MARITAL_STATUS', 'CD_CREDIT_RATING', 'CD_EDUCATION_STATUS']
    num_cols = ['C_BIRTH_YEAR', 'CD_DEP_COUNT']

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])

    preprocessor = ColumnTransformer(
    transformers=[('num', num_pipeline, num_cols),
                  ('encoder', OneHotEncoder(handle_unknown="ignore"), cat_cols) ])

    pipe = Pipeline([('preprocessor', preprocessor), 
                        ('xgboost', XGBRegressor())])
    pipe.fit(train_x, train_y)

    test_preds = pipe.predict(test_x)
    rmse = mean_squared_error(test_y, test_preds)
    model_file = os.path.join('/tmp', 'model.joblib')
    joblib.dump(pipe, model_file)
    session.file.put(model_file, "@ml_models",overwrite=True)
    return rmse


# Switch to Snowpark Optimized Warehouse for training and to run the stored proc
session.use_warehouse('snowpark_opt_wh')
train_model_sp = F.sproc(train_model, session=session, replace=True, is_permanent=True, name="xgboost_sproc", stage_location="@ml_models")
train_model_sp(session=session)

# Switch back to feature engineering/inference warehouse
session.use_warehouse('snowpark_wh')

import sys
import pandas as pd
import cachetools
import joblib
from snowflake.snowpark import types as T

session.add_import("@ml_models/model.joblib")  

features = [ 'C_BIRTH_YEAR', 'CA_ZIP', 'CD_GENDER', 'CD_MARITAL_STATUS', 'CD_CREDIT_RATING', 'CD_EDUCATION_STATUS', 'CD_DEP_COUNT']

@cachetools.cached(cache={})
def read_file(filename):
       import_dir = sys._xoptions.get("snowflake_import_directory")
       if import_dir:
              with open(os.path.join(import_dir, filename), 'rb') as file:
                     m = joblib.load(file)
                     return m

@F.pandas_udf(session=session, max_batch_size=10000, is_permanent=True, stage_location='@ml_models', replace=True, name="clv_xgboost_udf")
def predict(df:  T.PandasDataFrame[int, str, str, str, str, str, int]) -> T.PandasSeries[float]:
       m = read_file('model.joblib')       
       df.columns = features
       return m.predict(df)

inference_df = session.table('feature_store')
inference_df = inference_df.drop(['CUSTOMER_SK', 'C_CURRENT_HDEMO_SK', 'C_CURRENT_ADDR_SK', 'C_CUSTOMER_ID', 'CA_ADDRESS_SK', 'CD_DEMO_SK'])
inputs = inference_df.drop("TOTAL_SALES")
snowdf_results = inference_df.select(*inputs,
                    predict(*inputs).alias('PREDICTION'), 
                    (F.col('TOTAL_SALES')).alias('ACTUAL_SALES')
                    )
snowdf_results.write.mode('overwrite').save_as_table('predictions')

snowdf_results.show()

session.sql('alter warehouse snowpark_wh suspend')
session.sql('alter warehouse snowpark_opt_wh suspend')

# END