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

# SET WAREHOUSE CONTEXT
session.use_warehouse('nlp_wh')

# SET SCHEMA CONTEXT
session.use_schema('nlp_db.public')

# Import the needed packageinto the Snowpark session
session.add_packages("snowflake-snowpark-python")
session.add_packages("scikit-learn", "pandas", "numpy", "nltk", "joblib", "cachetools")

# TURN SNOWFLAKE INTO LOCAL DATAFRAME
train_dataset_name="TRAIN_DATASET"
train_dataset = session.table(train_dataset_name)

# VIEW DATA
train_dataset.show()

# ARE ANY ROWS MISSING SENTIMENT?
train_dataset.where(train_dataset["SENTIMENT"].isNotNull() == False).show()

train_dataset_flag = train_dataset.withColumn("SENTIMENT_FLAG", fn.when(train_dataset.SENTIMENT == "positive", 1)
                                     .otherwise(2))

# TRAIN MODEL
import io
import joblib

def save_file(session, model, path):
    input_stream = io.BytesIO()
    joblib.dump(model, input_stream)
    session._conn._cursor.upload_stream(input_stream, path)
    
    return "successfully created file: " + path

# TRAIN FUNCTION
def train_model_review_pipline(session : Session, train_dataset_name: str) -> Variant:
    
    from nltk.corpus import stopwords
    import sklearn.feature_extraction.text as txt
    from sklearn import svm
    
    import os
    from joblib import dump
        
    train_dataset = session.table(train_dataset_name)
    train_dataset_flag = train_dataset.withColumn("SENTIMENT_FLAG", fn.when(train_dataset.SENTIMENT == "positive", 1)
                                     .otherwise(2))
    nb_record = train_dataset_flag.count()
    train_x = train_dataset_flag.toPandas().REVIEW.values
    train_y = train_dataset_flag.toPandas().SENTIMENT_FLAG.values
    print('Taille train x : ', len(train_x))
    print('Taille train y : ', len(train_y))
    
    print('Configuring parameters ...')
    # bags of words: parametrage
    analyzer = u'word' # {‘word’, ‘char’, ‘char_wb’}
    ngram_range = (1,2) # unigrammes
    languages = ['english']
    lowercase = True
    token = u"[\\w']+\\w\\b" #
    max_df=0.02    #50. * 1./len(train_x)  #default
    min_df=1 * 1./len(train_x) # on enleve les mots qui apparaissent moins de 1 fois
    max_features = 100000
    binary=True # presence coding
    strip_accents = u'ascii' #  {‘ascii’, ‘unicode’, None}
    
    svm_max_iter = 100
    svm_c = 1.8
    
    print('Building Sparse Matrix ...')
    vec = txt.CountVectorizer(
        #encoding='latin1', \
        #strip_accents=strip_accents, \
        #lowercase=lowercase, \
        #preprocessor=process, \
        #tokenizer=token,\
        #stop_words=stop_words, \
        token_pattern=token, \
        ngram_range=ngram_range, \
        analyzer=analyzer,\
        max_df=max_df, \
        min_df=min_df, \
        #max_features=max_features, \
        vocabulary=None, 
        binary=binary)

    # pres => normalisation
    bow = vec.fit_transform(train_x)
    #transformer = txt.TfidfTransformer()
    #bow = transformer.fit_transform(bow)
    print('Taille vocabulaire : ', len(vec.get_feature_names_out()))
    
    print('Fitting model ...')
    model = svm.LinearSVC(C=svm_c, max_iter=svm_max_iter)
    print(model.fit(bow, train_y))
    
    # Upload the Vectorizer (BOW) to a stage
    print('Upload the Vectorizer (BOW) to a stage')
    session.sql('create stage if not exists ml_models').collect()
    model_output_dire = '/tmp'
    model_file = os.path.join(model_output_dire, 'vect_review.joblib')
    dump(vec, model_file, compress=True)
    session.file.put(model_file, "@ml_models", auto_compress=False, overwrite=True)
    
    # Upload trained model to a stage
    print('Upload trained model to a stage')
    session.sql('create stage if not exists ml_models').collect()
    model_output_dire = '/tmp'
    model_file = os.path.join(model_output_dire, 'model_review.joblib')
    dump(model, model_file, compress=True)
    session.file.put(model_file, "@ml_models", auto_compress=True, overwrite=True)
    
    return {"STATUS": "SUCCESS", "R2 Score Train": str(model.score(bow, train_y))}

# Push the train into Snowflake using a Store Procedure
session.sproc.register(func=train_model_review_pipline, name="train_model_proc", replace=True)

# Call the Store Procedure to train the model
session.call("train_model_proc", "TRAIN_DATASET")

# Import the needed files from the stage
session.add_import("@ml_models/model_review.joblib")
session.add_import("@ml_models/vect_review.joblib")

# Function to load the model from the Internal Stage (Snowflake)
import cachetools

@cachetools.cached(cache={})
def load_file(filename):
    
    import joblib
    import sys
    import os
    
    import_dir = sys._xoptions.get("snowflake_import_directory")
    
    if import_dir:
        with open(os.path.join(import_dir, filename), 'rb') as file:
            m = joblib.load(file)
            return m
        
# Deploy UDF
@udf(name='predict_review',is_permanent = False, stage_location = '@ml_models', replace=True)
def predict_review(args: list) -> float:
    
    import sys
    import pandas as pd
    from joblib import load

    model = load_file("model_review.joblib")
    vec = load_file("vect_review.joblib")
        
    features = list(["REVIEW", "SENTIMENT_FLAG"])
    
    row = pd.DataFrame([args], columns=features)
    bowTest = vec.transform(row.REVIEW.values)
    
    return model.predict(bowTest)

# Show Test Dataset
test_dataset = session.table("TEST_DATASET")
new_df = test_dataset.withColumn("SENTIMENT_FLAG", fn.when(test_dataset.SENTIMENT == "positive", 1)
                                     .otherwise(2))
new_df.show()

# Show Prediction on Test Dataset
new_df.select(new_df.REVIEW, new_df.SENTIMENT, new_df.SENTIMENT_FLAG,\
              fn.call_udf("predict_review", fn.array_construct(col("REVIEW"), col("SENTIMENT_FLAG"))).alias('PREDICTED_REVIEW')) \
        .show()

# Push prediction dataset into Snowflake
new_df.select(new_df.REVIEW, new_df.SENTIMENT, new_df.SENTIMENT_FLAG,\
              fn.call_udf("predict_review", fn.array_construct(col("REVIEW"), col("SENTIMENT_FLAG"))).alias('PREDICTED_REVIEW')) \
        .write.mode('overwrite').saveAsTable('review_prediction')

# SET SCHEMA CONTEXT
session.use_schema('nlp_db.public')

# Compare the target label with the predicted one
session.sql("SELECT * FROM REVIEW_PREDICTION WHERE SENTIMENT_FLAG <> PREDICTED_REVIEW").show()

# Close Session
session.close()