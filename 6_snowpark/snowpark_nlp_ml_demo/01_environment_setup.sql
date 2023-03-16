// SET ROLE CONTEXT
use role accountadmin;

// CREATE FEATURE ENGINEERING WAREHOUSE
create or replace warehouse nlp_wh with warehouse_size = small;

// CREATE DATABASE FOR IMDB DATA
create or replace database nlp_db;

// SET CONTEXT
use schema nlp_db.public;

// CREATE TABLES FOR IMDB DATA
create or replace table nlp_db.public.train_dataset (
	review string
	,sentiment string
);

create or replace table nlp_db.public.test_dataset (
	review string
	,sentiment string
);

// CREATE FILE FORMAT 
create or replace file format nlp_db.public.csv_load_format
type = csv
field_optionally_enclosed_by = '\042'
skip_header = 1;

// CREATE STAGE
create or replace stage nlp_db.public.csv_stage 
file_format = ( format_name = 'csv_load_format' );

