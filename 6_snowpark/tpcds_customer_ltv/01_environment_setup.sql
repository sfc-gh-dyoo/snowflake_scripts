// SET ROLE CONTEXT
use role accountadmin;

// CREATE FEATURE ENGINEERING WAREHOUSE
create or replace warehouse snowpark_wh with
    warehouse_size = x3large;

// CREATE SNOWPARK OPTIMIZED WAREHOUSE
create or replace warehouse snowpark_opt_wh with
    warehouse_size = medium
    warehouse_type = 'snowpark-optimized'
    max_concurrency_level = 1;

// CREATE SOURCE DATABASE
create database if not exists sample_data_db from share sfc_samples.sample_data; -- database name line 21 in .py

// CREATE DATABASE FOR PREDICTIONS
create or replace database snowpark_db;

// SET CONTEXT
use schema snowpark_db.public;

// CREATE STAGE FOR MODELS
create or replace stage ml_models;

// SUSPEND WAREHOUSES
alter warehouse snowpark_wh suspend;
alter warehouse snowpark_opt_wh suspend;