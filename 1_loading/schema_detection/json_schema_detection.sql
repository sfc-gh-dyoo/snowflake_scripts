// SET ROLE CONTEXT
use role accountadmin;

// SET WAREHOUSE CONTEXT
create or replace warehouse schema_detection_wh with warehouse_size = xsmall;
use warehouse schema_detection_wh;

// SET DATABASE CONTEXT
create or replace database schema_detection_db;
use schema schema_detection_db.public;

// CREATE FILE FORMAT
create or replace file format json_format type = json;

// CREATE STAGE
create or replace stage json_stage file_format = json_format;

// CREATE TABLE FOR SAMPLE DATA
create or replace table cloud_stocks_json(feed variant);

// INSERT SAMPLE DATA
insert into cloud_stocks_json select parse_json('{"ticker": "AMZN","date": "2023-03-10","close": 90.73}');
insert into cloud_stocks_json select parse_json('{"ticker": "GOOGLE","date": "2023-03-10","close": 90.63}');
insert into cloud_stocks_json select parse_json('{"ticker": "MSFT","date": "2023-03-10","close": 248.59}');
insert into cloud_stocks_json select parse_json('{"ticker": "AMZN","date": "2023-03-09","close": 92.25}');
insert into cloud_stocks_json select parse_json('{"ticker": "GOOGLE","date": "2023-03-09","close": 92.32}');
insert into cloud_stocks_json select parse_json('{"ticker": "MSFT","date": "2023-03-09","close": 252.32}');

// COPY DATA INTO INTERNAL STAGE AS IF DATA WAS IN CLOUD STORAGE
copy into @json_stage from cloud_stocks_json;

// DROP TABLE
drop table cloud_stocks_json;

// VIEW JSON DATA
select $1 from @json_stage;

// VIEW SCHEMA DETECTED
select * 
from table(
    infer_schema(
        location => '@json_stage'
            ,file_format => 'json_format'
    )
);

// CREATE TABLE FROM SCHEMA DETECTION WITHOUT WRITING DDL
create or replace table cloud_stocks 
using template (
    select array_agg(object_construct(*))
    from table( 
        infer_schema(
            location=>'@json_stage',
            file_format=>'json_format')));

// COPY SAMPLE DATA INTO NEW TABLE WITHOUT WRITING DDL
copy into cloud_stocks
from @json_stage
match_by_column_name = case_insensitive;

// VIEW SAMPLE DATA
select * from cloud_stocks;