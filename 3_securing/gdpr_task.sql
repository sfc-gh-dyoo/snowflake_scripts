// SET ROLE CONTEXT
use role accountadmin;

// CREATE WAREHOUSE
create or replace warehouse gdpr_wh with 
    warehouse_size = xsmall
    auto_suspend = 60;

// SET WAREHOUSE CONTEXT
use warehouse gdpr_wh;
    
// CREATE DATABASE
create or replace database gdpr_db;

// SET SCHEMA CONTEXT
use schema gdpr_db.public;

// CREATE FAKE PII DATA
create or replace table pii_sample as
    select uuid_string() id
        ,randstr(5, random()) firstname
        ,randstr(6, random()) lastname  
        ,to_varchar(
            uniform(1,9,random())
            || uniform(1,9,random())
            || uniform(1,9,random())
            ||'-'
            || uniform(1,9,random())
            || uniform(1,9,random())
            ||'-'
            || uniform(1,9,random())
            || uniform(1,9,random())
            || uniform(1,9,random())
            || uniform(1,9,random())) social_security_number
        ,dateadd(month,to_double(
            (case when uniform(0,1,random()) = 0 then '-' else '' end)
            || uniform(1,24,random())),current_date()) last_interaction
    from table( generator( rowcount => 100));

// CREATE TASK
create or replace task gdpr_task
    warehouse = gdpr_wh
    schedule = '1 minute'
    as
    update pii_sample 
        set pii_sample.social_security_number = '###-##-####'
        where pii_sample.last_interaction < dateadd(day,-30,current_date());

// TURN ON TASK
alter task gdpr_task resume;

/**************
WAIT 60 SECONDS
**************/

// CHECK TASK WORKED
select * 
from pii_sample
where pii_sample.last_interaction < dateadd(day,-30,current_date());

// CLEAN UP
drop database gdpr_db;
drop warehouse gdpr_wh;