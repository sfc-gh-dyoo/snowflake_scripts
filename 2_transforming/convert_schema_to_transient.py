import snowflake.connector
import os
import csv

#ACCOUNT='<ACCOUNT_NAME>'
#USER='<USER_NAME>'
#PASSWORD='<PASSWORD>'
#WAREHOUSE = '<WAREHOUSE>'
#DATABASE = '<DATABASE>'
#SCHEMA = '<SCHEMA>'

cnx = snowflake.connector.connect(
	user = USER,
	password=PASSWORD,
	account=ACCOUNT,
	warehouse=WAREHOUSE,
	database=DATABASE,
	schema=SCHEMA
	)

tablePaths=[];
cur = cnx.cursor()
try:
	cur.execute("USE ROLE ACCOUNTADMIN")
	cur.execute("USE WAREHOUSE " + WAREHOUSE)
	cur.execute("USE DATABASE " + DATABASE)
	cur.execute("USE SCHEMA " + SCHEMA)


	cur.execute("select table_name, deleted from SNOWFLAKE.ACCOUNT_USAGE.TABLES WHERE TABLE_CATALOG = '" + DATABASE + "' and TABLE_SCHEMA = '" + SCHEMA + "' and DELETED IS NULL;");
	tableNames = [];
	for (tableName, space) in cur:
		if (tableName != "$TABLENAME"):
			tableNames.append(tableName)
	for (tableName) in tableNames:
		try:
			print ("Converting Table : " + tableName)

			cur.execute("create transient table " + tableName + "_clone clone " + tableName)
			cur.execute("alter table " + tableName + " swap with " + tableName + "_clone")
			cur.execute("alter table " + tableName + " set DATA_RETENTION_TIME_IN_DAYS= 0")
			cur.execute("drop table " + tableName + "_clone")

		except:
			print ("Error : " + tableName)
			continue;

finally: 
	cur.close()



