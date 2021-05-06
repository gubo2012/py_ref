# -*- coding: utf-8 -*-
"""
This produces queries to download tables from WPDBSAS001 server 
So to replicate exactly you would need access to that server, 
but hopefully you can figure out how to update to your personal
case

Andy W, andrew.wheeler@hms.com
"""

import pyodbc
import pandas as pd


#On windows it uses your user credentials to authenticate
#So no need to save userid/password to query
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=WPDBSAS001;'
                      'Database=ComplexReviewSource;'
                      'Trusted_Connection=yes;')

#Select all fields and only 10 records to check the query
SQL_Query = pd.read_sql_query(
        '''SELECT TOP (10) * 
        FROM ComplexReviewSource.dbo.PharmacyResult
        ''', conn)

#This is just a pandas dataframe now with the columns nicely defined
print( SQL_Query )

#You can pass in more complicated SQL if you want
AmH_Fields = pd.read_sql_query(
        '''SELECT *
           FROM [ReviewResultDW].[dbo].[CVResult]
           WHERE ClaimID 
           IN
           (SELECT ClaimID 
           FROM ReportDW.dbo.ComplexReviewCompleted 
           WHERE ClientCompanyId = 1126)''', conn)

#And then can export to csv to keep from repeatedly querying the database
AmH_Fields.to_csv('ReportDW_AmHFields_02142020.csv', index=False)

#close the connection
conn.close()