# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:58:06 2020

@author: Yifei Yun
"""

#Install psycopg2 package
#pip install psycopg2


import pandas as pd
from sqlalchemy import create_engine

# General db_string format: 'dialect[+driver]://username:password@Host/dbname'
db_string_0 = 'postgres+psycopg2://sbxusr:Sbx1234!@10.30.185.223:5432/sandbox'
db_string_1 = 'postgres://sbxusr:Sbx1234!@10.30.185.223:5432/sandbox'
db_string_2 = 'postgres://dsmlusr:HMSdsml1!@10.30.185.223:5432/sandbox'


engine = create_engine(db_string_1)

# Write / Create table
sql_df=pd.DataFrame(data=[['A01',25],['A02',28],['A03',33]], columns=['ID','Age'])

sql_df.to_sql("test_table",
              con=engine,
              if_exists = 'replace')

# Read table

sql_df_read = pd.read_sql_table("test_table",con=engine)
print(sql_df_read)
