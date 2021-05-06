'''
This script copies postgres DBs from
test to prod 

Need to fill in items in db_info 
dictionary on line 27, and 
schema on line 25

Andy Wheeler
'''

import sqlalchemy
import pandas as pd
# you need this installed for postgres
# via sqlalchemy, but do not need it 
# imported
# import psycopg2 


##########################################
# PARAMETERS YOU NEED

schema = '?piclinicals?'

#should be a service account to write to prod!
# [uid, pwd, ip, dbname]
db_info = {'test': ['?svc_dsmlt_pic?','?pwd?','ltdbdrbot001','robot_test'],
           'prod': ['?svc_dsmlp_pic?','?pwd?','lpdbdrbot001','robot_prod']}

tl = db_info['test']
test_con = f'postgres://{tl[0]}:{tl[1]}@{tl[2]}.hms.hmsy.com:5432/{tl[3]}'

pl = db_info['prod']
prod_con = f'postgres://{pl[0]}:{pl[1]}@{pl[2]}.hms.hmsy.com:5432/{pl[3]}'


##########################################

##########################################
# Getting the list of tables that are 
# *already created* on prod to fill in
# need to do a change request to get the tables
# created, see CHG0038030 for an example

prod_eng = sqlalchemy.create_engine(prod_con)

table_names = f'''
SELECT *
FROM pg_catalog.pg_tables
WHERE schemaname = '{schema}'
'''

tables_prod = pd.read_sql(sql=table_names, con=prod_eng)
tables_list = list(tables_prod['tablename'])
print('These are the tables created on prod')
for t in tables_list:
    print(t)

##########################################

##########################################
# Grabbing those tables from test
# and importing them into prod

test_eng = sqlalchemy.create_engine(test_con)
save_tabs = []
chunk_n = 2000 #may want to up this

for t in tables_list:
    print(f'\nTrying to transfer table {t}')
    try:
        print(f'Grabbing {t} from test')
        # Get the test table
        test_table = pd.read_sql_table(t,schema=schema,con=test_eng, chunksize=chunk_n)
        test_fin = []
        print(f'Success in grabbing {t} from test')
        try:
            print('Transferring table to prod')
            chunk_iter = 0
            for test_chunk in test_table:
                test_fin.append(test_chunk)
                test_chunk.to_sql(t,con=prod_eng,schema=schema,if_exists = 'append', index=False)
                chunk_iter += 1
                print(f'Success in transferring table {t} chunk {chunk_iter} to prod')
            test_fin_tab = pd.concat(test_fin, axis=0)
            save_tabs.append( (t, test_fin_tab.copy() ) )
        except:
            print(f'\nUnable to transfer table {t} to prod\n')
    except:
        print(f'\nUnable to grab table {t} from test\n')

##########################################

##########################################
# For those tables that were successfully 
# Uploaded to prod, checking to make sure 
# They are all the same df.equals(?) or 
# Just check the shape

prod_tabs = []

for t, dat in save_tabs:
    print(f'\nTesting equality for table {t}')
    prod_table = pd.read_sql_table(t,schema=schema,con=prod_eng)
    check = dat.equals(prod_table)
    #print(check)
    if check:
        print(f'Table {t} in prod/test are all equal')
    else:
        print(f'Table {t} in prod/test are not equal')
        prod_tabs.append( (t, prod_table.copy()) )
        print(f'Shape of prod table {prod_table.shape}')
        print(f'Shape of test table {dat.shape}')

# Saving the objects so you can check interactively
# eg pl = set(list(prod_tabs[0][1]))
#    tl = set(list(save_tabs[0][1]))
#    pl.symmetric_difference(tl)
# to see if the columns are different

##########################################

##########################################
# If you messed up and need to delete
# and start over

#from sqlalchemy.sql import text
#prod_opencon = prod_eng.connect()
#
#for t in tables_list:
#    # Deletes the whole table
#    del_txt = f'DELETE FROM {schema}.{t};'
#    con.execute(del_txt)

##########################################
