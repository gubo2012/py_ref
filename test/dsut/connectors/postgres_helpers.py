'''
This is a python script to create
the necessary CREATE statments
for the change request for postgres

also function to transfer tables from
test to prod & create pg sqlalchemy 
connect strings

Andy Wheeler
'''

import sqlalchemy
from sqlalchemy.sql import text
import numpy as np
import math
import pandas as pd
# import psycopg2 
# you need this installed for postgres
# via sqlalchemy, but do not need it 
# imported

##########################################
# FUNCTIONS

# This function creates a sqlalchemy connection to postgres db
def create_eng(ip,dbname,uid,pwd):
    '''
    Creates a postgres sqlalchemy connection

    Args:
        - ip: string for ipaddress WITHOUT the .hms.hmsy:5432 at the end
        - dbname: string for dbname
        - uid: string for userid (or service account id)
        - pwd: string for password (that goes with the uid)
    
    Returns:
        If connection a success, returns sqlalchemy engine. If failed
        just returns the connection string
    '''
    test_con = f'postgres://{uid}:{pwd}@{ip}.hms.hmsy.com:5432/{dbname}'
    test_eng = sqlalchemy.create_engine(test_con)
    print('Checking the engine by querying tablespaces')
    tab_sql = 'SELECT * FROM pg_tablespace;'
    print(tab_sql)
    try:
        space_test = pd.read_sql(sql=tab_sql, con=test_eng)
        print(space_test)
        return test_eng
    except:
        print('Unable to query postgres, returning engine string')
        return test_con

# This creates text blob for SQL creation for change request 
def build_script(eng,schema,sel_acct,upd_acct,tab_space):
    '''
    Creates a text string representation of ALL tables/views 
    on a postgres server+schema. For uploading to change request

    Args:
        - eng: sqlalchemy pg engine
        - schema: string for schema you want
        - sel_acct: string for usergroup name associated with select priveleges
        - upd_acct: string for usergroup name associated with the update privs
        - tab_space: string for table space
    
    Returns:
        A text blob that contains the update script for all tables/views
        save as a txt and upload to change request
    '''
    table_names = f'''
    SELECT *
    FROM pg_catalog.pg_tables
    WHERE schemaname = '{schema}'
    '''
    tables_test = pd.read_sql(sql=table_names, con=eng)
    tables_list = list(tables_test['tablename'])
    # Getting the field names and types
    col_query = f'''SELECT *
    FROM information_schema.columns
    WHERE table_schema = '{schema}'
    '''
    col_test = pd.read_sql(sql=col_query, con=eng)
    col_type = col_test['data_type']
    name_val = col_test['column_name']
    var_def = col_type.where(col_type != 'text', 'text COLLATE pg_catalog."default"')
    col_test['row_def'] = '    "' + name_val + '" ' + var_def
    # Building variable string for each table
    var_string = {}
    for t in tables_list:
        var_string[t] = f"\n-- create table {t}\n\nCREATE TABLE {schema}.{t}\n("
        # No adding in each row
        sub_tab = col_test[col_test['table_name'] == t].copy()
        col_li = list(sub_tab['row_def'])
        for c in col_li[0:-1]:
            loc = f'\n{c},'
            var_string[t] += loc
        var_string[t] += f'\n{col_li[-1]}\n)'
        var_string[t] += f'\nWITH (\n    OIDS = FALSE\n)\nTABLESPACE {tab_space};\n'
        var_string[t] += f'\nGRANT SELECT ON TABLE {schema}.{t} TO {sel_acct};'
        var_string[t] += f'\nGRANT SELECT, UPDATE, INSERT, DELETE ON TABLE {schema}.{t} TO {upd_acct};\n'
        #print( var_string[t] )
    # Building the definitions for Views
    vsql = f'''SELECT viewname, definition from pg_catalog.pg_views 
               WHERE schemaname = '{schema}';'''
    vdef = pd.read_sql(vsql, con=pi_test)
    # Only if not empty adding in views
    if vdef.shape[0] > 0:
        views_list = list(vdef['viewname'])
        views_defli = list(vdef['definition'])
        for vw,de in zip(views_list, views_defli):
            var_string[vw] = f"\n-- create view {vw}\n\nCREATE OR REPLACE VIEW {schema}.{vw}\n AS\n"
            var_string[vw] += de
            var_string[vw] += f'\nWITH (\n    OIDS = FALSE\n)\nTABLESPACE {tab_space};\n'
            var_string[vw] += f'\nGRANT SELECT ON TABLE {schema}.{vw} TO {sel_acct};'
            var_string[vw] += f'\nGRANT SELECT, UPDATE, INSERT, DELETE ON TABLE {schema}.{vw} TO {upd_acct};\n'
        all_list = tables_list + views_list
    else:
        all_list = tables_list
    fin_str = f'-- Python creation script for ip: {ip}, db: {dbname}, schema: {schema}\n'
    fin_str += '-- for change request for production\n'
    fin_str += '-- python script created by Andrew Wheeler\n'
    #print(f'These are the tables/views in {schema} on test')
    for t in all_list:
        #print(t)
        fin_str += var_string[t]
    fin_str += '\n-- This script does not include any code to drop/replace tables/views'
    #print('\n PRINTING EXAMPLE GENERATION SCRIPT \n')
    #print( fin_str )
    return fin_str

# This function transfers tables from the from_eng
# To the to_eng, useful for updating tables eg test to prod
def transfer_table(from_eng,to_eng,schema,table_names=None,drop_testfields=None,
                   only_update=True,del_old=False,chunk_size=2000):
    '''
    Transfers exact tables between two postgres dbs, eg transfer from
    test to prod

    Args:
        - from_eng: postgres sqlalchemy engine for the database pulling new rows from
        - to_eng: postgres sqlalchemy engine rows are transferred to this DB
        - schema: string for the schema the tables are located on
        - table_names: either a string for a single table,
                       or list of multiple table strings. Default None
                       pulls ALL tables from the schema
        - drop_testfields: list of fields to *not* check for equality,
                           useful to drop float fields that may cause issues.
                           Default None includes all fields
        - only_update: Boolean, checks rows from the *from_eng* that are new
                       and not 100% the same from the *to_eng*. If False,
                       appends 100% of the rows from the *from_eng* without
                       checking. DEFAULT True.
        - del_old: Boolean, If True, delete 100% of the rows from the *to_eng*.
                   Good to clean out old table before replacing, but be careful!
                   only_update should probably be false if this is True.
                   DEFAULT False.
        - chunk_size: Integer, for the uploading new rows to table, chunks it up
                     into smaller buckets, e.g. only 2000 rows at a time.
                     This is easier to monitor progress on HMS uploads. 
                     DEFAULT 2000.
    
    Returns:
        Does not return anything, should print out status messages
    '''
    res_pd = {}
    str_type = 'aaa'
    list_type = [1,2,3]
    if table_names is None:
            table_sql = f'''SELECT * FROM pg_catalog.pg_tables WHERE schemaname = '{schema}';'''
            tables_test = pd.read_sql(sql=table_sql, con=eng)
            tables_list = list(tables_test['tablename'])
    elif type(table_names) == type(str_type):
        tables_list = [table_names]
    elif type(table_names) == type(list_type):
        tables_list = table_names
    else:
        print('table_names parameter needs to be either a string for a single table')
        print('or a list of strings with multiple tables')
        print('or none grabs all tables from the *from_eng*')
        return(-1)
    for t in tables_list:
        loc_tab = pd.read_sql_table(t, from_eng, schema) #may want to chunk this up
        # clean strings and floats?
        res_pd[t] = loc_tab.copy()
    if only_update:
        curr_pd = {}
        for t in tables_list:
            loc_tab = pd.read_sql_table(t, to_eng, schema)
            if drop_testfields:
                match_fields = list( set(list(loc_tab)) - set(drop_testfields) )
            else:
                match_fields = list(loc_tab)
            merge_tab = pd.merge(res_pd[t],loc_tab[match_fields],
                                 how='left',on=match_fields,indicator='info')
            curr_pd[t] = merge_tab[merge_tab['info'] == 'left_only'].copy()
            curr_pd[t] = curr_pd[t][list(res_pd[t])].copy()
    else:
        curr_pd = res_pd
    for t in tables_list:
        if del_old:
            print(f'Deleting all old records from the *to_eng* for table {t}')
            del_txt = text(f"""DELETE FROM {schema}.{t};""")
            pg_conn = to_eng.connect()
            pg_conn.execute(del_txt)
        if curr_pd[t].shape[0] > 0:
            to_upload = curr_pd[t] #may want to do this in chunks
            tot_split = math.ceil(to_upload.shape[0]/chunk_size)
            split_rows = np.array_split(to_upload.index,tot_split)
            for n,s in enumerate(split_rows):
                loc_upload = to_upload.loc[s,:].copy()
                print(f'Uploading round {n+1} of {tot_split} for table {t}')
                loc_upload.to_sql(t,schema=schema,con=to_eng,if_exists='append',index=False)
        else:
            print(f'Table {t} has 0 new rows to append')

##########################################

##########################################
# Now building the create script from the tables
# Need to insert the service accounts you want

# Stuff to query the table
schema = 'piclinicals'
uid = 'svc_dsmlt_pic'
pwd = '?????????????' #enter your password here
ip = 'ltdbdrbot001'
dbname = 'robot_test'

pi_test = create_eng(ip=ip,dbname=dbname,uid=uid,pwd=pwd)

res = build_script(eng=pi_test,
                   schema=schema,
                   sel_acct='robo_pi_sel',
                   upd_acct='robo_pi_upd',
                   tab_space='robodata')

print(res)
#saving to a text file?
##########################################

##########################################
# Example of transferring a table

# Stuff to query the table
sacct = 'svc_dsmlp_pic'
pwd = '??????????????' #enter your password here
ip = 'lpdbdrbot001'
dbname = 'robot_prod'

pi_prod = create_eng(ip=ip,dbname=dbname,uid=sacct,pwd=pwd)

# Example transferring just new rows for one table
transfer_table(pi_test,pi_prod,schema,
               drop_testfields=['estimatedsavings'],
               table_names='pic_staging')

# Example deleting old table and changing to new
transfer_table(pi_test,pi_prod,schema,
               only_update=False,del_old=True,
               table_names=['pic_current_pilot'])

##########################################
