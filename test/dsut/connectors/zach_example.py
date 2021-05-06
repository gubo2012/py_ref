"""

This file is an example file that pulls Recoup Findings/NoFindings for AMH and then retrieves
their master format version from the corresponding result sets. It writes the results into
a SQLite Database.

Notes:
    - replace e_num in environment prep below (may have to adjust file path that you have repo cloned to)
    - e_number gets put in global environment by the the environment prep function
"""
#################################### standard environment prep ####################################
# region

# function to properly set up environment for local imports
def get_environment_ready(e_num: str = None) -> None:
    """
    Gets the python system path correctly added and embeds e_number into space

    Dependencies:
        - re
        - sys

    :param e_num: employee number
    :return: None
    """

    from re import search
    from sys import path

    global e_number

    # was e_number supplied?
    if e_num:
        e_number = e_num
    else:
        raise ValueError('Please supply your employee number to get_environment_ready.')

    # check the e_number provided
    if search('e[0-9]{6}', e_number):
        path.append(f'C:/Users/{e_number}/Documents/data-science-utils/')
    else:
        raise ValueError(f'{e_number} does not have format e######')

# need to run before local imports
get_environment_ready(e_num = 'e008569')

# endregion
#################################### libraries and modules ####################################
# region

import datetime
import time

import pandas as pd
from sqlalchemy import create_engine
from sql_python.sql_helpers import *
# from python_utils.decorators import *

from concurrent.futures import ThreadPoolExecutor

# endregion
#################################### parameters ####################################
# region

# cols to write down to sqlite
select_cols = ['hdiassignedheaderpk',
 'h.hdiassignedmemberpk',
 'p.providerid',
 'billingprovidertaxid',
 'claimnum',
 'drgcodepaid',
 'netpaidamt',
 'admitdiagcode',
 'primarydiagcode',
 'diagcode2',
 'diagcode3',
 'diagcode4',
 'diagcode5',
 'diagcode6',
 'diagcode7',
 'diagcode8',
 'diagcode9',
 'diagcode10',
 'diagcode11',
 'diagcode12',
 'diagcode13',
 'diagcode14',
 'diagcode15',
 'diagcode16',
 'diagcode17',
 'diagcode18',
 'diagcode19',
 'diagcode20',
 'diagcode21',
 'diagcode22',
 'diagcode23',
 'diagcode24',
 'diagcode25',
 'proccode1',
 'proccode2',
 'proccode3',
 'proccode4',
 'proccode5',
 'proccode6',
 'poacode1',
 'poacode2',
 'poacode3',
 'poacode4',
 'poacode5',
 'poacode6',
 'poacode7',
 'poacode8',
 'poacode9',
 'poacode10',
 'poacode11',
 'poacode12',
 'poacode13',
 'poacode14',
 'poacode15',
 'poacode16',
 'poacode17',
 'poacode18',
 'poacode19',
 'poacode20',
 'poacode21',
 'poacode22',
 'poacode23',
 'poacode24',
 'poacode25',
 'm.birthdate',
 'm.gender',
 'p.state',
 'patientstatus',
 'admittype',
 'billfromdate',
 'billtodate']
select_cols_str = ', \n'.join(select_cols)

# query to get AMH findings/no-findings for C & D reviews for last two years
recoup_query = f"""
select * from (

    select
    c.hdiassignedheaderpk
    ,c.claimno
    ,a.QueryID
    ,r.ReviewType
    ,r.ReviewStage
    ,r.ReviewLevel
    ,r.CompletedDate
    ,e.ECodeFindingTypeID
    ,r.ECodeID
    ,ed.Description
    ,ed.ReviewTypeID
    ,row_number() over (partition by ClaimNo, ReviewType order by ReviewStage desc, ReviewLevel desc) as rn

    from RecoupComplex.dbo.Claim c (nolock)
    inner join RecoupComplex.dbo.ClaimDetail cd (nolock)
        on cd.ClaimID = c.ID
    inner join RecoupComplex.dbo.AuditDetail ad (nolock)
        on cd.ID = ad.ClaimDetailID
    inner join RecoupComplex.dbo.Audit a (nolock)
        on ad.AuditID = a.ID
    inner join  RecoupComplex.dbo.ReviewMaster rm (nolock)
        on rm.ClaimID = c.ID
    inner join RecoupComplex.dbo.Review r (nolock)
        on rm.ID = R.ReviewMasterID
               and r.RecordStatus = 1
               and rm.RecordStatus = 1
    inner join MST_LookupTables.dbo.EcodetoClientCompany e (nolock)
        on e.ClientCompanyID = e.ClientCompanyID
               and r.EcodeID = e.EcodeID
               and e.RecordStatus = 1
    left join MST_LookupTables.dbo.ECode ed (nolock)
        on r.ECodeID = ed.ID

    where 1=1
    and QueryID like 'A20250I%'
    and ReviewType in ('C', 'D')
    and c.BillFromDate >= GETDATE() - 365 * 2 ) a

where 1=1
and rn = 1
"""

# endregion
#################################### data pull down ####################################
# region

# get all the engines
credentials_override()
pi_eng = get_engine('WPDBPI001')
qda4_eng = get_engine('WPDBPIQDA004')
# sqlite_eng = create_engine(f'sqlite:///C:/Users/{e_number}/Documents/test.db')
njtpl_eng = get_engine('HMSNJTPL', 'TPLManager')

# multi-thread worker to retrieve data from result sets
def result_set_worker(job: pd.DataFrame, name: str = None) -> None:
    """
    Function to take Recoup record and collect results from
    :param job:
    :return:
    """
    try:
        # get query
        result_set_query = f"""
        select
        {select_cols_str}
        from comm_ReviewFileProcess.dbo.{job['queryid'].min()}_headerRecords h (nolock)
        inner join comm_ReviewFileProcess.dbo.{job['queryid'].min()}_memberRecords m (nolock)
        on h.hdiassignedmemberpk = m.hdiassignedmemberpk
        inner join comm_ReviewFileProcess.dbo.{job['queryid'].min()}_providerRecords p (nolock)
        on h.hdiassignedproviderpk = p.hdiassignedproviderpk
        where 1=1
        and hdiassignedheaderpk in {create_sql_wherein_list(job['hdiassignedheaderpk'].tolist())}
        """
        # print(result_set_query)
        df = pd.read_sql(result_set_query, qda4_eng)

    except:
        print(f"Query for {job['queryid'].min()} failed.")
        return None

    df.to_sql('zz_amh_master_format_data', njtpl_eng, if_exists='append', index=False)

    print(f'{name} processed: {df.shape[0]:,} claims.')

if __name__ == '__main__':

    # start timer
    a = time.time()
    print(datetime.datetime.now())

    num = 0
    for sub_df in pd.read_sql(recoup_query, pi_eng, chunksize=1_000):
        if num == 0:
            print('Query completed starting pull.')
        sub_df.columns = sub_df.columns.str.lower()
        sub_df.to_sql('zz_amh_recoup_review_data', njtpl_eng, if_exists='append', index=False)
        num += sub_df.shape[0]
        print(f'{num:,} records written down.')

    # retrieve the data from result sets in batch style
    # sqlite_eng.execute('pragma journal_mode=wal')
    with ThreadPoolExecutor(4) as ex:
        for jobs_df in pd.read_sql('select * from zz_amh_recoup_review_data', njtpl_eng, chunksize=10_000):
            for name, sub_df in jobs_df.groupby('queryid'):
                ex.submit(result_set_worker, sub_df, name)
    # sqlite_eng.execute('pragma journal_mode=delete')

    b = time.time()
    print(f'{int((b - a) / 3600)} hrs and {round((b - a) % 3600 / 60, 2)} mins')


# endregion
