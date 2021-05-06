"""

File to pull results sets for selection or back-testing

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

import pandas as pd
from sqlalchemy import create_engine
from sql_python.sql_helpers import *
from python_utils.decorators import *

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

des_result_set_table = 'zz_AMH_10_2019_result_sets'
des_result_set_recoup_table = 'zz_AMH_10_2019_recoup'

# Custom to A20250I for 10/2019
# tables for production process are on QDA001
query_id_query = f"""
select
resultSetName as QueryID,
taskCompletionDate as RunDate,
numBad as NumClaims

from CommercialMaster.dbo.production_runs

where 1=1
and year(taskCompletionDate) = 2019
and month(taskCompletionDate) = 10
and resultsetname like 'A20250I%'

order by taskCompletionDate desc
"""

# endregion
#################################### data pull down ####################################
# region

# get all the engines
credentials_override()
pi_eng = get_engine('WPDBPI001')
qda1_eng = get_engine('WPDBPIQDA001')
qda4_eng = get_engine('WPDBPIQDA004')
njtpl_eng = get_engine('HMSNJTPL', 'TPLManager')

# multi-thread worker to retrieve data from result sets
def result_set_worker(job: pd.Series,
                      result_set_des_table: str,
                      recoup_des_table: str) -> None:
    """
    worker to get result set and recoup data for a result set
    :param job:
    :param result_set_des_table:
    :param recoup_des_table:
    :return:
    """

    query_id = job['QueryID']
    try:
        # get query
        result_set_query = f"""
        select
        {select_cols_str}
        from comm_ReviewFileProcess.dbo.{query_id}_headerRecords h (nolock)
        inner join comm_ReviewFileProcess.dbo.{query_id}_memberRecords m (nolock)
        on h.hdiassignedmemberpk = m.hdiassignedmemberpk
        inner join comm_ReviewFileProcess.dbo.{query_id}_providerRecords p (nolock)
        on h.hdiassignedproviderpk = p.hdiassignedproviderpk
        where h.hdiassignedheaderpk in
        (select hdiassignedheaderpk_bad from comm_ReviewFileProcess.dbo.{query_id}_review r (nolock))
        """
        # print(result_set_query)
        df = pd.read_sql(result_set_query, qda4_eng)
        df['QueryID'] = query_id

    except:
        print(f"Result set query for {query_id} failed.")
        return None

    print(f'{query_id} result set has {df.shape[0]:,} rows.')

    # write out to table
    df.to_sql(result_set_des_table, njtpl_eng, if_exists='append', index=False)

    print(f'{query_id} result set data processed.')

    try:
        # get query
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
            left join  RecoupComplex.dbo.ReviewMaster rm (nolock)
                on rm.ClaimID = c.ID
            left join RecoupComplex.dbo.Review r (nolock)
                on rm.ID = R.ReviewMasterID
                       and r.RecordStatus = 1
                       and rm.RecordStatus = 1
            left join MST_LookupTables.dbo.EcodetoClientCompany e (nolock)
                on e.ClientCompanyID = e.ClientCompanyID
                       and r.EcodeID = e.EcodeID
                       and e.RecordStatus = 1
            left join MST_LookupTables.dbo.ECode ed (nolock)
                on r.ECodeID = ed.ID
        
            where 1=1
            and QueryID = '{query_id}'
            -- and ReviewType in ('C', 'D')
            and c.BillFromDate >= GETDATE() - 365 * 2 ) a
        
        where 1=1
        and rn = 1
        """
        # print(result_set_query)
        df = pd.read_sql(recoup_query, pi_eng)

    except:
        print(f"Recoup query for {query_id} failed.")
        return None

    print(f'{query_id} recoup data has {df.shape[0]:,} rows.')

    # write out to table
    df.to_sql(recoup_des_table, njtpl_eng, if_exists='append', index=False)

    print(f'{query_id} recoup data processed.')

if __name__ == '__main__':

    # start timer
    a = time.time()
    print(datetime.datetime.now())

    # pull the query ids to work
    qids_to_work = pd.read_sql(query_id_query, qda1_eng)

    # go through query ids to work
    with ThreadPoolExecutor(2) as ex:
        for _, row in qids_to_work.iterrows():
            print(f"Expected rows : {row['NumClaims']:,}")
            ex.submit(result_set_worker, row, des_result_set_table, des_result_set_recoup_table)

    b = time.time()
    print(f'{int((b - a) / 3600)} hrs and {round((b - a) % 3600 / 60, 2)} mins')


# endregion
