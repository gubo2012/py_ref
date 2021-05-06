"""

File of functions to help use HMS SQL data sources

Notes:
    - First time an engine is created the user will be asked for credentials (unless override is invoked)
    - Credentials are stored transiently in the Python environment

Todo:
    - map in and support all the DB2 server locations (below)

Need to integrate therese DB2 locations:
ECPROD: jdbc:db2://xpdb2ec100.hms.hmsy.com:50001/ECPROD
CTSPROD: jdbc:db2://ctsprod.hms.hmsy.com:50001/CTSPROD
ARPROD: jdbc:db2://lpdb2ar001.hms.hmsy.com:50001/ARPROD
ARRPROD: jdbc:db2://lpdb2arr001.hms.hmsy.com:50001/ARRPROD
EDIPROD: jdbc:db2://ediprod.hms.hmsy.com:50001/EDIPROD

"""
#################################### libraries and modules ####################################
# region

import sqlalchemy
import pymssql
import teradatasqlalchemy
import ibm_db_sa.ibm_db

import PySimpleGUI as sg
import typing
import traceback
import os

import pandas as pd
import numpy as np
import re

# endregion
#################################### params ####################################
# region

_engine_mapping = {
'edwprod' : 'teradatasql',
'ecprod' : 'ibm_db_sa',
'arprod' : 'ibm_db_sa',
'arrprod' : 'ibm_db_sa',
'pcfprod' : 'ibm_db_sa',
'ediprod' : 'ibm_db_sa',
'ctsprod' : 'ibm_db_sa',
'destdbp' : 'ibm_db_sa',
'wddb001' : 'mssql+pymssql',
'hmsnjtpl' : 'mssql+pymssql',
'wpdbpi001' : 'mssql+pymssql',
'wpdbpi003' : 'mssql+pymssql',
'wpdbpiqda001' : 'mssql+pymssql',
'wpdbpiqda002' : 'mssql+pymssql',
'wpdbpiqda003' : 'mssql+pymssql',
'wpdbpiqda004' : 'mssql+pymssql',
'wpdbpiqda005' : 'mssql+pymssql',
'wpdbpiqda006' : 'mssql+pymssql',
'wpdbpiqda007' : 'mssql+pymssql',
'wpdbpiqda008' : 'mssql+pymssql',
'wpdbpiqda009' : 'mssql+pymssql',
'wpdbpiqda010' : 'mssql+pymssql'
}

# endregion
#################################### public helper functions ####################################
# region

def chunker(seq: iter, size: int) -> typing.Generator:
    """
    Useful for chunking a pandas DataFrame row-wise:

    e.g.
    for sub_df in chunker(df, 100):
        # 100 row chunks accessible in loop

    :param seq: A Python iterable (that can be sliced) to break into chunks
    :param size: An integer representing the chunk size
    :return: A generator to be called to lazily produce chunks
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_credentials() -> None:
    """
    Function to check if credentials are in environment
    :return:
    """
    global _sql_python_username, _sql_python_pwd
    if '_sql_python_username' not in dict(globals()).keys():
        sg.theme('DarkAmber')
        _sql_python_username = sg.PopupGetText('Username', 'e-number')
        _sql_python_pwd = sg.PopupGetText('Password', '', password_char='*')

def reset_credentials() -> None:
    """
    Function to reset credentials
    :return:
    """
    global _sql_python_username, _sql_python_pwd
    del _sql_python_username, _sql_python_pwd
    get_credentials()

def credentials_override(username: str = '', pwd: str = '') -> None:
    """
    Function to override asking for credentials
    :param username:
    :param pwd:
    :return:
    """
    global _sql_python_username, _sql_python_pwd
    _sql_python_username = username
    _sql_python_pwd = pwd

def get_engine(server_name: str, database_name: str = None) -> sqlalchemy.engine.Engine:
    """
    Function returns a sqlalchemy engine based on registry above

    e.g.
    njtpl_eng = get_engine('HMSNJTPL')

    :param server_name: String for the server name
    :param database_name: String for the database name
    :return:
    """
    try:
        conn_type = _engine_mapping[server_name.lower()]
    except:
        except_str = f"""The server "{server_name}" is not a registered server in the Codebase. 
        Function does not know which driver to use. Ask Zach how to register it."""
        raise ValueError(except_str)

    # check and get credentials
    get_credentials()

    if database_name is None:
        database_name = ''

    if conn_type == 'mssql+pymssql':
        return sqlalchemy.create_engine(f"{conn_type}://{server_name.lower()}.hms.hmsy.com/{database_name}")
    elif conn_type == 'teradatasql':
        return sqlalchemy.create_engine(f"{conn_type}://{_sql_python_username}:{_sql_python_pwd}@{server_name.lower()}.hms.com/{database_name}?LOGMECH=LDAP")
    elif conn_type == 'ibm_db_sa':
        return sqlalchemy.create_engine(f'{conn_type}://{_sql_python_username}:{_sql_python_pwd}@eprodadm01.hms.hmsy.com:50001/{server_name.lower()}')

def create_sql_wherein_list(names: typing.List[str]) -> str:
    """
    Create sql "where in" list as strings for SQL
    :param names: list of values
    :return: string needed for SQL
    """
    # start with empty string
    where_str = '('
    where_str += ', '.join([f"'{n}'" for n in names])
    where_str += ')'
    return where_str


# endregion

# EOF