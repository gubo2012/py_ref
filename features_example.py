# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:06:11 2019

@author: 80124664
"""

import pandas as pd
import numpy as np


import featuretools as ft

data = ft.demo.load_mock_customer()

customers_df = data["customers"]
sessions_df = data["sessions"]
transactions_df = data["transactions"]

entities = {"customers" : (customers_df, "customer_id"),
            "sessions" : (sessions_df, "session_id", "session_start"),
            "transactions" : (transactions_df, "transaction_id", "transaction_time")
            }

relationships = [("sessions", "session_id", "transactions", "session_id"),
                 ("customers", "customer_id", "sessions", "customer_id")]

feature_matrix_customers, features_defs = ft.dfs(entities=entities,
                                                 relationships=relationships,
                                                 target_entity="customers")