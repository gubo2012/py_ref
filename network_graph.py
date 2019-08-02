# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:43:23 2019

@author: dxuser22
"""

# libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
 
directed_flag = True
#directed_flag = False

if directed_flag:
    
    # ------- DIRECTED
     
    # Build a dataframe with your connections
    # This time a pair can appear 2 times, in one side or in the other!
    df = pd.DataFrame({ 'from':['D', 'A', 'B', 'C','A'], 'to':['A', 'D', 'A', 'E','C']})
    df
     
    # Build your graph. Note that we use the DiGraph function to create the graph!
    G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph() )
     
    # Make the graph
    nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
     
else:
    
    
    # ------- UNDIRECTED
     
    # Build a dataframe with your connections
    # This time a pair can appear 2 times, in one side or in the other!
    df = pd.DataFrame({ 'from':['D', 'A', 'B', 'C','A'], 'to':['A', 'D', 'A', 'E','C']})
    df
     
    # Build your graph. Note that we use the Graph function to create the graph!
    G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph() )
     
    # Make the graphnx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
    nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
    plt.title("UN-Directed")


dict_graph = nx.to_dict_of_lists(G)