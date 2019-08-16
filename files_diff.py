# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:56:00 2019

@author: dxuser22
"""

import difflib

# get file content
def get_fc(file_name):
    f1 = open(file_name, 'r') 
#    fc1 = f1.read()
    fc1 = f1.readlines()
    f1.close()
    return fc1


file1 = 'events_display_2_pkl.py'
file2 = 'iri_events_display_2_pkl.py'

fc1 = get_fc(file1)
fc2 = get_fc(file2)

differ = difflib.HtmlDiff( tabsize=4, wrapcolumn=80 )
html = differ.make_file( fc1, fc2, context=False )

outfile = open( 'files_diff_output.html', 'w' )
outfile.write(html)
outfile.close() 
