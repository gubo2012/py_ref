'''
Example plots
Andy Wheeler
'''

import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import sys
import os

dat_dir = r'C:\Users\e009156\Documents\GitHub\data-science-utils\education\Intro_DataScience\DataViz_101'
os.chdir(dat_dir)
sys.path.append(dat_dir)
import hms_plotstyle

time_dat = pd.read_excel('ExampleTables.xlsx', sheet_name='Original')

######################################
#Example line graph

#Making a simple line graph
fig, ax = plt.subplots()
ax.plot(time_dat['cyclenum'], time_dat['TotalClaims'])
plt.show()

#Updating the HMS plot style
matplotlib.rcParams.update(hms_plotstyle.hms_style)

#Redoing the same chart, updating nicer names and styles
fig, ax = plt.subplots(figsize=(8,4)) #making the dimensions bigger and wider
#Setting the style for the line and marker
ax.plot(time_dat['cyclenum'], time_dat['TotalClaims'], 
        marker='o', markeredgecolor='w', 
        linewidth=2, markersize=9)
#X, Y axis and title
ax.set_ylabel('Total Claims')
ax.set_xlabel('Cycle')
plt.title('Claims Submitted')
#Setting y limit to include 0
ax.set_ylim(0, 9500)
#Setting the X and Y ticks
plt.xticks(range(time_dat['cyclenum'].max()+1))
plt.yticks(np.linspace(1000,9000,9))
#Saving the file to a high res PNG file
plt.savefig('LineChart.png', dpi=500, bbox_inches='tight')
plt.show()

######################################

######################################
#Example Bar Graph

bar_dat = pd.read_excel('ExampleTables.xlsx', sheet_name='ClientName_Orig')

#Ordering the data
bar_dat.sort_values(by='# Claims', inplace=True)

#Plot the finding rate per client
fig, ax = plt.subplots(figsize=(6,8))
ax.barh(bar_dat['Client Name'], bar_dat['# Claims'],
        edgecolor='black', alpha=0.8)
ax.set_xlabel('Total Claims')
plt.savefig('BarChart.png', dpi=500, bbox_inches='tight')
plt.show()

######################################