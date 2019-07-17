# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 21:43:56 2018

@author: dxuser22
"""

import os
from os import listdir
from os.path import isfile, join, getctime, getmtime, basename
from datetime import datetime as dt
from shutil import copyfile

mypath = os.getcwd()
print(mypath)

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

now = dt.now()

for file in onlyfiles:
    ctime = getctime(file)
    mtime = getmtime(file)
    print(file, dt.utcfromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S'), dt.utcfromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S'))
    if (now - dt.utcfromtimestamp(mtime)).days < 1:
        print('---new file---', (now - dt.utcfromtimestamp(mtime)).seconds / 3600, ' hours')

now_str = now.strftime('%Y%m%d')
print(now_str)


source_path = 'C:\\Users\\dxuser22\\Downloads'
files = [join(source_path, f) for f in listdir(source_path) if isfile(join(source_path, f))]
latest_file = max(files, key = getctime)

dst = join(mypath, basename(latest_file))
copyfile(latest_file, dst)

linux_path = '/var/lib/blob_files/myfile.blob'
windows_path = 'C:' + '\\'.join(linux_path.split('/'))