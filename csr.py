# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:38:31 2019

@author: dxuser22
"""

import numpy as np
np.random.seed(seed=12)  ## for reproducibility
dataset = np.random.binomial(1, 0.1, 20000000).reshape(2000,10000)  ## dummy data
y = np.random.binomial(1, 0.5, 2000)  ## dummy target variable

from scipy.sparse import csr_matrix
sparse_dataset = csr_matrix(dataset)


#import matplotlib.pyplot as plt
#plt.spy(dataset)
#plt.title("Sparse Matrix");


import seaborn as sns

dense_size = np.array(dataset).nbytes/1e6
sparse_size = (sparse_dataset.data.nbytes + sparse_dataset.indptr.nbytes + sparse_dataset.indices.nbytes)/1e6

sns.barplot(['DENSE', 'SPARSE'], [dense_size, sparse_size])
plt.ylabel('MB')
plt.title('Compression')