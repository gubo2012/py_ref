'''
Attention is all you need
very simple demo of encode/decode
+ attention matrix algebra

Andy Wheeler
'''

import numpy as np
from scipy.special import softmax

encode = np.array([[1.0,1.0],
                   [1.0,0.1],
                   [0.1,0.5]])

decode = np.array([[0.6,0.9,-0.05],
                   [0.6,-0.05,0.9]])

# Label Encoding for each row
a = 0
b = 1
c = 2

##################
#For one at a time
lin = np.matmul(encode[a],decode)
print(lin)

sf = softmax(lin)
print(sf)
##################

##################
#For all 3
lin = np.matmul(encode,decode)
print(lin)

sf = softmax(lin, axis=1)
print(sf)
##################

##################
#Now expand to multiple inputs

sen = [a,a,c,a]

lin = np.matmul(encode[sen],decode)
print(lin)

#We want only one output though?
lin_me = lin.mean(axis=0) #average down columns
print(lin_me)
softmax(lin_me) #softmax to get probability scale
##################

##################
#Attention is all you need

#Give the initial inputs more weight
attention = np.array([[0.6,0.25,0.10,0.05]])

np.matmul(attention,lin)

#or
(lin*attention.T).sum(axis=0)

##################








