'''
Example embedding high dimensional
diagnoses codes in a deep learning
framework

Want to try this papers ideas
http://www.farbmacher.de/working_papers/Farbmacher_etal_2020.pdf

Andy Wheeler
'''

'''
ToDo notes

 - activation functions from embedding is not the same as Farbmachers
 - runs slow compared to fully dense example in other code, try using
   torch.scatter_add and torch.expand to embed ICD codes in long space
   as oppossed to wide format (prevent many 0's)
 - get pytorch working on sandbox in local conda environment
'''

'''
High level ideas

 - interaction layers between low dim inputs (e.g. age, netpaid, los)
   and high dim ICD codes (but do the interaction on later hidden layers)
 - variational auto-encoding for the hidden layers for ICD codes
'''


import os
import pandas as pd 
import numpy as np
import scipy.sparse
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import LabelEncoder
import torch
torch.manual_seed(10)

from datetime import datetime

def print_now():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(" ")
    print(f"Current Time = {current_time}")
    print(" ")

#For now have put these datasets
#on shared drive at 
#\\hmsdalfile\general\DSC\projects\PI Clinical Claims
#need to update to grab from terradata
mydir = r'C:\Users\e009156\Desktop\PI_Notes'
os.chdir(mydir)

device = torch.device("cpu")
#device = torch.device("cuda:0") #No GPU for me :(


#########################################################
#DATA PREP

train_dat = pd.read_csv('TrainMS.csv')
test_dat = pd.read_csv('TestMS.csv')

#To make it easier only selecting out a subset of variables
# los - length of stay
# netpaid - $$ of claim
# age, age of beneficiary at admit date
# primary diag = primary ICD10 code
keep_vars = ['Finding',
             'los',
             'netpaidamt_num',
             'Age',
             'primarydiagcode']
#should add in some non-linear stuff for los/netpaid
#restricted cubic splines & interactions

# secondary ICD10 codes, order does not matter here
diags = ['diagcode' + str(i) for i in range(2,26)]
keep_vars += diags
pred_var = 'Finding'

train_dat = train_dat[keep_vars]
test_dat = test_dat[keep_vars]

full_diags = ['primarydiagcode'] + diags

#I want to do label encoding for the cumulative data

def prep_labels(data,fields,fill="_",limit=1000):
    long_dat = pd.Series( data[fields].values.ravel() )
    dat_count = long_dat.value_counts()
    dat_count = dat_count.reset_index()
    dat_count.reset_index(inplace=True)
    dat_count.columns = ['LabelIndex','field','count']
    dat_count['LabelIndex'] = dat_count['LabelIndex'] + 1
    dat_count['LabelIndex'] = dat_count['LabelIndex'].clip(0,limit)
    mis_dat = pd.DataFrame(zip([0],[fill],[-1]),
                        columns=['LabelIndex','field','count'])
    dat_count = pd.concat([mis_dat,dat_count], axis=0)
    dat_count.set_index('field', inplace=True)
    return dat_count

def label_data(data,fields,prep,fill="_"):
    dat_cop = data[fields].fillna(fill).copy()
    #If not in original prepped data, replace with limit
    #value at the end
    rep_val = prep.index[-1]
    fin_values = list(prep.index)
    test_new = dat_cop.isin(fin_values)
    dat_cop.where(test_new, rep_val, inplace=True)
    for d in list(dat_cop):
        dat_cop[d] = list( prep.loc[dat_cop[d], 'LabelIndex'])
    dat_cop = dat_cop.astype(int)
    return dat_cop

limit_diags = 2000

prep_diags = prep_labels(train_dat,full_diags,limit=limit_diags)
labeled_train = label_data(train_dat,full_diags,prep_diags)
labeled_test = label_data(test_dat,full_diags,prep_diags)

#I recieved an AUC of 0.8 for this data when using data robot
#Building the model on train and scoring out of sample test data
#########################################################


#########################################################
#MODEL EXAMPLE 1, ALL DIAGS LABEL ENCODED

#creating my torch tensors
y_train = torch.Tensor( train_dat[[pred_var]].to_numpy() )
y_test = torch.Tensor( test_dat[[pred_var]].to_numpy() )

#should maybe prep for sparse torch tensors
diag_train_torch = torch.tensor( labeled_train.to_numpy(), dtype=torch.long )
diag_test_torch = torch.tensor( labeled_test.to_numpy(), dtype=torch.long )

#Can do this first example all in torch sequential
hidden = 100 #number of hidden layers
#might also do sum and some other type of activation 
#function

#I need to make a custom layer to do what I want I believe
#To use sequential like this
class clamp_norm(torch.nn.Module):
    def forward(self, x):
        return x.clamp(0).norm(dim=1)

#Add in another relu before the sigmoid

model = torch.nn.Sequential(
    torch.nn.Embedding(limit_diags+1,hidden,padding_idx=0,max_norm=1),
    clamp_norm(),
    torch.nn.Linear(hidden, 1, bias=True),
    torch.nn.Sigmoid(),
)

#ReLu constrains the effects to be positive
#So only find diags that increase risk, not decrease

#Logistic loss function and Adam optimizer
loss_fn = torch.nn.BCELoss(reduction='mean')
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#This takes much longer than my other models
#Takes a few seconds for one iteration
#Using the dense layers, should test on GPU to see
#if it speeds up any or not
print("")
print(f'Model 1, only diags with hidden {hidden} and limit {limit_diags}')
print_now()
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(diag_train_torch)
    # Compute and print loss.
    #if t == 300:
    #    learning_rate = 1e-2
    #    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = loss_fn(y_pred, y_train)
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print_now()
#Loss should be around 0.41 (50 hidden layers, 300 labels)

#Checking AUC in-sample vs out of sample
pred_train = y_pred.detach().numpy()
pred_test = model(diag_test_torch).detach().numpy()

#How well is data calibrated?
#pd.Series(pred_train.flatten()).hist(bins=100)
print( pred_train.mean() )
print( train_dat['Finding'].mean() ) #Prediction average should be close

#Better than random, but not so good
#AUC insample of 0.7, out of sample 0.68
#Doing larger labels (2000) and hidden (100)
#Gives larger in sample 0.75, but out of sample still 0.68
#Data Robot for this model is 0.77 for holdout (0.83 in sample)
print( roc_auc_score(train_dat[pred_var], pred_train) )
print( roc_auc_score(test_dat[pred_var], pred_test) )
#########################################################

#########################################################
#MODEL EXAMPLE 2, INCORPORATE OTHER CONTINUOUS VARS

##############
#scaling the other variables
scaler = MinMaxScaler()
oth_vars = ['los',
            'netpaidamt_num',
            'Age']

scaler.fit(train_dat[oth_vars])
train_other = scaler.transform(train_dat[oth_vars])
test_other = scaler.transform(test_dat[oth_vars])
train_other_torch = torch.Tensor( train_other )
test_other_torch = torch.Tensor( test_other )
##############

#embed should be the number of labels encoded
#hidden should be the number of hidden layers

s = torch.nn.Sigmoid()

class EmbedLogit(torch.nn.Module):
    def __init__(self, embed_n, hidden_n, fixed_n):
        super(EmbedLogit, self).__init__()
        self.embed_layer = torch.nn.Embedding(embed_n,hidden_n,padding_idx=0,max_norm=1)
        self.final = torch.nn.Linear(fixed_n + hidden_n, 1, bias=True)
        
    def forward(self, label, fixed):
        embed_weights = self.embed_layer(label).clamp(0).norm(dim=1)
        con_weights = torch.cat([fixed,embed_weights],1)
        final_pred_lin = self.final(con_weights)
        final_pred = s(final_pred_lin)
        return final_pred

model2 = EmbedLogit(embed_n=limit_diags+1,
                    hidden_n=hidden,
                    fixed_n=train_other_torch.shape[1])

#May want to do logloss and not worry about sigmoid transform
loss_fn = torch.nn.BCELoss(reduction='mean')
learning_rate = 1e-2
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)

#Takes a few seconds per each iteration
print("")
print(f'Model 2, includes fixed effects for age, length of stay, and netpaid')
print_now()
for t in range(1000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model2(label=diag_train_torch,fixed=train_other_torch)
    # Compute and print loss
    loss = loss_fn(y_pred, y_train)
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print_now()
#Checking AUC in-sample vs out of sample
pred_train = y_pred.detach().numpy()
print( pred_train.mean() )
print( train_dat['Finding'].mean() ) #Prediction average should be close
#pd.Series(pred_train.flatten()).hist(bins=100)

pred_test = model2(label=diag_test_torch,fixed=test_other_torch).detach().numpy()

#In sample AUC ????
#Out of sample ????
print( roc_auc_score(train_dat[pred_var], pred_train) )
print( roc_auc_score(test_dat[pred_var], pred_test) )
#########################################################



#########################################################

#########################################################
#MODEL EXAMPLE 3, CONTINUOUS/EMBED INTERACTIONS

#Making interaction layers directly in torch tensors
def ten_int(a,b):
    rows = a.shape[0]
    cols = a.shape[1]*b.shape[1]
    return torch.einsum('ij,ik->ijk', a, b).reshape((rows,cols))

#Could also do interactions betwee
#fixed and itself

class EmbedLogitInt(torch.nn.Module):
    def __init__(self, embed_n, hidden_n, fixed_n):
        super(EmbedLogitInt, self).__init__()
        self.embed_layer = torch.nn.Embedding(embed_n,hidden_n,padding_idx=0,max_norm=1)
        self.final = torch.nn.Linear(fixed_n + hidden_n + fixed_n*hidden_n, 1, bias=True)
        
    def forward(self, label, fixed):
        embed_weights = self.embed_layer(label).clamp(0).norm(dim=1)
        int_weights = ten_int(embed_weights, fixed)
        con_weights = torch.cat([fixed,embed_weights,int_weights],1)
        final_pred_lin = self.final(con_weights)
        final_pred = s(final_pred_lin)
        return final_pred

model3 = EmbedLogitInt(embed_n=limit_diags+1,
                    hidden_n=hidden,
                    fixed_n=train_other_torch.shape[1])

#May want to do logloss and not worry about sigmoid transform
loss_fn = torch.nn.BCELoss(reduction='mean')
learning_rate = 1e-2
optimizer = torch.optim.Adam(model3.parameters(), lr=learning_rate)

#Takes a few seconds per each iteration
print("")
print(f'Model 3, includes interactions between emedded layer and fixed effects')
print_now()
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model3(label=diag_train_torch,fixed=train_other_torch)
    # Compute and print loss
    loss = loss_fn(y_pred, y_train)
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print_now()
#Checking AUC in-sample vs out of sample
pred_train = y_pred.detach().numpy()
print( pred_train.mean() )
print( train_dat['Finding'].mean() ) #Prediction average should be close
#pd.Series(pred_train.flatten()).hist(bins=100)

pred_test = model3(label=diag_test_torch,fixed=test_other_torch).detach().numpy()

#In sample AUC ????
#Out of sample ????
print( roc_auc_score(train_dat[pred_var], pred_train) )
print( roc_auc_score(test_dat[pred_var], pred_test) )
#########################################################