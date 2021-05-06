'''
Example embedding high dimensional
diagnoses codes in a deep learning
framework

Andy Wheeler
'''

import os
import pandas as pd 
import numpy as np
import scipy.sparse
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import torch

#For now have put these datasets
#on shared drive at 
#\\hmsdalfile\general\DSC\projects\PI Clinical Claims
#need to update to grab from terradata
mydir = r'C:\Users\e009156\Desktop\DeepLearningExample'
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

#For testing, make the training data smaller
#train_dat = train_dat.head(3000)

#I recieved an AUC of 0.8 for this data when using data robot
#Building the model on train and scoring out of sample test data

#I want to do one-hot encoding, but drop rare categories
#This creates a dataset that I can use to score
#The secondary data
def dummy_rep(data, var, limit):
    sub_dat = pd.Series( data[var].values.ravel() )
    res_counts = sub_dat.value_counts().reset_index()
    res_counts.rename(columns={0:'Counts'}, inplace=True)
    res_counts['rep_str'] = res_counts['index']
    res_counts['rep_str'].where(res_counts['Counts'] > limit, 'Other', inplace=True)
    #creating a list representation
    list_groups = res_counts.groupby('rep_str')['index'].apply(list).reset_index()
    list_groups.rename(columns={'index':'code_list'}, inplace=True)
    #Print some summary stats when doing this?
    tot_records = res_counts['Counts'].sum()
    other_records = (res_counts['Counts'] * (res_counts['rep_str'] == 'Other')).sum()
    print(f'Total groups are {list_groups.shape[0]}')
    print(f'Total Other records are {other_records} out of {tot_records}')
    return list_groups

def cum_score(data, var, list_groups, base, cum=False, sparse=True):
    fin_cols = []
    var_names = list(base + list_groups['rep_str'])
    if cum:
        for i, r in list_groups.iterrows():
            temp = data[var].isin(r['code_list']).sum(axis=1)
            fin_cols.append( temp )
    else:
        for i, r in list_groups.iterrows():
            temp = data[var].isin(r['code_list']).any(axis=1)
            fin_cols.append( temp )        
    if sparse:
        sp = scipy.sparse.csr_matrix(fin_cols).T
        fin_pd = pd.DataFrame.sparse.from_spmatrix(sp, columns=var_names)
        mb = fin_pd.memory_usage().sum()/1e6
        print(f'Sparse DF Memory: {mb} (mega bytes)')
    else:
        fin_pd = pd.concat(fin_cols, axis=1)
        fin_pd.columns = var_names
        mb = fin_pd.memory_usage().sum()/1e6
        print(f'Non-Sparse DF Memory: {mb} (mega bytes)')
    return fin_pd
     
#One set for primary diags, another for secondary
prim_res = dummy_rep(train_dat, var='primarydiagcode', limit=100)
sec_res = dummy_rep(train_dat, var=diags, limit=1000)

#Now creating the dummy variable sets, takes a like two minutes
pdum_train = cum_score(train_dat, ['primarydiagcode'], prim_res, 'P_')
sdum_train = cum_score(train_dat, diags, sec_res, 'S_')

pdum_test = cum_score(test_dat, ['primarydiagcode'], prim_res, 'P_')
sdum_test = cum_score(test_dat, diags, sec_res, 'S_')
#########################################################


#########################################################
#MODEL EXAMPLE 1, PRIMARY DIAG LOWER DIM EMBEDDING

#creating my torch tensors
y_train = torch.Tensor( train_dat[[pred_var]].to_numpy() )
y_test = torch.Tensor( test_dat[[pred_var]].to_numpy() )

#should maybe prep for sparse torch tensors
pdum_train_torch = torch.Tensor( pdum_train.to_numpy() )
pdum_test_torch = torch.Tensor( pdum_test.to_numpy() )

#Can do this first example all in torch sequential
prim_shape = pdum_train.shape[1] #input data size
hidden = 50 #number of hidden layers

model = torch.nn.Sequential(
    torch.nn.Linear(prim_shape, hidden, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden, 1, bias=True),
    torch.nn.Sigmoid(),
)
#ReLu constrains the effects to be positive
#So only find diags that increase risk, not decrease

#Logistic loss function and Adam optimizer
loss_fn = torch.nn.BCELoss(reduction='mean')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#6000 runs pretty fast on my machine, only around 20 minutes
for t in range(6000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(pdum_train_torch)
    # Compute and print loss.
    loss = loss_fn(y_pred, y_train)
    if t % 10 == 9:
        print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Checking AUC in-sample vs out of sample
pred_train = y_pred.detach().numpy()
pred_test = model(pdum_test_torch).detach().numpy()

#How well is data calibrated?
pd.Series(pred_train.flatten()).hist(bins=100)
print( train_dat['Finding'].mean() ) #Prediction average should be close

#Better than random, but not so good
#Only AUC of around 0.6 both insample and out
#Data Robot for this model is 0.77 for holdout (0.83 in sample)
print( roc_auc_score(train_dat[pred_var], pred_train) )
print( roc_auc_score(test_dat[pred_var], pred_test) )

######################
#To get back an embedding that is a unique set of
#Primary diag codes
pdiag_embed = model[1](model[0](pdum_train_torch)).detach().numpy()
embed_names = ['DiagE_' + str(i+1).zfill(3) for i in range(pdiag_embed.shape[1])]
pdiag_embed_df = pd.DataFrame(pdiag_embed, columns=embed_names)
pdiag_embed_df['primarydiagcode'] = train_dat['primarydiagcode']

#Reducing the set down to unique codes
other_pdiag = list(prim_res.loc[prim_res['rep_str'] == 'Other','code_list'])[0]
pdiag_embed_df.drop_duplicates('primarydiagcode', inplace=True)
other_code = pdiag_embed_df['primarydiagcode'].isin(other_pdiag)
pdiag_embed_df['ReducedStr'] = pdiag_embed_df['primarydiagcode'].where(~other_code, 'Other')
print( pdiag_embed_df['ReducedStr'].value_counts() )
pdiag_embed_df.reset_index(drop=True, inplace=True)
#One of the columns is zero
print( pdiag_embed_df[embed_names].sum(axis=0) )
#adding in the final predicted probability
#Now save to teradata
#########################################################


#########################################################
#MODEL EXAMPLE 2, PRIMARY DIAG AND SECONDARY DIAG
#AND OTHER VARS

#torch tensors for secondary diags
sdum_train_torch = torch.Tensor( sdum_train.to_numpy() )
sdum_test_torch = torch.Tensor( sdum_test.to_numpy() )

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

##############
#For this model not sure how to use sequential
#So need to build it myself

s = torch.nn.Sigmoid()
relu = torch.nn.ReLU()

class HiddenLogit(torch.nn.Module):
    def __init__(self, groups, fixed):
        super(HiddenLogit, self).__init__()
        gl = []
        tot_h = fixed
        for g,h in groups:
            gen_layer = torch.nn.Linear(g, h, bias=False)
            tot_h += h
            gl.append(gen_layer)
        self.orig_layers = torch.nn.ModuleList(gl)
        #I should maybe set the bias here to a small negative
        #Number (intercept in logit model)
        self.final = torch.nn.Linear(tot_h, 1, bias=True)
        
    def forward(self, group_ten, fixed):
        con_weights = [fixed]
        for t,l in zip(group_ten, self.orig_layers):
            loc_weight = relu(l(t))
            con_weights.append(loc_weight)
        stack_hidden = torch.cat(con_weights, 1)
        final_pred_lin = self.final(stack_hidden)
        final_pred = s(final_pred_lin)
        return final_pred


ten_x = [pdum_train_torch,
         sdum_train_torch]

#fix_x = train_other_torch
hid_size = [50,50]
g_sizes = [(x.shape[1],h) for x,h in zip(ten_x,hid_size)]
fix_size = train_other_torch.shape[1]

model2 = HiddenLogit(groups=g_sizes,
                     fixed=fix_size)
#May want to do logloss and not worry about sigmoid transform
loss_fn = torch.nn.BCELoss(reduction='mean')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)

#Alittle bit slower than prior since more parameters
#This is more like 45 minutes
for t in range(6000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model2(group_ten=ten_x,
                    fixed=train_other_torch)
    # Compute and print loss.
    loss = loss_fn(y_pred, y_train)
    if t % 10 == 9:
        print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Checking AUC in-sample vs out of sample
pred_train = y_pred.detach().numpy()
pd.Series(pred_train.flatten()).hist(bins=100)

ten_x_test = [pdum_test_torch,
              sdum_test_torch]

pred_test = model2(group_ten=ten_x_test,
                   fixed=test_other_torch).detach().numpy()

#In sample AUC 0.77
#Out of sample 0.69
print( roc_auc_score(train_dat[pred_var], pred_train) )
print( roc_auc_score(test_dat[pred_var], pred_test) )
#########################################################

#########################################################
#MODEL EXAMPLE 3, INTERACTION LAYERS BETWEEN 
#PRIM/SECONDARY/OTHERVARS

#Need to make another interaction layer
#Could prob figure out how to do directly in torch layers
def comb_diags(x):
    res_vec = []
    b_split = x.split(",")
    prim = b_split.pop(0)
    for diag in b_split:
        if diag != '+':
            res_vec.append( prim + "_" + diag )
    return res_vec

train_dat['diag_list'] = train_dat[['primarydiagcode'] + diags].fillna('+').apply(lambda x: ",".join(x), axis=1)
comb_prim_sec = pd.DataFrame(train_dat['diag_list'].apply(comb_diags).to_list())
combo_codes = dummy_rep(comb_prim_sec, var=list(comb_prim_sec), limit=500)

test_dat['diag_list'] = test_dat[['primarydiagcode'] + diags].fillna('+').apply(lambda x: ",".join(x), axis=1)
comb_prim_sec_test = pd.DataFrame(test_dat['diag_list'].apply(comb_diags).to_list())

comb_dum_train = cum_score(comb_prim_sec, list(comb_prim_sec), combo_codes, 'C_')
comb_dum_test = cum_score(comb_prim_sec_test, list(comb_prim_sec_test), combo_codes, 'C_')

#Making interaction layers directly in torch tensors
def ten_int(a,b):
    rows = a.shape[0]
    cols = a.shape[1]*b.shape[1]
    return torch.einsum('ij,ik->ijk', a, b).reshape((rows,cols))

int_other_prim_train = ten_int(train_other_torch,pdum_train_torch)
int_prim_sec_train = torch.Tensor( comb_dum_train.to_numpy() )
#This does not work, too large
#int_prim_sec_train = ten_int(pdum_train_torch,sdum_train_torch)
ten_x += [int_other_prim_train, int_prim_sec_train]

int_other_prim_test = ten_int(test_other_torch,pdum_test_torch)
int_prim_sec_test = torch.Tensor( comb_dum_test.to_numpy() )
ten_x_test += [int_other_prim_test, int_prim_sec_test]

hid_size = [50,50,50,50]
g_sizes = [(x.shape[1],h) for x,h in zip(ten_x,hid_size)]

model3 = HiddenLogit(groups=g_sizes,
                     fixed=train_other_torch.shape[1])
loss_fn = torch.nn.BCELoss(reduction='mean')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model3.parameters(), lr=learning_rate)

#this is slightly faster than an iteration per second
#so 6000 iterations takes less than 100 minutes
for t in range(6000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model3(group_ten=ten_x,
                    fixed=train_other_torch)
    # Compute and print loss.
    loss = loss_fn(y_pred, y_train)
    if t % 10 == 9:
        print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Checking AUC in-sample vs out of sample
pred_train = y_pred.detach().numpy()
pd.Series(pred_train.flatten()).hist(bins=100)
pred_test = model3(group_ten=ten_x_test,
                   fixed=test_other_torch).detach().numpy()

#In sample AUC 0.83 (and would still get better with more iterations)
#Out of sample still only 0.68
print( roc_auc_score(train_dat[pred_var], pred_train) )
print( roc_auc_score(test_dat[pred_var], pred_test) )
#########################################################