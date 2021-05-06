import pulp
import pandas as pd
import numpy as np

#example data frame, claim id, predicted prob, and claim $$
#should pred prob be PPV instead?????

########################################
#Now create a nice function

def sel_cases(data,pr,val,fpr,tot_n):
    """
    Uses linear programming to select cases, limiting
    to a particular false positive rate
    
    Parameters
    ----------
    data : panda data frame
    pr   : (string) column in data containing predicted probabilities of the outcome
    val  : (string) column in data associated with the claim value
    fpr  : (float) scalar quantile (between 0 & 1) of the number of false positives to limit 
    tot_n  : (int) total number of cases you want to select
    
    Returns
    -------
    sel_cases : pandas series of cases selected given the parameters
    """
    # Setting up the data
    my_dat = data[[pr,val]].copy()
    cases = range(len(my_dat.index))
    prob_min_fpr = 1 - my_dat[pr] - fpr
    exp_claim = my_dat[pr] * my_dat[val]
    # Setting up the model and contraints
    mod = pulp.LpProblem("Selecting Cases",pulp.LpMaximize)
    dec_vars = pulp.LpVariable.dicts("Cases to select", [i for i in cases], lowBound=0, upBound=1, cat=pulp.LpInteger)
    mod += pulp.lpSum( exp_claim[i]*dec_vars[i] for i in cases ), "Maximize Claims given pred prob"
    mod += pulp.lpSum( prob_min_fpr[i]*dec_vars[i] for i in cases ) <= 0, "Setting false positive rate"
    mod += pulp.lpSum( dec_vars[i] for i in cases ) <= tot_n, "Threshold on total number of cases selected"
    mod.solve()
    stat = pulp.LpStatus[mod.status]
    print("Status is %s" % (stat))
    if stat != "Optimal":
        return stat
    results = []
    for i in cases:
        results.append( dec_vars[i].varValue )
    return pd.Series(results)

#Test it out

######################################
#Creating a simple fake dataset

ids = ['A','B','C','D','E']
pr = [0.25,0.5, 0.1, 0.9, 0.8]
val = [100, 20, 30, 10, 25]
z_list = list(zip(ids,pr,val))

my_df = pd.DataFrame(z_list, columns=['Ids','pr','val'])
my_df['ER'] = my_df['pr'] * my_df['val']


my_df['Chose'] = sel_cases(my_df, 'pr', 'val', 0.5, 2)
print(my_df)
######################################

######################################
#Test on much bigger dataset

np.random.seed(10)
tot_n = 100000
sim_dat = pd.DataFrame({'Id': range(tot_n)})
sim_dat['pr'] = np.random.rand(tot_n)
sim_dat['ll'] = np.exp(4 - 0.8*sim_dat['pr'])
sim_dat['val'] = np.random.poisson(lam=sim_dat['ll'])
print( sim_dat.describe() )

sim_dat['Chose'] = sel_cases(sim_dat,'pr','val', 0.05, 1000)
sub_dat = sim_dat[sim_dat['Chose'] == 1]
print( sub_dat.describe() )

#On laptop
#    10,000 cases, choose  100, instant
#   100,000 cases, choose 1000, only a few seconds
# 1,000,000 cases, choose 5000, about 4 minutes 

#Check to make sure it behaves as expected
cum_mean = 0
sim_n = 1000
for i in range(sim_n):
    rout = np.random.binomial(1, sub_dat['pr'])
    sim_mean = 1 - rout.mean()
    cum_mean += sim_mean
    #print( sim_mean )

print("")
print("False Positives over Simulation")
cum_mean = cum_mean/sim_n
print(cum_mean)
#yep, this is fine
######################################



########################################
# Soft constraint on false positive rate????
# Fix N?????? or set upper bound on N (may not be feasible????)
# Value Added Risk models (for claims/probs with range?????)

# This requires the input model to be well calibrated!!!!
# Threshold (as probabilities) vs fpr should be a straight 1-1 line!
# What if not, should I pass in fpr's for each predicted prob theshold?

########################################