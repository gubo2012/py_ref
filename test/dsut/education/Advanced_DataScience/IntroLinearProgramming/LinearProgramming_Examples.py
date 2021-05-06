'''
Linear programming examples
Andy Wheeler, andrew.wheeler@hms.com
'''

#This is the library I like to use
#Many exist though
import pulp

#These are only necessary for simulating data
#You can pass in lists to pulp
import numpy as np
import pandas as pd

#############################################
#Simple Example

#Creating data
prob = [0.05, 0.10, 0.50]
over = [10000, 3000, 500]
exp_ret = [p*o for p,o in zip(prob,over)]
case_index = list(range(len(prob)))
tot_audit = 2 #total number of claims to select
hit_rate = 0.2 #finding rate constraint

#This is the model
P = pulp.LpProblem("Choosing Cases to Select", pulp.LpMaximize)

#These are the binary decision variables
D = pulp.LpVariable.dicts("Decision Variable", [i for i in case_index],
	                      lowBound=0, upBound=1, cat=pulp.LpInteger)

#Objective Function
P += pulp.lpSum( D[i]*exp_ret[i] for i in case_index)

#Constraint on total number of claims selected
P += pulp.lpSum( D[i] for i in case_index ) == tot_audit

#Constraint on the overall hit rate
P += pulp.lpSum( D[i]*prob[i] for i in case_index ) >= hit_rate*tot_audit

#Solve the problem
P.solve()

#Get the decision variables
dec_list = [D[i].varValue for i in case_index]

#Turning lists into data frame for easier computation
simple_dat = pd.DataFrame(zip(prob, over, exp_ret, dec_list),
	                      columns=['prob', 'over', 'exp_ret', 'selected'])

#Expected return
print( (simple_dat['exp_ret'] * simple_dat['selected']).sum() )

#Should be the same
print( pulp.value(P.objective) )

#Hit rate of selected cases
print( (simple_dat['prob'] * simple_dat['selected']).sum()/tot_audit )
#############################################


#############################################
#More complicated example data
#Has example providers as well

np.random.seed(10)
n = 20000 #total number of cases
underpay_est = np.random.lognormal(mean=7.6,sigma=0.5,size=n)
#about 25% overall finding rate
prob_over = np.random.beta(2.5,10,size=n)
exp_return = prob_over*underpay_est

#providers have a differential probability of being selected
prov = list('ABCDE')
prov_n = len(prov)
prov_index = list(range(1,prov_n+1))
prov_tot = sum(prov_index)
prov_prob = [i/prov_tot for i in prov_index]
prov_claims = np.random.choice(a=prov, size=n, replace=True, p=prov_prob)

sim_dat = pd.DataFrame(zip(underpay_est, prob_over, 
                           exp_return, prov_claims),
	                   columns=['underpay_est', 'prob_over', 
                                'exp_return', 'prov_claims'])

##############################################


#############################################
#Setting up a function to do provider constraints 

def selection_model(er,prob,provider,data,cases_const,finding_const,provider_const):
    #Preparing simpler lists
    er_list = list(data[er])
    min_const_list = list( data[prob] - finding_const )
    prov_list = list(data[provider])
    index_list = list(range(len(er_list)))
    #getting the locations for each provider in the data
    all_prov = set(prov_list)
    prov_loc = {}
    for p in all_prov:
        prov_loc[p] = list( np.where(data[provider] == p)[0])
    #Now setting up the model
    Sel_Mod = pulp.LpProblem("Selection Model", pulp.LpMaximize)
    Dec_Vars = pulp.LpVariable.dicts("Selected Cases", 
                                     [i for i in sim_dat.index],
                                     lowBound=0, upBound=1,
                                     cat=pulp.LpInteger)
    #Objective Function
    Sel_Mod += pulp.lpSum( Dec_Vars[i]*er_list[i] for i in index_list)
    #Constraint on total number of claims selected, has to be equal or fewer
    Sel_Mod += pulp.lpSum( Dec_Vars[i] for i in index_list ) <= cases_const
    #Constraint on finding rate, taking into account total cases selected
    Sel_Mod += pulp.lpSum( Dec_Vars[i]*min_const_list[i] for i in index_list ) >= 0
    #Provider constraints
    for p in all_prov:
        Sel_Mod += pulp.lpSum( Dec_Vars[i] for i in prov_loc[p] ) <= provider_const
    #Solve the Problem
    Sel_Mod.solve()
    #Get the decision variables
    dec_list = [Dec_Vars[i].varValue for i in index_list]
    return dec_list

#If we set the provider constraints to a very high 
#number (so no constraints) we get a few providers
#selected too often
sim_dat['Selected1'] = selection_model(er='exp_return',prob='prob_over',provider='prov_claims',
                                       data=sim_dat,cases_const=1000,finding_const=0.3,
                                       provider_const=1000)

print( sim_dat.loc[ sim_dat['Selected1'] == 1, 'prov_claims'].value_counts() )

#But if we run it with provider constraints
sim_dat['Selected2'] = selection_model(er='exp_return',prob='prob_over',provider='prov_claims',
                                       data=sim_dat,cases_const=1000,finding_const=0.3,
                                       provider_const=250)

print( sim_dat.loc[ sim_dat['Selected2'] == 1, 'prov_claims'].value_counts() )
#############################################

#############################################
#We can see how changing the finding rate constraint
#Effects our estimates of the expected return

hit_rate = np.linspace(0.30, 0.50, 20)
total_return = []

for h in hit_rate:
    sel_list = selection_model(er='exp_return',prob='prob_over',provider='prov_claims',
                               data=sim_dat,cases_const=1000,finding_const=h,
                               provider_const=250)
    sel_np = np.asarray(sel_list)
    est_ret = (sel_np * sim_dat['exp_return']).sum()
    print(f'running hit rate {h}, return is {est_ret}')
    total_return.append(est_ret)

#Now make a graph
    
##################
import matplotlib.pyplot as plt
#importing dataviz from ds-utilities folder
#https://github.com/hmsholdings/data-science-utils/tree/master/plt
import sys
sys.path.append(r'C:\Users\e009156\Documents\GitHub\data-science-utils\plt')
import hms_plotstyle
hms_plotstyle.hms_plots()
##################

#Now can make a nice matplotlib plot
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(hit_rate, total_return)
ax.set_xlabel('Finding Rate Constraint')
ax.set_ylabel('Expected Return')
#plt.savefig(r'C:\Users\e009156\Documents\GitHub\data-science-utils\education\Advanced_DataScience\IntroLinearProgramming\TradeOff_FindingRate.png', dpi=500, bbox_inches='tight')
plt.show()
#############################################