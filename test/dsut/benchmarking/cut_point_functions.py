'''
Using cost-benefit analysis to set a treshold
For predictive models based on true positives
and false positives

Andrew Wheeler
'''

import numpy as np

#May want to expand to include the full cost matrix (true negatives/false negatives)
def util_cut_point(tp,fp,th,pr,be,co,curve=False):
    """
    Returns the optimal cut point given costs/benefits and prevalence population
    estimates
    
    Parameters
    ----------
    
    # Take these straight from sklearn.metrics.roc_curve()
    tp : numpy array of true positive rates
    fp : numpy array of false positive rates
    th : numpy array of thresholds based on prediction model
    
    # These you need to figure out for the local problem
    pr : scaler (float) of prevalence estimate in pop
    be : scaler (float/int) benefit of identifying true positive case (should be positive)
    co : scaler (float/int) cost of false positive (should be negative)
    
    # Optional
    curve : Boolean whether to return entire utility curve, default False
    
    Returns
    -------
    cut_point : float of the optimal cut point
    util_scores : optional numpy array of utility curve along entire trp/fpr/thresholds
    """
    util_scores = pr*tp*be + (1-pr)*fp*co
    cut_point = th[np.argmax(util_scores)]
    #return the full utility curve if curve=True
    if curve:
        return cut_point, util_scores
    else:
        return cut_point
     
#Note, this does not handle ties for best cost (just picks the first point)
#If the ROC function is monotonic should not be much of a problem
#Picking first point will pick a lower threshold, so flag MORE cases by default (if ties)

#Big data notes, may want to cache pr*tp and (1-pr)*fp (which should be small & fit in memory) 
#then redo with many different costs/benefits and/or memoize function