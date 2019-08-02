# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:27:42 2019

@author: gubo
"""

#https://ericphanson.com/blog/2016/the-traveling-salesman-and-10-lines-of-python/

import random, numpy, math, copy, matplotlib.pyplot as plt

random.seed(1024)

n_cities = 15



cities = [random.sample(range(100), 2) for x in range(n_cities)];
tour = random.sample(range(n_cities),n_cities);

def cal_distance(tour, full_flag = False):
    if full_flag:
        distance = sum([ math.sqrt(sum([(cities[tour[(k+1) % n_cities]][d] - cities[tour[k % n_cities]][d])**2 for d in [0,1] ])) for k in range(n_cities)])
    else:
        distance = sum([ math.sqrt(sum([(cities[tour[(k+1) % n_cities]][d] - cities[tour[k % n_cities]][d])**2 for d in [0,1] ])) for k in [j,j-1,i,i-1]])
    return distance

count = 0
for temperature in numpy.logspace(0,5,num=100000)[::-1]:
    
#    swap randomly selected two cities
    [i,j] = sorted(random.sample(range(n_cities),2));
    
#    newTour =  tour[:i] + tour[j:j+1] +  tour[i+1:j] + tour[i:i+1] + tour[j+1:];
#    easier to read
    a, b = tour[i], tour[j]
    newTour = tour.copy()
    newTour[i], newTour[j] = b, a
    
#7   if math.exp( ( sum([ math.sqrt(sum([(cities[tour[(k+1) % n_cities]][d] - cities[tour[k % n_cities]][d])**2 for d in [0,1] ])) for k in [j,j-1,i,i-1]]) - sum([math.sqrt(sum([(cities[newTour[(k+1) % n_cities]][d] - cities[newTour[k % n_cities]][d])**2 for d in [0,1] ])) for k in [j,j-1,i,i-1]])) / temperature) > random.random():
#8       tour = copy.copy(newTour);
    
    oldDistances =  cal_distance(tour)
    newDistances = cal_distance(newTour)

    if math.exp( ( oldDistances - newDistances) / temperature) > random.random():
#    if oldDistances > newDistances:
#        tour = copy.copy(newTour)
        tour = newTour.copy()
    
    
    if count % 1000 == 0:
        print(count, cal_distance(tour, True))
    count +=1 

plt.plot([cities[tour[i % n_cities]][0] for i in range(16)], [cities[tour[i % n_cities]][1] for i in range(16)], 'xb-');
plt.show()