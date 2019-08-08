# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:30:32 2019

@author: dxuser22
"""

from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.g = defaultdict(list)
        
    def addEdge(self, parent, child):
        self.g[parent].append(child)
        
    def bfs(self, q, visited): # p parent
        
        print(q)
        
        if (len(q)) == 0:
            print('path', visited)
        else:
            p = q.popleft()
            print(p)
            if len(self.g[p]) > 0: # with c
                for v in self.g[p]:
                    if v not in visited:
                        q.append(v)
                        visited.append(v)
            self.bfs(q, visited)         
            
    def dfs(self, q, visited):
                
        if (len(q)) == 0:
            print('path', visited)
        else:
            p = q.popleft()
            print(p)
            
            if len(self.g[p]) > 0: # with c
                for v in self.g[p]:
                    if v not in visited:
                        q.appendleft(v)
                        visited.append(v)
                        self.dfs(q, visited)  
        
   

g = Graph() 
g.addEdge(0, 1) 
g.addEdge(0, 2) 
g.addEdge(1, 2) 
g.addEdge(2, 0) 
g.addEdge(2, 3) 
g.addEdge(3, 3) 

print(g.g)

#q = deque()
#q.append(0)
#visited = [0]
#g.bfs(q, visited)

q = deque()
q.append(2)
visited = [2]
g.dfs(q, visited)