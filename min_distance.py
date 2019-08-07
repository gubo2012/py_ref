graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0], 
           [4, 0, 8, 0, 0, 0, 0, 11, 0], 
           [0, 8, 0, 7, 0, 4, 0, 0, 2], 
           [0, 0, 7, 0, 9, 14, 0, 0, 0], 
           [0, 0, 0, 9, 0, 10, 0, 0, 0], 
           [0, 0, 4, 14, 10, 0, 2, 0, 0], 
           [0, 0, 0, 0, 0, 2, 0, 1, 6], 
           [8, 11, 0, 0, 0, 0, 1, 0, 7], 
           [0, 0, 2, 0, 0, 0, 6, 7, 0] 
          ]; 


def find_min_node(min_dists):
    m_dist = max_dist
    for k, v in min_dists.items():
        if v < m_dist:
            node = k
            m_dist = v
    return node


def update_spt(node):
	if node == 0:
		spt[node] = 0
	else:
		parent = min_dists_parent[node]
		spt[node] = spt[parent] + graph[parent][node]

	for j in range(n):
		if (j in min_dists) and graph[node][j] > 0:
			new_dist = spt[node] + graph[node][j]
			if new_dist < min_dists[j]:
				min_dists[j] = new_dist
				min_dists_parent[j] = node

def get_path(node):
    parent = min_dists_parent[node]
    if parent == 0:
        return str(parent) 
    else:        
        return get_path(parent) + '->' + str(parent)
    

print(graph)         
max_dist = 9999
n = len(graph)

min_dists = {}
min_dists_parent = {}
for i in range(n):
	min_dists[i] = max_dist
	min_dists_parent[i] = 0
min_dists[0] = 0

spt = {}

while len(min_dists)>0:

    node = find_min_node(min_dists)
    update_spt(node)
    print(node, spt[node])
    min_dists.pop(node)


print('final results')
for i in range(len(graph)):
	print(i, spt[i], get_path(i))






