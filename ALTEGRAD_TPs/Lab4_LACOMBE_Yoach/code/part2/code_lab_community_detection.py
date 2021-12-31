"""
Graph Mining - ALTEGRAD - Dec 2021
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from random import randint
from sklearn.cluster import KMeans


#eye, diags

############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
  d = k

  n = G.number_of_nodes()

  degree_sequence = [1/G.degree(node) for node in G.nodes()]

  laplacian = np.identity(n) - np.diag(degree_sequence)@nx.adjacency_matrix(G)

  eigenvalues, eigenvectors = eigs(laplacian, which = 'SR', k = d)
  eigenvectors = np.real(eigenvectors)
  

  kmeans = KMeans(n_clusters=k).fit(eigenvectors)


  clustering = {}
  for i,node in enumerate(G.nodes()):
    clustering[node] = kmeans.labels_[i]
  return clustering



############## Task 7
k = 50

clustering = spectral_clustering(sub_G, k)
print(clustering)




############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
  m = G.number_of_edges()

  clusters = set(clustering.values())
  modularity = 0

  for cluster in clusters:
    nodes_in_cluster = [node for node in G.nodes() if clustering[node] == cluster]
    
    cluster_graph = G.subgraph(nodes_in_cluster)
    
    lc = cluster_graph.number_of_edges()

    dc = 0
    for node in nodes_in_cluster:
      dc += G.degree(node)
    
    modularity += lc/m -(dc/(2*m))**2
  return modularity



############## Task 9


print("modularity of the giant connected component", modularity(sub_G, clustering))

random_clustering = {}
for node in G.nodes():
  random_clustering[node] = randint(0,49)

print("modularity of the random graph", modularity(sub_G, random_clustering))
