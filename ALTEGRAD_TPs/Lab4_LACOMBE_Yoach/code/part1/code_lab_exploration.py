"""
Graph Mining - ALTEGRAD - Dec 2021
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

G = nx.readwrite.edgelist.read_edgelist(PATH, delimiter = "\t")
print('number of nodes', len(G.nodes))
print('number of edges', len(G.edges))



############## Task 2

largest_cc = max(nx.connected_components(G), key=len)
print("number of connected components", nx.number_connected_components(G))
print("number of nodes in the largest connected component of G:",len(largest_cc))

sub_G = G.subgraph(largest_cc)
print("number of edges in the largest connected component of G:",sub_G.number_of_edges())



############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

print("min", np.min(degree_sequence))
print("max", np.max(degree_sequence))
print("mean", np.mean(degree_sequence))
print("median", np.median(degree_sequence))



############## Task 4

plt.bar(np.arange(len(nx.degree_histogram(G))),nx.degree_histogram(G))
plt.show()

plt.loglog(nx.degree_histogram(G))
plt.xlabel("log(degree)")
plt.ylabel("log(frequency)")
plt.show()

############## Task 5

print("Global clustering coeff", nx.transitivity(G))