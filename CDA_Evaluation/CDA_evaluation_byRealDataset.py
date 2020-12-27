# Dataset used in the following code:
# M. Girvan and M. E. J. Newman, Community structure in social and biological networks,
#Proc. Natl. Acad. Sci. USA 99, 7821-7826 (2002).

from cdlib import algorithms as cd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
import time

def extract_communities_list(communities_list, G):
    community_labels = np.zeros(N)
    for comm_index, comm_nodes in enumerate(communities_list):
        for node in comm_nodes:
            community_labels[list(G).index(node)] = comm_index
    return community_labels

gml_file = open("football.gml", "r")

gml_list = []
for line in gml_file:
    gml_list.append(line)
gml_list = [item.replace('\n', '') for item in gml_list]

G = nx.parse_gml(gml_list)
N = G.number_of_nodes()
dt_g = nx.get_node_attributes(G, 'value')
true_communities = list(dt_g.values())

plt.figure()
plt.title("American football games labeled by conference")
nx.draw(G, nx.spring_layout(G), node_color=true_communities, cmap=plt.cm.get_cmap('rainbow'), node_size=70)
plt.show()
'''
############################### Infomap ###############################
start_time = time.time()
infomap_partition = cd.infomap(G)
infomap_time = time.time() - start_time
infomap_communities = extract_communities_list(infomap_partition.communities, G)
nmi_infomap = normalized_mutual_info_score(true_communities, infomap_communities)

############################### Leading Eigenvector ###############################
start_time = time.time()
eigenvector_partition = cd.eigenvector(G)
eifenvector_time = time.time() - start_time
eigenvector_communities = extract_communities_list(eigenvector_partition.communities, G)
nmi_eigenvector = normalized_mutual_info_score(true_communities, eigenvector_communities)

############################### Louvian ###############################
start_time = time.time()
louvian_partition = cd.louvain(G)
louvian_time = time.time() - start_time
louvian_communities = extract_communities_list(louvian_partition.communities, G)
nmi_louvian = normalized_mutual_info_score(true_communities, louvian_communities)

############################### Leiden ###############################
start_time = time.time()
leiden_partition = cd.leiden(G)
leiden_time = time.time() - start_time
leiden_communities = extract_communities_list(leiden_partition.communities, G)
nmi_ledien = normalized_mutual_info_score(true_communities, leiden_communities)

############################### Walktrap ###############################
start_time = time.time()
walktrap_partition = cd.walktrap(G)
walktrap_time = time.time() - start_time
walktrap_communities = extract_communities_list(walktrap_partition.communities, G)
nmi_walktrap = normalized_mutual_info_score(true_communities, walktrap_communities)

############################### Markov Clustering ###############################
start_time = time.time()
markov_partition = cd.markov_clustering(G)
markov_time = time.time() - start_time
markov_communities = extract_communities_list(markov_partition.communities, G)
nmi_markov = normalized_mutual_info_score(true_communities, markov_communities)

############################### Greedy ###############################
start_time = time.time()
greedy_partition = cd.greedy_modularity(G)
greedy_time = time.time() - start_time
greedy_communities = extract_communities_list(greedy_partition.communities, G)
nmi_greedy = normalized_mutual_info_score(true_communities, greedy_communities)

############################### Label Propagation ###############################
start_time = time.time()
propagation_partition = cd.label_propagation(G)
propagation_time = time.time() - start_time
propagation_communities = extract_communities_list(propagation_partition.communities, G)
nmi_propagation = normalized_mutual_info_score(true_communities, propagation_communities)

'''
'''
#Plot NMI score
plt.grid(zorder=0)
plt.bar(["Infomap", "Eigenvector", "Louvian", "Leiden", "Walktrap", "Markov", "Greedy", "LPA"],
         [nmi_infomap, nmi_eigenvector, nmi_louvian, nmi_ledien, nmi_walktrap, nmi_markov, nmi_greedy, nmi_propagation],
         color='#266ed9',zorder=3)
plt.ylabel("NMI")
plt.title("Algorithms score for the football games dataset")
plt.xticks(rotation=45, ha="right")
plt.show()
'''
'''
#Plot time
plt.grid(zorder=0)
plt.bar(["Infomap", "Eigenvector", "Louvian", "Leiden", "Walktrap", "Markov", "Greedy", "LPA"],
         [infomap_time, eifenvector_time, louvian_time, leiden_time, walktrap_time, markov_time, greedy_time, propagation_time],
         color='#de8221',zorder=3)
plt.ylabel("Time[s]")
plt.title("Algorithms execution time for the football games dataset")
plt.xticks(rotation=45, ha="right")
plt.show()
'''