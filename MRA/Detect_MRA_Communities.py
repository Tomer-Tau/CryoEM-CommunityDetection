from cdlib import algorithms as cd
import MRA_Graphs
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_cd
import matplotlib.pyplot as plt
import igraph as ig

def extract_communities_list(communities_list, N):
    community_labels = np.zeros(N)
    for comm_index, comm_nodes in enumerate(communities_list):
        for node in comm_nodes:
            community_labels[node] = comm_index
    return community_labels

sigma_list = np.array(range(11))/10
N = 100

################## Rectangle&Triangle MRA ##################
greedy_nmi = []
leiden_nmi = []
louvain_nmi = []
cpm_nmi = []
lpa_nmi = []
infomap_nmi = []
leading_eigenvector_nmi = []
multilevel_nmi = []
edge_betweenness_nmi = []
spinglass_nmi = []
walktrap_nmi = []

for sigma in sigma_list:
    G, true_labels = MRA_Graphs.MRA_Rect_Trian(N, L=50, K=2, sigma=sigma)

    # Partition Graph by various methods
    greedy = cd.greedy_modularity(G, weight='weight')
    greedy_labels = extract_communities_list(greedy.communities, N)
    greedy_nmi.append(normalized_mutual_info_score(true_labels, greedy_labels))

    leiden = cd.leiden(G, weights='weight')
    leiden_labels = extract_communities_list(leiden.communities, N)
    leiden_nmi.append(normalized_mutual_info_score(true_labels, leiden_labels))

    louvain = cd.louvain(G, weight='weight')
    louvain_labels = extract_communities_list(louvain.communities, N)
    louvain_nmi.append(normalized_mutual_info_score(true_labels, louvain_labels))

    cpm = cd.cpm(G, weights='weight')
    cpm_labels = extract_communities_list(cpm.communities, N)
    cpm_nmi.append(normalized_mutual_info_score(true_labels, cpm_labels))

    lpa = nx_cd.asyn_lpa_communities(G, weight='weight')
    lpa_labels = extract_communities_list(list(lpa), N)
    lpa_nmi.append(normalized_mutual_info_score(true_labels, lpa_labels))

    #Convert NetworkX graph to Igraph graph
    G = ig.Graph.from_networkx(G)

    infomap = ig.Graph.community_infomap(G, edge_weights="weight")
    infomap_nmi.append(normalized_mutual_info_score(true_labels, infomap.membership))

    leading_eigenvector = ig.Graph.community_leading_eigenvector(G, weights="weight")
    leading_eigenvector_nmi.append(normalized_mutual_info_score(true_labels, leading_eigenvector.membership))

    multilevel = ig.Graph.community_multilevel(G, weights="weight")
    multilevel_nmi.append(normalized_mutual_info_score(true_labels, multilevel.membership))

    edge_betweenness = ig.Graph.community_edge_betweenness(G, weights="weight")
    edge_betweenness = edge_betweenness.as_clustering()
    edge_betweenness_nmi.append(normalized_mutual_info_score(true_labels, edge_betweenness.membership))

    spinglass = ig.Graph.community_spinglass(G, weights="weight")
    spinglass_nmi.append(normalized_mutual_info_score(true_labels, spinglass.membership))

    walktrap = ig.Graph.community_walktrap(G, weights="weight")
    walktrap = walktrap.as_clustering()
    walktrap_nmi.append(normalized_mutual_info_score(true_labels, walktrap.membership))

plt.plot(greedy_nmi,label="greedy")
plt.plot(leiden_nmi,label="leiden")
plt.plot(louvain_nmi,label="louvain")
plt.plot(cpm_nmi,label="CPM")
plt.plot(lpa_nmi,label="LPA")
plt.plot(infomap_nmi,label="infomap")
plt.plot(leading_eigenvector_nmi,label="leading eigenvector")
plt.plot(multilevel_nmi,label="multilevel")
plt.plot(edge_betweenness_nmi,label="edge betweenness")
plt.plot(spinglass_nmi,label="spinglass")
plt.plot(walktrap_nmi,label="walktrap")
plt.xticks(list(range(len(sigma_list))), sigma_list)
plt.title("Triangle&Rectangle MRA")
plt.xlabel("Noise level (sigma)")
plt.ylabel("NMI")
plt.legend()
plt.show()

################## Standard Normal MRA ##################
greedy_nmi = []
leiden_nmi = []
louvain_nmi = []
cpm_nmi = []
lpa_nmi = []
infomap_nmi = []
leading_eigenvector_nmi = []
multilevel_nmi = []
edge_betweenness_nmi = []
spinglass_nmi = []
walktrap_nmi = []

for sigma in sigma_list:
    G, true_labels = MRA_Graphs.MRA_StandardNormal(N, L=50, K=2, sigma=sigma)

    # Partition Graph by various methods
    greedy = cd.greedy_modularity(G, weight='weight')
    greedy_labels = extract_communities_list(greedy.communities, N)
    greedy_nmi.append(normalized_mutual_info_score(true_labels, greedy_labels))

    leiden = cd.leiden(G, weights='weight')
    leiden_labels = extract_communities_list(leiden.communities, N)
    leiden_nmi.append(normalized_mutual_info_score(true_labels, leiden_labels))

    louvain = cd.louvain(G, weight='weight')
    louvain_labels = extract_communities_list(louvain.communities, N)
    louvain_nmi.append(normalized_mutual_info_score(true_labels, louvain_labels))

    cpm = cd.cpm(G, weights='weight')
    cpm_labels = extract_communities_list(cpm.communities, N)
    cpm_nmi.append(normalized_mutual_info_score(true_labels, cpm_labels))

    lpa = nx_cd.asyn_lpa_communities(G, weight='weight')
    lpa_labels = extract_communities_list(list(lpa), N)
    lpa_nmi.append(normalized_mutual_info_score(true_labels, lpa_labels))

    # Convert NetworkX graph to Igraph graph
    G = ig.Graph.from_networkx(G)

    infomap = ig.Graph.community_infomap(G, edge_weights="weight")
    infomap_nmi.append(normalized_mutual_info_score(true_labels, infomap.membership))

    leading_eigenvector = ig.Graph.community_leading_eigenvector(G, weights="weight")
    leading_eigenvector_nmi.append(normalized_mutual_info_score(true_labels, leading_eigenvector.membership))

    multilevel = ig.Graph.community_multilevel(G, weights="weight")
    multilevel_nmi.append(normalized_mutual_info_score(true_labels, multilevel.membership))

    edge_betweenness = ig.Graph.community_edge_betweenness(G, weights="weight")
    edge_betweenness = edge_betweenness.as_clustering()
    edge_betweenness_nmi.append(normalized_mutual_info_score(true_labels, edge_betweenness.membership))

    spinglass = ig.Graph.community_spinglass(G, weights="weight")
    spinglass_nmi.append(normalized_mutual_info_score(true_labels, spinglass.membership))

    walktrap = ig.Graph.community_walktrap(G, weights="weight")
    walktrap = walktrap.as_clustering()
    walktrap_nmi.append(normalized_mutual_info_score(true_labels, walktrap.membership))

plt.plot(greedy_nmi, label="greedy")
plt.plot(leiden_nmi, label="leiden")
plt.plot(louvain_nmi, label="louvain")
plt.plot(cpm_nmi, label="CPM")
plt.plot(lpa_nmi, label="LPA")
plt.plot(infomap_nmi, label="infomap")
plt.plot(leading_eigenvector_nmi, label="leading eigenvector")
plt.plot(multilevel_nmi, label="multilevel")
plt.plot(edge_betweenness_nmi, label="edge betweenness")
plt.plot(spinglass_nmi, label="spinglass")
plt.plot(walktrap_nmi, label="walktrap")
plt.xticks(list(range(len(sigma_list))), sigma_list)
plt.title("Standard Normal MRA")
plt.xlabel("Noise level (sigma)")
plt.ylabel("NMI")
plt.legend()
plt.show()

################## Random Normal MRA ##################
greedy_nmi = []
leiden_nmi = []
louvain_nmi = []
cpm_nmi = []
lpa_nmi = []
infomap_nmi = []
leading_eigenvector_nmi = []
multilevel_nmi = []
edge_betweenness_nmi = []
spinglass_nmi = []
walktrap_nmi = []

for sigma in sigma_list:
    G, true_labels = MRA_Graphs.MRA_RandomNormal(N, L=50, K=2, sigma=sigma)

    # Partition Graph by various methods
    greedy = cd.greedy_modularity(G, weight='weight')
    greedy_labels = extract_communities_list(greedy.communities, N)
    greedy_nmi.append(normalized_mutual_info_score(true_labels, greedy_labels))

    leiden = cd.leiden(G, weights='weight')
    leiden_labels = extract_communities_list(leiden.communities, N)
    leiden_nmi.append(normalized_mutual_info_score(true_labels, leiden_labels))

    louvain = cd.louvain(G, weight='weight')
    louvain_labels = extract_communities_list(louvain.communities, N)
    louvain_nmi.append(normalized_mutual_info_score(true_labels, louvain_labels))

    cpm = cd.cpm(G, weights='weight')
    cpm_labels = extract_communities_list(cpm.communities, N)
    cpm_nmi.append(normalized_mutual_info_score(true_labels, cpm_labels))

    lpa = nx_cd.asyn_lpa_communities(G, weight='weight')
    lpa_labels = extract_communities_list(list(lpa), N)
    lpa_nmi.append(normalized_mutual_info_score(true_labels, lpa_labels))

    # Convert NetworkX graph to Igraph graph
    G = ig.Graph.from_networkx(G)

    infomap = ig.Graph.community_infomap(G, edge_weights="weight")
    infomap_nmi.append(normalized_mutual_info_score(true_labels, infomap.membership))

    leading_eigenvector = ig.Graph.community_leading_eigenvector(G, weights="weight")
    leading_eigenvector_nmi.append(normalized_mutual_info_score(true_labels, leading_eigenvector.membership))

    multilevel = ig.Graph.community_multilevel(G, weights="weight")
    multilevel_nmi.append(normalized_mutual_info_score(true_labels, multilevel.membership))

    edge_betweenness = ig.Graph.community_edge_betweenness(G, weights="weight")
    edge_betweenness = edge_betweenness.as_clustering()
    edge_betweenness_nmi.append(normalized_mutual_info_score(true_labels, edge_betweenness.membership))

    spinglass = ig.Graph.community_spinglass(G, weights="weight")
    spinglass_nmi.append(normalized_mutual_info_score(true_labels, spinglass.membership))

    walktrap = ig.Graph.community_walktrap(G, weights="weight")
    walktrap = walktrap.as_clustering()
    walktrap_nmi.append(normalized_mutual_info_score(true_labels, walktrap.membership))

plt.plot(greedy_nmi, label="greedy")
plt.plot(leiden_nmi, label="leiden")
plt.plot(louvain_nmi, label="louvain")
plt.plot(cpm_nmi, label="CPM")
plt.plot(lpa_nmi, label="LPA")
plt.plot(infomap_nmi, label="infomap")
plt.plot(leading_eigenvector_nmi, label="leading eigenvector")
plt.plot(multilevel_nmi, label="multilevel")
plt.plot(edge_betweenness_nmi, label="edge betweenness")
plt.plot(spinglass_nmi, label="spinglass")
plt.plot(walktrap_nmi, label="walktrap")
plt.xticks(list(range(len(sigma_list))), sigma_list)
plt.title("Random Normal MRA")
plt.xlabel("Noise level (sigma)")
plt.ylabel("NMI")
plt.legend()
plt.show()