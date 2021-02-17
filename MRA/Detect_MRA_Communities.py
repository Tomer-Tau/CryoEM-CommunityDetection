from cdlib import algorithms as cd
import MRA_Graphs
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
import networkx.algorithms.community as nx_cd
import matplotlib.pyplot as plt

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

plt.plot(greedy_nmi,label="greedy")
plt.plot(leiden_nmi,label="leiden")
plt.plot(louvain_nmi,label="louvain")
plt.plot(cpm_nmi,label="CPM")
plt.plot(lpa_nmi,label="LPA")
plt.xticks(list(range(len(sigma_list))), sigma_list)
plt.title("Triangle&Rectangle MRA (w/ edge threshold)")
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

plt.plot(greedy_nmi,label="greedy")
plt.plot(leiden_nmi,label="leiden")
plt.plot(louvain_nmi,label="louvain")
plt.plot(cpm_nmi,label="CPM")
plt.plot(lpa_nmi,label="LPA")
plt.xticks(list(range(len(sigma_list))), sigma_list)
plt.title("Standard Normal MRA (w/ edge threshold)")
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

plt.plot(greedy_nmi,label="greedy")
plt.plot(leiden_nmi,label="leiden")
plt.plot(louvain_nmi,label="louvain")
plt.plot(cpm_nmi,label="CPM")
plt.plot(lpa_nmi,label="LPA")
plt.xticks(list(range(len(sigma_list))), sigma_list)
plt.title("Random Normal MRA (w/ edge threshold)")
plt.xlabel("Noise level (sigma)")
plt.ylabel("NMI")
plt.legend()
plt.show()