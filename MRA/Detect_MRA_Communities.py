from cdlib import algorithms as cd
import MRA_Graphs
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
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
for sigma in sigma_list:
    G_trian_rect, true_labels = MRA_Graphs.MRA_Rect_Trian(N, L=50, K=2, sigma=sigma)

    # Partition Graph by various methods
    greedy = cd.greedy_modularity(G_trian_rect, weight='weight')
    greedy_labels = extract_communities_list(greedy.communities, N)
    greedy_nmi.append(normalized_mutual_info_score(true_labels, greedy_labels))

    leiden = cd.leiden(G_trian_rect, weights='weight')
    leiden_labels = extract_communities_list(leiden.communities, N)
    leiden_nmi.append(normalized_mutual_info_score(true_labels, leiden_labels))

    louvain = cd.louvain(G_trian_rect, weight='weight')
    louvain_labels = extract_communities_list(louvain.communities, N)
    louvain_nmi.append(normalized_mutual_info_score(true_labels, louvain_labels))

plt.plot(greedy_nmi,label="greedy")
plt.plot(leiden_nmi,label="leiden")
plt.plot(louvain_nmi,label="louvain")
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
for sigma in sigma_list:
    G_std, true_labels = MRA_Graphs.MRA_StandardNormal(N, L=50, K=2, sigma=sigma)

    # Partition Graph by various methods
    greedy = cd.greedy_modularity(G_std, weight='weight')
    greedy_labels = extract_communities_list(greedy.communities, N)
    greedy_nmi.append(normalized_mutual_info_score(true_labels, greedy_labels))

    leiden = cd.leiden(G_std, weights='weight')
    leiden_labels = extract_communities_list(leiden.communities, N)
    leiden_nmi.append(normalized_mutual_info_score(true_labels, leiden_labels))

    louvain = cd.louvain(G_std, weight='weight')
    louvain_labels = extract_communities_list(louvain.communities, N)
    louvain_nmi.append(normalized_mutual_info_score(true_labels, louvain_labels))

plt.plot(greedy_nmi,label="greedy")
plt.plot(leiden_nmi,label="leiden")
plt.plot(louvain_nmi,label="louvain")
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
for sigma in sigma_list:
    G_std, true_labels = MRA_Graphs.MRA_RandomNormal(N, L=50, K=2, sigma=sigma)

    # Partition Graph by various methods
    greedy = cd.greedy_modularity(G_std, weight='weight')
    greedy_labels = extract_communities_list(greedy.communities, N)
    greedy_nmi.append(normalized_mutual_info_score(true_labels, greedy_labels))

    leiden = cd.leiden(G_std, weights='weight')
    leiden_labels = extract_communities_list(leiden.communities, N)
    leiden_nmi.append(normalized_mutual_info_score(true_labels, leiden_labels))

    louvain = cd.louvain(G_std, weight='weight')
    louvain_labels = extract_communities_list(louvain.communities, N)
    louvain_nmi.append(normalized_mutual_info_score(true_labels, louvain_labels))

plt.plot(greedy_nmi,label="greedy")
plt.plot(leiden_nmi,label="leiden")
plt.plot(louvain_nmi,label="louvain")
plt.xticks(list(range(len(sigma_list))), sigma_list)
plt.title("Random Normal MRA")
plt.xlabel("Noise level (sigma)")
plt.ylabel("NMI")
plt.legend()
plt.show()