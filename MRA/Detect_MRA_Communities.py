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

def plot_nmi_scores(N, sigma_list, samples_num, MRA_type):
    nmi_scores = {"leiden": [], "louvain": [], "leading_eigenvector": [], "multilevel": [], "spinglass": [],
                  "walktrap": []}
    for sigma in sigma_list:
        nmi_samples = {"leiden": [], "louvain": [], "leading_eigenvector": [], "multilevel": [], "spinglass": [],
                       "walktrap": []}

        print("sigma: {}".format(sigma))

        # Calculate NMI score for a number (samples_num) of random graphs
        for i in range(samples_num):
            # Generate appropriate MRA graph
            if MRA_type == "Rect_Trian": G, true_labels = MRA_Graphs.MRA_Rect_Trian(N, L=50, K=2, sigma=sigma)
            elif MRA_type == "Standard Normal": G, true_labels = MRA_Graphs.MRA_StandardNormal(N, L=50, K=10, sigma=sigma)
            else: G, true_labels = MRA_Graphs.MRA_CorrelatedNormal(N, L=51, K=10, a=1, b=2, choice=1, sigma=sigma)

            print("true: {}".format(true_labels))

            leiden = cd.leiden(G, weights='weight')
            leiden_labels = extract_communities_list(leiden.communities, N)
            print("leiden: {}".format(leiden_labels))
            nmi_samples["leiden"].append(normalized_mutual_info_score(true_labels, leiden_labels))

            louvain = cd.louvain(G, weight='weight')
            louvain_labels = extract_communities_list(louvain.communities, N)
            print("louvain: {}".format(louvain_labels))
            nmi_samples["louvain"].append(normalized_mutual_info_score(true_labels, louvain_labels))

            # Convert NetworkX graph to Igraph graph
            G = ig.Graph.from_networkx(G)

            leading_eigenvector = ig.Graph.community_leading_eigenvector(G, weights="weight")
            print("leading eigenvector: {}".format(leading_eigenvector.membership))
            nmi_samples["leading_eigenvector"].append(
                normalized_mutual_info_score(true_labels, leading_eigenvector.membership))

            multilevel = ig.Graph.community_multilevel(G, weights="weight")
            print("multilevel: {}".format(multilevel.membership))
            nmi_samples["multilevel"].append(normalized_mutual_info_score(true_labels, multilevel.membership))

            spinglass = ig.Graph.community_spinglass(G, weights="weight")
            print("spinglass: {}".format(spinglass.membership))
            nmi_samples["spinglass"].append(normalized_mutual_info_score(true_labels, spinglass.membership))

            walktrap = ig.Graph.community_walktrap(G, weights="weight")
            walktrap = walktrap.as_clustering()
            print("walktrap: {}".format(walktrap.membership))
            nmi_samples["walktrap"].append(normalized_mutual_info_score(true_labels, walktrap.membership))

            # Set NMI score for each algrorithm
        nmi_scores["leiden"].append(np.mean(nmi_samples["leiden"]))
        nmi_scores["louvain"].append(np.mean(nmi_samples["louvain"]))
        nmi_scores["leading_eigenvector"].append(np.mean(nmi_samples["leading_eigenvector"]))
        nmi_scores["multilevel"].append(np.mean(nmi_samples["multilevel"]))
        nmi_scores["spinglass"].append(np.mean(nmi_samples["spinglass"]))
        nmi_scores["walktrap"].append(np.mean(nmi_samples["walktrap"]))

    plt.plot(nmi_scores["leiden"], label="leiden")
    plt.plot(nmi_scores["louvain"], label="louvain")
    plt.plot(nmi_scores["leading_eigenvector"], label="leading eigenvector")
    plt.plot(nmi_scores["multilevel"], label="multilevel")
    plt.plot(nmi_scores["spinglass"], label="spinglass")
    plt.plot(nmi_scores["walktrap"], label="walktrap")
    plt.xticks(list(range(len(sigma_list))), sigma_list)
    plt.title("{0} MRA".format(MRA_type))
    plt.xlabel("Noise level (sigma)")
    plt.ylabel("NMI")
    plt.legend()
    plt.show()

sigma_list = np.array(range(11))/10
N = 100
samples_num = 10

#plot_nmi_scores(N, sigma_list, samples_num, "Rect_Trian")
plot_nmi_scores(N, sigma_list, samples_num, "Standard_Normal")
plot_nmi_scores(N, sigma_list, samples_num, "Correlated_Normal")
