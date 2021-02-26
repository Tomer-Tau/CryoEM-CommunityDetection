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
samples_num = 10

################## Rectangle&Triangle MRA ##################
nmi_scores = {"greedy": [], "leiden": [], "louvain": [], "cpm": [], "lpa": [], "infomap": [],
              "leading_eigenvector": [], "multilevel": [], "edge_betweenness": [], "spinglass": [], "walktrap": []}

for sigma in sigma_list:
    nmi_samples = {"greedy": [], "leiden": [], "louvain": [], "cpm": [], "lpa": [], "infomap": [],
              "leading_eigenvector": [], "multilevel": [], "edge_betweenness": [], "spinglass": [], "walktrap": []}

    print("sigma: {}".format(sigma))

    # Calculate NMI score for a number (samples_num) of random graphs
    for i in range(samples_num):
        G, true_labels = MRA_Graphs.MRA_Rect_Trian(N, L=50, K=2, sigma=sigma)
        print("true: {}".format(true_labels))

        # Partition Graph by various methods
        #greedy = cd.greedy_modularity(G, weight='weight')
        #greedy_labels = extract_communities_list(greedy.communities, N)
        #print("greedy: {}".format(greedy_labels))
        #nmi_samples["greedy"].append(normalized_mutual_info_score(true_labels, greedy_labels))

        leiden = cd.leiden(G, weights='weight')
        leiden_labels = extract_communities_list(leiden.communities, N)
        print("leiden: {}".format(leiden_labels))
        nmi_samples["leiden"].append(normalized_mutual_info_score(true_labels, leiden_labels))

        louvain = cd.louvain(G, weight='weight')
        louvain_labels = extract_communities_list(louvain.communities, N)
        print("louvain: {}".format(louvain_labels))
        nmi_samples["louvain"].append(normalized_mutual_info_score(true_labels, louvain_labels))

        #cpm = cd.cpm(G, weights='weight')
        #cpm_labels = extract_communities_list(cpm.communities, N)
        #print("cpm: {}".format(cpm_labels))
        #nmi_samples["cpm"].append(normalized_mutual_info_score(true_labels, cpm_labels))

        #lpa = nx_cd.asyn_lpa_communities(G, weight='weight')
        #lpa_labels = extract_communities_list(list(lpa), N)
        #print("lpa: {}".format(lpa_labels))
        #nmi_samples["lpa"].append(normalized_mutual_info_score(true_labels, lpa_labels))

        #Convert NetworkX graph to Igraph graph
        G = ig.Graph.from_networkx(G)

        #infomap = ig.Graph.community_infomap(G)
        #print("infomap: {}".format(infomap.membership))
        #nmi_samples["infomap"].append(normalized_mutual_info_score(true_labels, infomap.membership))

        leading_eigenvector = ig.Graph.community_leading_eigenvector(G, weights="weight")
        print("leading eigenvector: {}".format(leading_eigenvector.membership))
        nmi_samples["leading_eigenvector"].append(normalized_mutual_info_score(true_labels, leading_eigenvector.membership))

        multilevel = ig.Graph.community_multilevel(G, weights="weight")
        print("multilevel: {}".format(multilevel.membership))
        nmi_samples["multilevel"].append(normalized_mutual_info_score(true_labels, multilevel.membership))

        #edge_betweenness = ig.Graph.community_edge_betweenness(G, weights="weight")
        #edge_betweenness = edge_betweenness.as_clustering()
        #print("edge betweenness: {}".format(edge_betweenness.membership))
        #nmi_samples["edge_betweenness"].append(normalized_mutual_info_score(true_labels, edge_betweenness.membership))

        spinglass = ig.Graph.community_spinglass(G, weights="weight")
        print("spinglass: {}".format(spinglass.membership))
        nmi_samples["spinglass"].append(normalized_mutual_info_score(true_labels, spinglass.membership))

        walktrap = ig.Graph.community_walktrap(G, weights="weight")
        walktrap = walktrap.as_clustering()
        print("walktrap: {}".format(walktrap.membership))
        nmi_samples["walktrap"].append(normalized_mutual_info_score(true_labels, walktrap.membership))

    # Set NMI score for each algrorithm
    #nmi_scores["greedy"].append(np.mean(nmi_samples["greedy"]))
    nmi_scores["leiden"].append(np.mean(nmi_samples["leiden"]))
    nmi_scores["louvain"].append(np.mean(nmi_samples["louvain"]))
    #nmi_scores["cpm"].append(np.mean(nmi_samples["cpm"]))
    #nmi_scores["lpa"].append(np.mean(nmi_samples["lpa"]))
    #nmi_scores["infomap"].append(np.mean(nmi_samples["infomap"]))
    nmi_scores["leading_eigenvector"].append(np.mean(nmi_samples["leading_eigenvector"]))
    nmi_scores["multilevel"].append(np.mean(nmi_samples["multilevel"]))
    #nmi_scores["edge_betweenness"].append(np.mean(nmi_samples["edge_betweenness"]))
    nmi_scores["spinglass"].append(np.mean(nmi_samples["spinglass"]))
    nmi_scores["walktrap"].append(np.mean(nmi_samples["walktrap"]))

#plt.plot(nmi_scores["greedy"],label="greedy")
plt.plot(nmi_scores["leiden"],label="leiden")
plt.plot(nmi_scores["louvain"],label="louvain")
#plt.plot(nmi_scores["cpm"],label="CPM")
#plt.plot(nmi_scores["lpa"],label="LPA")
#plt.plot(nmi_scores["infomap"],label="infomap")
plt.plot(nmi_scores["leading_eigenvector"],label="leading eigenvector")
plt.plot(nmi_scores["multilevel"],label="multilevel")
#plt.plot(nmi_scores["edge_betweenness"],label="edge betweenness")
plt.plot(nmi_scores["spinglass"],label="spinglass")
plt.plot(nmi_scores["walktrap"],label="walktrap")
plt.xticks(list(range(len(sigma_list))), sigma_list)
plt.title("Triangle&Rectangle MRA")
plt.xlabel("Noise level (sigma)")
plt.ylabel("NMI")
plt.legend()
plt.show()
'''
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
'''
'''
################## Random Normal MRA ##################
greedy_nmi = np.zeros((10, 10))
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
    for i in range(10):
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
'''
