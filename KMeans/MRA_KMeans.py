import New_Naive
import New_Naive_old_version
import MRA_Samples
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def plot_nmi_scores(N, sigma_list, runs_num, MRA_type):
    nmi_scores = []
    snr_list = []
    for sigma in sigma_list:
        nmi_samples = []
        print("sigma: {}".format(sigma))

        if MRA_type == "Rect_Trian":
            K = 2
            y, corr, true_labels = MRA_Samples.MRA_Rect_Trian(N, L=4, K=K, sigma=sigma)
        elif MRA_type == "Standard Normal":
            K = 5
            y, corr, true_labels = MRA_Samples.MRA_StandardNormal(N, L=50, K=K, sigma=sigma)

        print("true: {}".format(true_labels))

        kmeans_partitions = []

        # Calculate NMI score for a number (samples_num) of random graphs
        for i in range(runs_num):
            print("iteration number: {}".format(i))

            kmeans_partition = New_Naive.Naive_k_means(y, 4, isCircularData=True, addInitial=True, max_iteration=20, method='add_biggest', initial_state='random')
            kmeans_partitions.append(kmeans_partition[1])
            #kmeans_partition = New_Naive_old_version.Naive_k_means(y, K, max_iteration=50, method='none', initial_state='first')
            print("centroids: {}".format(kmeans_partition[0]))
            print("kmeans: {}".format(kmeans_partition[1]))
            nmi_samples.append(normalized_mutual_info_score(true_labels, kmeans_partition[1]))


        print("NMI scores throught iterartions: {}".format(nmi_samples))
        print("KMeans partitions: {}".format(kmeans_partitions))
        nmi_scores.append(np.mean(nmi_samples))
        print("NMI score: {}".format(nmi_scores))

    plt.plot(nmi_scores)
    plt.xticks(list(range(len(sigma_list))), sigma_list)
    plt.title("{0} MRA partitioned by KMeans (processed correlation matrix)".format(MRA_type))
    plt.xlabel("Noise level (sigma)")
    plt.ylabel("NMI")
    #plt.show()


sigma_list = np.array(range(1)) / 10
N = 5
runs_num = 10
plot_nmi_scores(N, sigma_list, runs_num, "Rect_Trian")