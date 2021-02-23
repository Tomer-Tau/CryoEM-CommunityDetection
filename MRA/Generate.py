import networkx as nx
import numpy as np
from scipy import stats
from numpy.fft import fft, ifft


def generate_graph(adjacency_matrix, node_labels):
    G = nx.Graph()
    # Add nodes
    for i in range(len(node_labels)):
        G.add_node(i, label=node_labels[i])
    # Add edges

    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix[i])):
            if not np.isnan(adjacency_matrix[i][j]):
                G.add_edge(i, j, weight=adjacency_matrix[i][j])
    return G


def generate_MRA(N, K, L, sigma, x):
    k = stats.randint.rvs(low=0, high=K, size=N)  # Random uniformly distributed selections of signals
    s = stats.randint.rvs(low=0, high=L, size=N)  # Random uniformly distributed selections of shifts

    # Generate Noise array
    epsilon = np.zeros((N, L))
    for n in range(N):
        epsilon[n] = sigma * np.random.randn(L)

    # Generate MRA samples
    y = np.zeros((N, L))
    true_signals = np.zeros(N)  # List the holds the signal from which y[i] sample was generated, where i is the index
    for n in range(N):
        true_signals[n] = k[n]
        shifted_x = np.roll(x[k[n]], s[n])
        y[n] = shifted_x + epsilon[n]

    return y, true_signals


def generate_maxcorr(N, L, y):
    max_corr = np.zeros((N, N))  # Matrix of maximal correlations between samples y[i] and y[j]
    # where i is the row number and j is the column number.
    # Note: max_corr[i][i] = None, because the correlation of the sample with itself is irrelevant.
    # Note: For negative or null correlations, the value in the matrix is None

    # Calculate max correlations for each sample
    for i in range(N - 1):
        max_corr[i][i] = None
        for j in range(i + 1, N):
            circular_corr = max(ifft(fft(y[i]).conj() * fft(y[j])).real)
            if circular_corr > 0:
                max_corr[i][j] = circular_corr
                max_corr[j][i] = circular_corr
            else:
                max_corr[i][j] = None
                max_corr[j][i] = None
    max_corr[N - 1][N - 1] = None
    return max_corr
