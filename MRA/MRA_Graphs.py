import numpy as np
from scipy import signal
import Generate
from scipy import stats

def MRA_Rect_Trian(N, L, K, sigma):
    x = np.zeros((K, L))
    # Generate Rectangle at x[0]
    for l in range(int(L / 4)):
        x[0][l] = 1

    x[0] = (x[0] - np.mean(x[0])) / np.linalg.norm(x[0] - np.mean(x[0]), 2)  # Normalize signal

    # Generate Triangle at x[1]
    x[1] = signal.triang(L)
    x[1] = (x[1] - np.mean(x[1])) / np.linalg.norm(x[1] - np.mean(x[1]), 2)  # Normalize signal

    y, true_partition = Generate.generate_MRA(N, K, L, sigma, x)
    max_corr = Generate.generate_maxcorr(N, L, y)

    G = Generate.generate_graph(max_corr, true_partition)

    return G, true_partition

def MRA_StandardNormal(N, L, K, sigma):
    x = np.zeros((K, L))
    # Generate Standard Normally Distributed signals
    for k in range(K):
        x[k] = np.random.standard_normal(L)
        x[k] = (x[k] - np.mean(x[k])) / np.linalg.norm(x[k] - np.mean(x[k]), 2)  # Normalize signal

    y, true_partition = Generate.generate_MRA(N, K, L, sigma, x)
    max_corr = Generate.generate_maxcorr(N, L, y)

    G = Generate.generate_graph(max_corr, true_partition)

    return G, true_partition

def MRA_RandomNormal(N, L, K, sigma):
    x = np.zeros((K, L))
    random_std = stats.randint.rvs(low=1, high=5, size=K)  # Random uniformly distributed selections of signals
    # Generate Standard Normally Distributed signals
    for k in range(K):
        x[k] = np.random.normal(0, random_std[k], L)
        x[k] = (x[k] - np.mean(x[k])) / np.linalg.norm(x[k] - np.mean(x[k]), 2)  # Normalize signal

    y, true_partition = Generate.generate_MRA(N, K, L, sigma, x)
    max_corr = Generate.generate_maxcorr(N, L, y)

    G = Generate.generate_graph(max_corr, true_partition)

    return G, true_partition
