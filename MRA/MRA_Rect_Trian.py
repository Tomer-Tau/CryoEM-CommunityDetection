import numpy as np
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt


# Parameters
N = 100 # Number of observations
L = 50 # Signals length
K = 2 # Number of signals
sigma = 0.1 # Noise level

x = np.zeros((K,L))
# Generate Rectangle at x[0]
for i in range(int(L/2)):
    x[0][i] = 1
# Generate Triangle at x[1]
x[1] = signal.triang(L)

k = stats.randint.rvs(low=0, high=K, size=N) # Random uniformly distributed selections of signals
s = stats.randint.rvs(low=0, high=L, size=N) # Random uniformly distributed selections of shifts

# Generate Noise array
epsilon = np.zeros((N,L))
for i in range(N):
    epsilon[i] = sigma*np.random.randn(L)

# Generate MRA samples
y = np.zeros((N,L))
for i in range(N):
    shifted_x = np.roll(x[k[i]], s[i])
    y[i] = shifted_x + epsilon[i]

max_corr = np.zeros((N,2)) # Array of 2-d lists, where first element is the maximal correlation,
                           # second element is the sample y[j] that has this correlation with y[i], where i is the index
# Calculate max correlation for each sample
for i in range(N-1):
    normalized_corr = []
    for j in range(i+1,N):
        normalized_corr.append(np.correlate(y[i] - np.mean(y[i]), y[j] - np.mean(y[j]))[0]/
                               L*(np.std(y[i]) * np.std(y[j]))) # Calculate the normalized correlation between y[i] and y[j]
    max_corr[i][0] = max(normalized_corr)
    max_corr[i][1] = normalized_corr.index(max_corr[i][0])

for i in range(len(max_corr)):
    print('{0},{1}'.format(max_corr[i][0], max_corr[i][1]))