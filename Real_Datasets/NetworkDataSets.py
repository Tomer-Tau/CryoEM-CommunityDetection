import networkx as nx
from scipy.io import mmread
import matplotlib.pyplot as plt

def draw_network_graph(set_name, i):
    set_mat = mmread("datasets/{0}/{0}.mtx".format(set_name))
    set_net = nx.from_scipy_sparse_matrix(set_mat)
    set_pos = nx.spring_layout(set_net)
    plt.figure(i)
    plt.title("{0}".format(set_name))
    nx.draw_networkx(set_net, pos=set_pos)

def draw_network_graph_gml(set_name, i):
    set_net = nx.read_gml("datasets/{0}/{0}.gml".format(set_name))
    set_pos = nx.spring_layout(set_net)
    plt.figure(i)
    plt.title("{0}".format(set_name))
    nx.draw_networkx(set_net, pos=set_pos)

datasets = ["karate", "jazz"]
for i, set_name in enumerate(datasets):
    draw_network_graph(set_name, i+1)
'''
#datasets_gml = ["polblogs"]
datasets_gml = ["netscience", "polbooks"]
for i, set_name in enumerate(datasets_gml):
    draw_network_graph_gml(set_name, i+1)
'''
plt.show()