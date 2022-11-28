from scipy import sparse
import scipy.sparse.linalg as slinalg
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, id):
        self.id = id
        self.neighbs = [] # List of (weight, node)
        self.touched = False
        self.visited = False

def get_laplacian(nodes):
    """
    Compute the unweighted graph laplacian of a set of nodes

    Parameters
    ----------
    nodes: list of N Node
        Nodes of the graph
    
    Returns
    -------
    L: sparse array (N, N)
        Laplacian
    """
    N = len(nodes)
    I = []
    J = []
    V = []
    for i, node in enumerate(nodes):
        M = len(node.neighbs)
        I.append(i)
        J.append(i)
        V.append(M)
        I += [i]*M
        J += [n[1].id for n in node.neighbs]
        V += [-1]*M
    V = np.array(V, dtype=float)
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N))
    return L.tocsc()

def plot_graph_dists(nodes, dists):
    """
    Plot a graph with a dists, using the Laplacian to lay it out
    """
    L = get_laplacian(nodes)
    w, v = slinalg.eigsh(L, k=L.shape[1]-1, which='SM')
    # Find the first two non-zero eigenvalues
    k = 0
    eps = 1e-10
    while w[k] < eps and k < L.shape[1]-2:
        k += 1
    x = w[k]*v[:, k]
    y = w[k+1]*v[:, k+1]
    plt.scatter(x, y)
    for n1 in nodes:
        plt.text(x[n1.id], y[n1.id], "{} ({:.2f})".format(n1.id, dists[n1.id]), zorder=10)
        for (w, n2) in n1.neighbs:
            xs = [x[n1.id], x[n2.id]]
            ys = [y[n1.id], y[n2.id]]
            plt.plot(xs, ys)
            xs = np.mean(xs)
            ys = np.mean(ys)
            plt.text(xs, ys, "{:.2f}".format(w))
    plt.show()

def make_neighbors(a, b, w):
    """
    Make two nodes neighbors with a weight

    Parameters
    ----------
    a: Node
        First node
    b: Node
        Second node
    w: float
        Weight between nodes
    """
    a.neighbs.append((w, b))
    b.neighbs.append((w, a))

def get_sample_graph():
    nodes = []
    for i in range(6):
        nodes.append(Node(len(nodes)))
    [a, b, c, d, e, f] = nodes
    make_neighbors(a, b, 2)
    make_neighbors(a, c, 10)
    make_neighbors(b, d, 3)
    make_neighbors(c, d, 2)
    make_neighbors(a, e, 6)
    make_neighbors(e, c, 3)
    return nodes

def shortest_path(node):
    """
    Compute the shortest path from this node to all other nodes
    """
    N = len(nodes)
    dists = [np.inf]*N
    dists[node.id] = 0

    return dists

nodes = get_sample_graph()
dists = shortest_path(nodes[0])
plot_graph_dists(nodes, dists)