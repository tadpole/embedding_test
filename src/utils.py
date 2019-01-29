import numpy as np
import operator
import os
import networkx as nx
import scipy.io as sio

def load_wv(filename, index_from_one=False):
    with open(filename) as f:
        l = f.readline().split()
        total_num, embedding_size = int(l[0]), int(l[1])
        ls = list(map(lambda x: x.strip().split(), f.readlines()))
        tn = max([int(line[0]) for line in ls])
        assert tn == total_num or tn+1 == total_num
        res = np.zeros((tn+1, embedding_size), dtype=float)
        for line in ls:
            res[int(line[0])] = list(map(float, line[1:]))
        f.close()
    return res

def save_wv(filename, data):
    with open(filename, 'w') as f:
        nums, embedding_size = data.shape
        print(nums, embedding_size, file=f)
        for j in range(nums):
            print(j, *data[j], file=f)

def set_default(A, default_values):
    for key, value in default_values.items():
        if key not in A:
            A[key] = value
    return A

def load_mat(filename):
    return np.loadtxt(filename, dtype=float)

def load_SDNE(filename):
    return sio.loadmat(filename)['embedding']

def load_AROPE(filename):
    return sio.loadmat(filename)['U']

def load_embeddings(filename, type_=None):
    if type_ is None:
        name = os.path.basename(filename)
        ### ugly code, may fix it by checking suffix
        default_type = {'deepwalk': 'wv', 'line': 'wv', 'node2vec': 'wv',
                    'line1': 'wv', 'line2': 'wv', 'SDNE': 'SDNE', 'AROPE': 'AROPE'}
        type_ = default_type.get(name.split('_')[0], 'mat')
    res = eval("load_{}".format(type_))(filename)
    return res

def load_graph(filename, type_='edgelist'):
    if type_ == 'edgelist':
        G = nx.read_edgelist(filename, nodetype=int)
    return G

def get_graph_info(G):
    print(G.number_of_nodes(), G.number_of_edges())

def dot_product(X, Y):
    return np.sum(X*Y, axis=1)

def batch_dot_product(emb, edges, batch_size=None):
    if batch_size is None:
        return dot_product(emb[edges[:, 0]], emb[edges[:, 1]])
    n = edges.shape[0] // batch_size
    res = []
    for i in range(n):
        r = dot_product(emb[edges[i*batch_size:(i+1)*batch_size, 0]], emb[edges[i*batch_size:(i+1)*batch_size, 1]])
        res.append(r)
    a = edges.shape[0]-n*batch_size
    if a > 0:
        res.append(dot_product(emb[edges[n*batch_size:, 0]], emb[edges[n*batch_size:, 1]]))
    return np.hstack(res)

def euclidean_distance(X, Y):
    return np.linalg.norm(X-Y, axis=1)

def split_dataset(dataset_name, radio=0.8):
    filename = os.path.join('data', dataset_name, '{}.edgelist'.format(dataset_name))
    graph = load_graph(filename)
    graph_train = nx.Graph()
    graph_test = nx.Graph()
    edges = np.random.permutation(list(graph.edges()))
    nodes = set()
    for a, b in edges:
        if a not in nodes or b not in nodes:
            graph_train.add_edge(a, b)
            nodes.add(a)
            nodes.add(b)
        else:
            graph_test.add_edge(a, b)
    assert len(nodes) == graph.number_of_nodes()
    assert len(nodes) == graph_train.number_of_nodes()
    num_test_edges = int((1-radio)*graph.number_of_edges())
    now_number = graph_test.number_of_edges()
    if num_test_edges < now_number:
        test_edges = list(graph_test.edges())
        graph_train.add_edges_from(test_edges[:now_number-num_test_edges])
        graph_test.remove_edges_from(test_edges[:now_number-num_test_edges])

    get_graph_info(graph)
    get_graph_info(graph_train)
    get_graph_info(graph_test)

    data_path = os.path.join('data', '{}_{}'.format(dataset_name, radio))

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    nx.write_edgelist(graph_train, os.path.join(data_path, '{}_{}.edgelist'.format(dataset_name, radio)), data=False)
    nx.write_edgelist(graph_test, os.path.join(data_path, '{}_{}_test.edgelist'.format(dataset_name, radio)), data=False)

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


if __name__ == '__main__':
    #split_dataset('cora')
    print(load_wv('/home/tuke/mle/embeddings/BlogCatalog_0.8/sampled/s0/deepwalk_64_2_2_2'))
