import numpy as np
import os
import collections
import random
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
from functools import reduce

from . import utils

def sampling_edges(G, sampling, G_test=None):
    recon_flag = (G_test is None)
    print(recon_flag)
    N = G.number_of_nodes()
    edges = []
    labels = []
    if sampling is None:
        for i in range(N):
            for j in range(i+1, N):
                if not recon_flag and G.has_edge(i, j):
                    continue
                edges.append([i, j])
                if recon_flag:
                    labels.append(1 if G.has_edge(i, j) else 0)
                else:
                    labels.append(1 if G_test.has_edge(i, j) else 0)
    else:
        edge_set = set()
        edges = []
        n = 0
        while True:
            i = np.random.randint(0, N-1)
            j = np.random.randint(i+1, N)
            if (i, j) not in edge_set:
                if not recon_flag and G.has_edge(i, j):
                    continue
                edge_set.add((i, j))
                edges.append([i, j])
                if recon_flag:
                    labels.append(1 if G.has_edge(i, j) else 0)
                else:
                    labels.append(1 if G_test.has_edge(i, j) else 0)
                n += 1
                if n % 100000:
                    print("sampling: {}/{}".format(n, sampling))
                if n >= sampling:
                    break
        pos = sum(labels)
        neg = len(labels)-pos
        print("sampling: pos: {}, neg: {}".format(pos, neg))
    edges = np.array(edges)
    labels = np.array(labels)
    return edges, labels

def make_train_test(label, radio):
    l = collections.defaultdict(list)
    for i, j in label:
        l[i].append(j)
    multi_label = max([len(x) for x in l.values()]) > 1
    mlb = MultiLabelBinarizer(sparse_output=True)
    y = mlb.fit_transform(list(l.values()))
    n = int(radio*len(l))
    x = np.array(list(l.keys()))
    ind = np.random.permutation(range(len(l)))
    print(set(reduce(lambda x, y: x+y, [l[i] for i in x[ind[:n]]])))
    print(set(reduce(lambda x, y: x+y, [l[i] for i in x[ind[n:]]])))
    return x[ind[:n]], y[ind[:n]], x[ind[n:]], y[ind[n:]]

def save_result(save_filename, embedding_filenames, res):
    if not os.path.exists(os.path.dirname(save_filename)):
        os.makedirs(os.path.dirname(save_filename))
    with open(save_filename, 'w') as f:
        f_name = open("{}_names".format(save_filename), 'w')
        print("################ save results in ", save_filename)
        for i, n in enumerate(embedding_filenames):
            print(os.path.basename(n), file=f_name)
            if type(res[i]) == np.ndarray or type(res[i]) == list:
                print("\t".join(list(map(lambda x: "{:.4f}".format(x), res[i]))), file=f)
            else:
                print("{:.4f}".format(res[i]), file=f)
        f_name.close()

def reconstruction(edges, labels, embedding_filenames, evalution, sampling, args):
    res = []
    for fn in embedding_filenames:
        emb = utils.load_embeddings(fn)
        if sampling is None:
            matrix_sim = emb.dot(emb.T)
            sim = matrix_sim[edges[:, 0], edges[:, 1]]
        else:
            #sim = utils.dot_product(emb[edges[:, 0]], emb[edges[:, 1]])
            sim = utils.batch_dot_product(emb, edges, batch_size=1000000)
        #sim = -utils.euclidean_distance(emb[edges[:, 0]], emb[edges[:, 1]])
        ind = np.argsort(sim)[::-1]
        labels_ordered = labels[ind]
        if evalution == 'precision_k':
            max_n = 1000000 if 'max_n' not in args else args['max_n']
            positive = np.cumsum(labels_ordered[:max_n])
            x = np.arange(max_n)
            pk = positive*1.0/(x+1)
            res.append(pk)
            print(os.path.basename(fn), '\t', pk[[10**i for i in range(5)]])
        elif evalution == 'AUC':
            rank = len(labels)-np.where(labels_ordered == 1)[0]
            M = len(rank)
            N = len(labels)-M
            auc = (np.sum(rank)-M*(M+1)/2)*1.0/M/N
            res.append(auc)
            print(os.path.basename(fn), '\t', res[-1])
    return res

link_predict = reconstruction

def classification(train_id, train_label, test_id, test_label, embedding_filenames, args):
    res = []
    for fn in embedding_filenames:
        emb = utils.load_embeddings(fn)
        cf = OneVsRestClassifier(svm.SVC(probability=True))
        cf.fit(emb[train_id], train_label)
        y_prob = cf.predict_proba(emb[test_id])
        top_k_list = [r.nnz for r in test_label]
        data, row, col = [], [], []
        for idx, k in enumerate(top_k_list):
            prob = y_prob[idx]
            classes = prob.argsort()[-k:]
            for c in classes:
                data.append(1)
                row.append(idx)
                col.append(c)
        y_pred = csr_matrix((data, (row, col)), shape=test_label.shape)
        averages = ["micro", "macro"]
        r = [f1_score(test_label, y_pred, average=a) for a in averages]
        print(os.path.basename(fn), '\t', r)
        res.append(r)
    return np.array(res)

def node_recommendation(G_train, G_test, embedding_filenames, evalution, args):
    assert evalution == 'MAP'
    if args['degree_radio'] is not None:
        d = [i[1] for i in G_train.degree()]
        degree_max = np.sort(d)[int(args['degree_radio']*len(d))]
        print('degree_max: ', degree_max)
    else:
        degree_max = G_train.number_of_nodes()
    res = []
    for fn in embedding_filenames:
        emb = utils.load_embeddings(fn)
        ap = []
        for i in range(G_train.number_of_nodes()):
            if i not in G_test or G_train.degree(i) > degree_max:
                continue
            l = list(set(range(G_train.number_of_nodes()))-set(G_train.neighbors(i)))
            l_test = set(G_test.neighbors(i))
            grouth = np.array([int(j in l_test) for j in l])
            s = emb[[i]].dot(emb[l].T)[0]
            ind = s.argsort()[::-1]
            g = grouth[ind][:args['max_length']]
            sg = np.cumsum(g)
            p = sg/(np.arange(len(g))+1.0)
            sp = np.cumsum(p*g)
            ap.append(sp/(sg+1e-6))
        r = np.mean(ap, 0)
        print(os.path.basename(fn), '\t', r[range(0, len(r), 10)])
        res.append(r)
    return res

def test(task, evalution, dataset, embedding_filenames, save_filename=None, **args):
    args = utils.set_default(args, {'seed': 0})
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    if task == 'reconstruction':
        args = utils.set_default(args, {'sampling_mapping': None})
        dataset_name = os.path.join('data', dataset, "{}.edgelist".format(dataset))
        G = utils.load_graph(dataset_name)

        utils.get_graph_info(G)

        sampling = None if dataset_name not in args['sampling_mapping'] else args['sampling_mapping'][dataset_name]
        edges, labels = sampling_edges(G, sampling)
        res = reconstruction(edges, labels, embedding_filenames, evalution, sampling, args)
    elif task == 'link_predict':
        args = utils.set_default(args, {'sampling_mapping': None})
        dataset_train_name = os.path.join('data', dataset, "{}.edgelist".format(dataset))
        dataset_test_name = os.path.join('data', dataset, "{}_test.edgelist".format(dataset))
        G_train = utils.load_graph(dataset_train_name)
        G_test = utils.load_graph(dataset_test_name)

        sampling = None if dataset_name not in args['sampling_mapping'] else args['sampling_mapping'][dataset_name]
        edges, labels = sampling_edges(G_train, sampling, G_test=G_test)
        res = link_predict(edges, labels, embedding_filenames, evalution, sampling, args)
    elif task == 'classification':
        args = utils.set_default(args, {'radio': 0.8})
        if 'label_name' not in args:
            label_name = os.path.join('data', dataset, '{}_label.txt'.format(dataset))
        else:
            label_name = args['label_name']
        label = np.loadtxt(label_name, dtype=int)
        if type(args['radio']) == np.ndarray or type(args['radio']) == list:
            res = np.zeros((len(embedding_filenames), 2*len(args['radio'])))
            for i, radio in enumerate(args['radio']):
                train_id, train_label, test_id, test_label = make_train_test(label, radio)
                print(train_id)
                r = classification(train_id, train_label, test_id, test_label, embedding_filenames, args)
                res[:, i] = r[:, 0]
                res[:, i+len(args['radio'])] = r[:, 1]
        else:
            train_id, train_label, test_id, test_label = make_train_test(label, args['radio'])
            res = classification(train_id, train_label, test_id, test_label, embedding_filenames, args)
    elif task == 'node_recommendation':
        args = utils.set_default(args, {'max_length': 100})
        dataset_train_name = os.path.join('data', dataset, "{}.edgelist".format(dataset))
        dataset_test_name = os.path.join('data', dataset, "{}_test.edgelist".format(dataset))
        G_train = utils.load_graph(dataset_train_name)
        G_test = utils.load_graph(dataset_test_name)

        res = node_recommendation(G_train, G_test, embedding_filenames, evalution, args)
    if save_filename is not None:
        save_result(save_filename, embedding_filenames, res)

if __name__ == '__main__':
    task = 'node_recommendation'  ### reconstruction | link_predict | classification | node_recommendation
    datasets = ['cora', 'citeseer', 'BlogCatalog']
    datasets = [x+('_0.8' if task in ['link_predict', 'node_recommendation'] else '') for x in datasets]
    evalution = 'MAP'

    args = {
            'sampling_mapping' : {'Flickr': 10000000}, # for reconstruction or link_predict
            'radio' : [0.1*x for x in range(1, 10)],   # for classification
            'degree_radio' : 0.2,                      # for node_recommendation
    }
    print(task, '\t', evalution)
    for dataset_name in datasets:
        print('############################')
        print(dataset_name)
        if evalution == 'AUC':
            save_filename = 'result/{}/auc.txt'.format(dataset_name)
        elif evalution == 'precision_k':
            save_filename = 'result/{}/pk.txt'.format(dataset_name)
        if task == 'classification':
            save_filename = 'result/{}/cf'.format(dataset_name)
        elif task == 'node_recommendation':
            save_filename = 'result/{}/nr_{}'.format(dataset_name, args['degree_radio'])
        baseline = ['deepwalk_128_10_40_5', 'deepwalk_128_80_40_10']+\
                ['line1_128_{}_5'.format(10**i) for i in range(4)]+\
                ['line2_128_{}_5'.format(10**i) for i in range(4)]+\
                ['node2vec_c_128_10_80_10_{}_{}'.format(p, q) for p in [0.5, 1 ,2] for q in [0.5, 1, 2]]+\
                ['GraRep_128_4']
        baselines = [os.path.join('embeddings', dataset_name, i) for i in baseline]
        baselines += ['src/baseline/SDNE/result/{}_128/SDNE_embedding.mat'.format(dataset_name)]
        test(task, evalution, dataset_name, baselines, save_filename=save_filename, **args)
