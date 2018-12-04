import sys, os
import numpy as np
import subprocess
from collections import namedtuple
import utils


def deepwalk(dataset_name, embedding_size, **kargs):
    kargs = utils.set_default(kargs, {'number-walks': 40, 'walk-length': 40, 'window-size': 10, 'workers': 8, 'format': 'edgelist'})
    edgelist_filename = os.path.join("data", dataset_name, "{}.edgelist".format(dataset_name))
    output_filename = os.path.join("embeddings", dataset_name, "{}_{}_{}_{}_{}".format(sys._getframe(0).f_code.co_name, embedding_size, kargs['number-walks'], kargs['walk-length'], kargs['window-size']))
    print(edgelist_filename, output_filename, kargs)
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    cmd = ("deepwalk --input {} --output {} --representation-size {} ".format(edgelist_filename, output_filename, embedding_size)+\
            " ".join(["--{} {}".format(key, value) for key, value in kargs.items()]))
    print(cmd)
    os.system(cmd)

def line(dataset_name, embedding_size, **kargs):
    kargs = utils.set_default(kargs, {'negative': 5, 'samples': 100, 'threads': 8, 'order': None})
    order = kargs['order']
    edgelist_filename = os.path.join("data", dataset_name, "{}.edgelist".format(dataset_name))
    output_dir = os.path.join('embeddings', dataset_name)
    if order is None:
        output_filename = os.path.join(output_dir, 'line_{}_{}_{}'.format(embedding_size, kargs['samples'], kargs['negative']))
        output_filename_1 = os.path.join(output_dir, 'line_1.embeddings')
        output_filename_norm_1 = os.path.join(output_dir, 'line_norm_1.embeddings')
        output_filename_2 = os.path.join(output_dir, 'line_2.embeddings')
        output_filename_norm_2 = os.path.join(output_dir, 'line_norm_2.embeddings')
        #if not os.path.exists(edgelist_filename):
        if not os.path.exists(os.path.dirname(output_filename)):
            os.mkdir(os.path.dirname(output_filename))
        os.system("Line -train {} -output {} -binary 1 -size {} -negative {} -threads {} -samples {} -order {}".format(edgelist_filename, output_filename_1, embedding_size/2, kargs['negative'], kargs['threads'], kargs['samples'], 1))
        os.system("Line -train {} -output {} -binary 1 -size {} -negative {} -threads {} -samples {} -order {}".format(edgelist_filename, output_filename_2, embedding_size/2, kargs['negative'], kargs['threads'], kargs['samples'], 2))
        os.system("Line_normalize -input {} -output {} -binary 1".format(output_filename_1, output_filename_norm_1))
        os.system("Line_normalize -input {} -output {} -binary 1".format(output_filename_2, output_filename_norm_2))
        os.system("Line_concatenate -input1 {} -input2 {} -output {}".format(output_filename_norm_1, output_filename_norm_2, output_filename))
        os.system("rm {}".format(output_filename_1))
        os.system("rm {}".format(output_filename_2))
        os.system("rm {}".format(output_filename_norm_1))
        os.system("rm {}".format(output_filename_norm_2))
    else:
        output_filename = os.path.join(output_dir, 'line{}_{}_{}_{}'.format(order, embedding_size, kargs['samples'], kargs['negative']))
        if not os.path.exists(os.path.dirname(output_filename)):
            os.mkdir(os.path.dirname(output_filename))
        cmd = "Line -train {} -output {} -size {} -negative {} -threads {} -samples {} -order {}".format(edgelist_filename, output_filename, embedding_size/2, kargs['negative'], kargs['threads'], kargs['samples'], order)
        print(cmd)
        os.system(cmd)

def node2vec(dataset_name, embedding_size, **kargs):
    kargs = utils.set_default(kargs, {'p': 1, 'q': 0.5, 'num-walks': 40, 'walk-length': 40, 'window-size': 10, 'workers': 8})
    edgelist_filename = os.path.join("data", dataset_name, "{}.edgelist".format(dataset_name))
    output_filename = os.path.join("embeddings", dataset_name, "{}_{}_{}_{}_{}_{}_{}".format(sys._getframe(0).f_code.co_name, embedding_size, kargs['num-walks'], kargs['walk-length'], kargs['window-size'], kargs['p'], kargs['q']))
    cmd = ("python2 src/baseline/node2vec/src/main.py --input {} --output {} --dimensions {} ".format(edgelist_filename, output_filename, embedding_size)+\
            " ".join(["--{} {}".format(key, value) for key, value in kargs.items()]))
    print(cmd)
    os.system(cmd)

def node2vec_c(dataset_name, embedding_size, **kargs):
    kargs = utils.set_default(kargs, {'p': 1, 'q': 0.5, 'num-walks': 40, 'walk-length': 40, 'window-size': 10, 'workers': 8})
    edgelist_filename = os.path.join("data", dataset_name, "{}.edgelist".format(dataset_name))
    output_filename = os.path.join("embeddings", dataset_name, "{}_{}_{}_{}_{}_{}_{}".format(sys._getframe(0).f_code.co_name, embedding_size, kargs['num-walks'], kargs['walk-length'], kargs['window-size'], kargs['p'], kargs['q']))
    cmd = 'node2vec -i:{} -o:{} -d:{} -r:{} -l:{} -k:{} -p:{} -q:{}'.format(edgelist_filename, output_filename, embedding_size, kargs['num-walks'], kargs['walk-length'], kargs['window-size'], kargs['p'], kargs['q'])
    print(cmd)
    os.system(cmd)

def GraRep(dataset_name, embedding_size, **kargs):
    kargs = utils.set_default(kargs, {'K': 4})
    edgelist_filename = os.path.join("data", dataset_name, "{}.edgelist".format(dataset_name))
    output_filename = os.path.join("embeddings", dataset_name, "{}_{}_{}".format(sys._getframe(0).f_code.co_name, embedding_size, kargs['K']))
    cmd = "matlab -nosplash -nodisplay -nodesktop -nojvm -r \"GraRep('{}','{}',{},{});exit\"".format(os.path.abspath(edgelist_filename), os.path.abspath(output_filename), kargs['K'], embedding_size//kargs['K'])
    print(cmd)
    with utils.cd('src/baseline/GraRep/code/core/'):
        os.system(cmd)

def baseline(method, dataset_name, embedding_size, **kargs):
    f = eval(method)
    f(dataset_name, embedding_size, **kargs)

if __name__ == '__main__':
    datasets = ['cora', 'citeseer', 'BlogCatalog']
    datasets = [x+'_0.8' for x in datasets]
    methods  = ['GraRep']
    emd_size = 128
    for method in methods:
        for d in datasets:
            print("###############################")
            print(d, method)
            if method == 'deepwalk':
                baseline(method, d, emd_size, **{'number-walks': 80, 'walk-length': 40, 'window-size': 10})
                baseline(method, d, emd_size, **{'number-walks': 10, 'walk-length': 40, 'window-size': 5})
            elif method == 'line':
                for i in range(4):
                    i = 10**i
                    baseline(method, d, emd_size, samples=i, order=1)
                    baseline(method, d, emd_size, samples=i, order=2)
            elif method in ['node2vec', 'node2vec_c']:
                for p in [0.5, 1, 2]:
                    for q in [0.5, 1, 2]:
                        baseline(method, d, emd_size, **{'num-walks': 10, 'walk-length': 80, 'window-size':10, 'p': p, 'q':q})
            elif method == 'GraRep':
                baseline(method, d, emd_size, **{'K': 4})
