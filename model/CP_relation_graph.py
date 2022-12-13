#!/usr/bin/env python3

# use this file to get weight mask
"""Relational graph modules"""

from parameters import *
from CPnetDetection import *
import cpnet
import pandas as pd
import networkx as nx

def compute_size(channel, group, seed=1):
    np.random.seed(seed)
    divide = channel // group
    remain = channel % group

    out = np.zeros(group, dtype=int)
    out[:remain] = divide + 1
    out[remain:] = divide
    #out = np.random.permutation(out)
    return out


def compute_densemask(in_channels, out_channels, group_num, adj):
    repeat_in = compute_size(in_channels, group_num)
    repeat_out = compute_size(out_channels, group_num)
    mask = adj
    node_mask = mask
    mask = np.repeat(mask, repeat_out, axis=0)
    mask = np.repeat(mask, repeat_in, axis=1)
    return mask, node_mask, repeat_in, repeat_out


def get_mask(in_channels, out_channels, adj):
    group_num, group_num = adj.shape
    assert group_num <= in_channels and group_num <= out_channels
    in_sizes = compute_size(in_channels, group_num)
    out_sizes = compute_size(out_channels, group_num)
    # decide low-level node num
    group_num_low = int(min(np.min(in_sizes), np.min(out_sizes)))
    # decide how to fill each node
    mask_high, node_mask, repeat_in, repeat_out = compute_densemask(in_channels, out_channels, group_num, adj)
    return mask_high, node_mask, repeat_in, repeat_out

############## Linear model

def file_to_list(file):
    rtn: object = []
    file_object: object = open(file, "r")
    rtn: object = file_object.read().splitlines()
    file_object.close()
    return list(filter(None, pd.unique(rtn).tolist())) # Remove Empty/Duplicates Values
    pass

if __name__ == '__main__':
    cp_graph_path = opt.cp_graph_path
    graphs = os.listdir(cp_graph_path)

    for graph in graphs:
        G = nx.read_gexf(cp_graph_path + '/' + graph)

        print('adj matrxi\n', nx.to_numpy_array(G))
        adjG = nx.to_numpy_array(G)
        mask_high, node_mask, repeat_in, repeat_out = get_mask(284, 284, adjG)
        print(mask_high)
        print(node_mask)
        print(repeat_in)
        print(repeat_out)