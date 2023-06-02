import glob
import os
import copy
import numpy as np
import igraph as ig
import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def is_acyclic(adjacency):
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0:
            return False
    return True

def pruning_1(G, A):
    A = torch.tensor(A)
    with torch.no_grad():
        while not is_acyclic(A):
            A_nonzero = torch.nonzero(A)
            rand_int_index = np.random.randint(0, len(A_nonzero))
            A[A_nonzero[rand_int_index][0]][A_nonzero[rand_int_index][1]] = 0
        new_G = ig.Graph.Weighted_Adjacency(A.tolist())
        return new_G, A

def matrix_poly(matrix, d):
    x = torch.eye(d).double().to(matrix.device) + torch.div(matrix, d)
    return torch.matrix_power(x, d)

# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

def relaxed_softmax(logits: torch.Tensor, tau = 1, hard = False, dim = -1):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    # relaxed_logits = (logits) / tau  # ~Gumbel(logits,tau)
    # y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret