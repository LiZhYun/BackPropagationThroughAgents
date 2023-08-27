import glob
import os
import numpy as np
import math
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict
import copy
from typing import NoReturn, Optional, List
string_classes=str
import re
from collections.abc import Sequence, Mapping
import collections.abc as container_abcs


int_classes = int
np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)

def flatten(l):
    return np.stack([item for sublist in l for item in sublist])

def default_collate_with_dim(batch, device='cpu',dim=0, k=None,cat=False):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        #if k is not None:
        #    print(k)

        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            try:
                if cat == True:
                    return torch.cat(batch, dim=dim, out=out).to(device=device)
                else:
                    return torch.stack(batch, dim=dim, out=out).to(device=device)
            except:
                print(batch)
                if k is not None:
                    print(k)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return default_collate_with_dim([torch.as_tensor(b,device=device) for b in batch],device=device,dim=dim,cat=cat)
            elif elem.shape == ():  # scalars
                try:
                    return torch.as_tensor(batch,device=device)
                except:
                    print(batch)
                    if k is not None:
                        print(k)
        elif isinstance(elem, float):
            try:
                return torch.tensor(batch,device=device)
            except:
                print(batch)
                if k is not None:
                    print(k)
        elif isinstance(elem, int_classes):
            try:
                return torch.tensor(batch,device=device)
            except:
                print(batch)
                if k is not None:
                    print(k)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: default_collate_with_dim([d[key] for d in batch if key in d.keys()],device=device,dim=dim, k=key, cat=cat) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(default_collate_with_dim(samples,device=device,dim=dim,cat=cat) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [default_collate_with_dim(samples,device=device,dim=dim,cat=cat) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))

def deep_merge_dicts(original: dict, new_dict: dict) -> dict:
    """
    Overview:
        merge two dict using deep_update
    Arguments:
        - original (:obj:`dict`): Dict 1.
        - new_dict (:obj:`dict`): Dict 2.
    Returns:
        - (:obj:`dict`): A new dict that is d1 and d2 deeply merged.
    """
    original = original or {}
    new_dict = new_dict or {}
    merged = copy.deepcopy(original)
    if new_dict:  # if new_dict is neither empty dict nor None
        deep_update(merged, new_dict, True, [])

    return merged

def deep_update(
    original: dict,
    new_dict: dict,
    new_keys_allowed: bool = False,
    whitelist: Optional[List[str]] = None,
    override_all_if_type_changes: Optional[List[str]] = None
):
    """
    Overview:
        Updates original dict with values from new_dict recursively.

    .. note::

        If new key is introduced in new_dict, then if new_keys_allowed is not
        True, an error will be thrown. Further, for sub-dicts, if the key is
        in the whitelist, then new subkeys can be introduced.

    Arguments:
        - original (:obj:`dict`): Dictionary with default values.
        - new_dict (:obj:`dict`): Dictionary with values to be updated
        - new_keys_allowed (:obj:`bool`): Whether new keys are allowed.
        - whitelist (Optional[List[str]]): List of keys that correspond to dict
            values where new subkeys can be introduced. This is only at the top
            level.
        - override_all_if_type_changes(Optional[List[str]]): List of top level
            keys with value=dict, for which we always simply override the
            entire value (:obj:`dict`), if the "type" key in that value dict changes.
    """
    whitelist = whitelist or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise RuntimeError("Unknown config parameter `{}`. Base config have: {}.".format(k, original.keys()))

        # Both original value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and \
                    "type" in value and "type" in original[k] and \
                    value["type"] != original[k]["type"]:
                original[k] = value
            # Whitelisted key -> ok to add new subkeys.
            elif k in whitelist:
                deep_update(original[k], value, True)
            # Non-whitelisted key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original

def read_config(path: str) -> EasyDict:
    """
    Overview:
        read configuration from path
    Arguments:
        - path (:obj:`str`): Path of source yaml
    Returns:
        - (:obj:`EasyDict`): Config data from this file with dict type
    """
    if path:
        assert os.path.exists(path), path
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    return EasyDict(config)

def is_acyclic(A):
    adjacency = A.clone()
    prod = torch.eye(adjacency.shape[0]).to(adjacency.device)
    for _ in range(1, adjacency.shape[0] + 1):
        prod = torch.matmul(adjacency, prod)
        if torch.trace(prod) != 0:
            return False
    return True

def pruning(adjacency):
    A = adjacency.clone()
    while not is_acyclic(A):
        A_nonzero = torch.nonzero(A).to(adjacency.device)
        rand_int_index = np.random.randint(0, len(A_nonzero))
        A[A_nonzero[rand_int_index][0]][A_nonzero[rand_int_index][1]] = 0
    return A

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def matrix_poly(matrix, d):
    x = torch.eye(d).double().to(matrix.device) + torch.div(matrix, d)
    return torch.matrix_power(x, d)

# compute constraint h(A) value
def cal_acyclic_loss(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    elif obs_space.__class__.__name__ == 'Dict':
        obs_shape = obs_space.spaces
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

def generate_mask_from_order(agent_order, ego_exclusive):
    """Generate execution mask from agent order.

    Used during autoregressive training.

    Args:
        agent_order (torch.Tensor): Agent order of shape [*, n_agents].

    Returns:
        torch.Tensor: Execution mask of shape [*, n_agents, n_agents - 1].
    """
    shape = agent_order.shape
    n_agents = shape[-1]
    agent_order = agent_order.view(-1, n_agents)
    bs = agent_order.shape[0]

    cur_execution_mask = torch.zeros(bs, n_agents).to(agent_order)
    all_execution_mask = torch.zeros(bs, n_agents, n_agents).to(agent_order)

    batch_indices = torch.arange(bs)
    for i in range(n_agents):
        agent_indices = agent_order[:, i].long()

        cur_execution_mask[batch_indices, agent_indices] = 1
        all_execution_mask[batch_indices, :,
                           agent_indices] = 1 - cur_execution_mask
        all_execution_mask[batch_indices, agent_indices, agent_indices] = 1
    
    if not ego_exclusive:
        # [*, n_agent, n_agents]
        all_execution_mask = all_execution_mask.view(*shape[:-1], n_agents, n_agents) * (1-torch.eye(n_agents).to(agent_order)).unsqueeze(0).repeat(bs, 1, 1)
        # for batch_idx in batch_indices:
        #     for agent_idx in range(n_agents):
        #         tmp_mask = torch.zeros(n_agents).to(agent_order)
        #         indices = torch.nonzero(all_execution_mask[batch_idx, agent_idx] == 1)
        #         if indices.numel() > 0:
        #             last_index = indices[-1]
        #             tmp_mask[last_index] = 1
        #             all_execution_mask[batch_idx, agent_idx] = tmp_mask
        return all_execution_mask
    else:
        # [*, n_agents, n_agents - 1]
        execution_mask = torch.zeros(bs, n_agents,
                                     n_agents - 1).to(agent_order)
        for i in range(n_agents):
            execution_mask[:, i] = torch.cat([
                all_execution_mask[..., i, :i], all_execution_mask[..., i,
                                                                   i + 1:]
            ], -1)
        return execution_mask.view(*shape[:-1], n_agents, n_agents - 1)

def args_str2bool(flag: str):
    assert flag == "True" or flag == "False"
    if flag == "True":
        return True
    else:
        return False
    
def get_weights(parameters):
    """
    Function used to get the value of a set of torch parameters as
    a single vector of values.

    Args:
        parameters (list): list of parameters to be considered.

    Returns:
        A numpy vector consisting of all the values of the vectors.

    """
    weights = list()

    for p in parameters:
        w = p.data.detach().cpu().numpy()
        weights.append(w.flatten())

    weights = np.concatenate(weights, 0)

    return weights

def set_weights(parameters, weights, device):
    """
    Function used to set the value of a set of torch parameters given a
    vector of values.

    Args:
        parameters (list): list of parameters to be considered;
        weights (numpy.ndarray): array of the new values for
            the parameters;
        use_cuda (bool): whether the parameters are cuda tensors or not;

    """
    idx = 0
    for p in parameters:
        shape = p.data.shape

        c = 1
        for s in shape:
            c *= s

        if type(weights) == np.ndarray:
            w = np.reshape(weights[idx:idx + c], shape)
        else:
            w = np.array(weights)

        # if not use_cuda:
        #     w_tensor = torch.from_numpy(w).type(p.data.dtype)
        # else:
        w_tensor = torch.from_numpy(w).type(p.data.dtype).to(device)

        p.data = w_tensor
        idx += c

    assert idx == weights.size