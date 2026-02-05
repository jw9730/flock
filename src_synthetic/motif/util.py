import os
import ast
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch import distributed as dist
from torch_geometric.utils.sparse import index2ptr
from torch_geometric.utils import index_sort

import numpy as np


logger = logging.getLogger(__file__)


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    tree = env.parse(raw)
    vars = meta.find_undeclared_variables(tree)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def literal_eval(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    parser.add_argument("-p", "--project", help="neptune project name",  type=str, default=None)
    
    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def get_root_logger(file=True):
    format = "%(asctime)-10s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=format, datefmt=datefmt)
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    if file:
        handler = logging.FileHandler("log.txt")
        format = logging.Formatter(format, datefmt)
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def synchronize():
    if get_world_size() > 1:
        dist.barrier()


def get_device(cfg):
    if cfg.train.gpus:
        device = torch.device(cfg.train.gpus[get_rank()])
    else:
        device = torch.device("cpu")
    return device


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = get_world_size()
    if cfg.train.gpus is not None and len(cfg.train.gpus) != world_size:
        error_msg = "World size is %d but found %d GPUs in the argument"
        if world_size == 1:
            error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
        raise ValueError(error_msg % (world_size, len(cfg.train.gpus)))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")

    if cfg.dataset.get("version") is None:
        working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.model["class"], cfg.dataset["class"], time.strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.model["class"], cfg.dataset["class"], str(cfg.dataset["version"]), time.strftime("%Y-%m-%d-%H-%M-%S"))
    
    # synchronize working directory
    if get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    synchronize()
    if get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    synchronize()
    if get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir



def static_positional_encoding(max_arity, input_dim):
    """
    Generate a static positional encoding.

    Args:
    - max_arity (int): Maximum arity for which to create positional encodings.
    - input_dim (int): Dimension of the input feature vector.

    Returns:
    - torch.Tensor: A tensor containing positional encodings for each position.
    """
    # Initialize the positional encoding matrix
    position = torch.zeros(max_arity + 1, input_dim)

    # Compute the positional encodings
    for pos in range(max_arity + 1):
        # position[pos, pos] = 1
        for i in range(0, input_dim, 2):
            position[pos, i] = np.sin(pos / (10000 ** ((2 * i) / input_dim)))
            if i + 1 < input_dim:
                position[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / input_dim)))


    return position

def coo_to_csr(row, col, edge_types, num_nodes=None):
    if num_nodes is None:
        num_nodes = int(row.max()) + 1

    row, perm = index_sort(row, max_value=num_nodes)
    col = col[perm]
    types = edge_types[perm]

    rowptr = index2ptr(row, num_nodes)
    return rowptr, col, types

def coo_to_csr_hyper(row, col, edge_types, pos_index, num_nodes=None):
    # The only difference is that now col is a 2D tensor
    # Row is the source node, col is the destination node list. 
    if num_nodes is None:
        num_nodes = int(row.max()) + 1

    row, perm = index_sort(row, max_value=num_nodes) # TODO: alternatively we can use stable
    col = col[:,perm] # 
    types = edge_types[perm]
    pos_index = pos_index[:, perm]
    rowptr = index2ptr(row, num_nodes)
    return rowptr, col, types, pos_index


def smart_split(edge_index):
    max_arity = edge_index.shape[0]
    file = torch.cat([
            torch.cat([edge_index[:arity,:], edge_index[arity+1:,:]], dim = 0) # exclude the current arity
            for arity in range(max_arity)]
            , dim = 1
            )
    return file

def preprocess_triton_hypergraph(edge_index, edge_type, num_node):
    max_arity = edge_index.shape[0]
    destination = edge_index.flatten()
    source = smart_split(edge_index)
    edge_type = edge_type.repeat(max_arity) # expand as if destination

    # Apply the sequence tensor to the non-zero elements
    pos_node_in_edge = torch.arange(1, max_arity+1).unsqueeze(1).repeat(1, edge_index.shape[1]).to(edge_index.device)
    pos_index = smart_split(pos_node_in_edge)

    assert pos_index.shape == source.shape, "pos_index and source should have the same shape"
    # Remove the destination node that is 0
    mask = destination != 0
    destination = destination[mask]
    source = source[:, mask]
    edge_type = edge_type[mask]
    pos_index  = pos_index[:, mask]

    rowptr, indices, etypes, pos_index = coo_to_csr_hyper(destination, source, edge_type, pos_index, num_node)

    num_rel = edge_type.max().item() + 1
    # also create a tensor of shape [num_node, relation] to indicate the source node and incoming degree of edge type

    node_edge_type_degree = torch.sparse_coo_tensor(
        torch.stack([destination, edge_type], dim = 0),
        torch.ones_like(edge_type), device = edge_index.device).coalesce().to_dense().transpose(0,1)
    assert node_edge_type_degree.stride() == (1, num_rel), f"node_edge_type_degree should have stride (1, num_rels), but have {node_edge_type_degree.stride()} instead"

    return rowptr, indices, etypes, pos_index, node_edge_type_degree
