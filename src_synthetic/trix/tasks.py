from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch
from torch_geometric.utils import degree


def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match

def negative_sampling_relation(data, batch, num_negative, strict=True):
    h_index, t_index, r_index = batch.unbind(-1)
    pos = torch.zeros_like(r_index).tolist()
    r_index = r_index.tolist()
    index = list(range(len(r_index)))

    r_batch = all_negative_relation(data, batch)

    r_batch[index, [r_index, pos], :] = r_batch[index, [pos, r_index], :]
    return r_batch

def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # strict negative sampling vs random negative sampling
    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        # draw samples for negative heads
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)


def all_negative(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
    # generate all negative tails for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index, indexing="ij")  # indexing "xy" would return transposed
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    # generate all negative heads for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index, indexing="ij")
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)

    return t_batch, h_batch


def all_negative_relation(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    all_index = torch.arange(data.num_relations.item() // 2, device=batch.device)

    h_index, r_index = torch.meshgrid(pos_h_index, all_index, indexing="ij")
    t_index, r_index = torch.meshgrid(pos_t_index, all_index, indexing="ij")
    r_batch = torch.stack([h_index, t_index, r_index], dim=-1)

    return r_batch


def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = torch.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask


def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        # filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        # unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking

def compute_ranking_relation(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        # filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        # unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1)
    return ranking

def build_relation_graph(graph):
    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_node, num_relation = graph.num_nodes, graph.num_relations
    device = edge_index.device

    node_degree = degree(edge_index[1], num_nodes = num_node) + 1
    
    relation_adj = {}
    head_set = {}
    tail_set = {}

    index_hh = [[], [], []]
    value_hh = []

    index_ht = [[], [], []]
    value_ht = []

    index_th = [[], [], []]
    value_th = []

    index_tt = [[], [], []]
    value_tt = []

    for k in range(len(edge_index[0])):
        h = edge_index[0][k].item()
        t = edge_index[1][k].item()
        r = edge_type[k].item()

        if h in head_set:
            if r in head_set[h]:
                head_set[h][r] += 1
            else:
                head_set[h][r] = 1
        else:
            head_set[h] = {r:1}

        if t in tail_set:
            if r in tail_set[t]:
                tail_set[t][r] += 1
            else:
                tail_set[t][r] = 1
        else:
            tail_set[t] = {r:1}

    for node in head_set:
        heads = head_set.get(node, [])
        tails = tail_set.get(node, [])

        for r_1 in heads:
            for r_2 in heads:
                if r_1 != r_2:
                    index_hh[0].append(r_1)
                    index_hh[1].append(r_2)
                    index_hh[2].append(node)

            for r_2 in tails:
                index_ht[0].append(r_1)
                index_ht[1].append(r_2)
                index_ht[2].append(node)

        for r_1 in tails:
            for r_2 in tails:
                if r_1 != r_2:
                    index_tt[0].append(r_1)
                    index_tt[1].append(r_2)
                    index_tt[2].append(node)
            for r_2 in heads:
                index_th[0].append(r_1)
                index_th[1].append(r_2)
                index_th[2].append(node)

    relation_adj["hh"] = Data(
        edge_index=torch.Tensor(index_hh[0:2]).to(torch.int64), 
        edge_type=torch.Tensor(index_hh[2]).to(torch.int64),
        num_nodes=num_relation, 
        num_relations=num_node
    )
    relation_adj["ht"] = Data(
        edge_index=torch.Tensor(index_ht[0:2]).to(torch.int64), 
        edge_type=torch.Tensor(index_ht[2]).to(torch.int64),
        num_nodes=num_relation, 
        num_relations=num_node
    )
    relation_adj["th"] = Data(
        edge_index=torch.Tensor(index_th[0:2]).to(torch.int64), 
        edge_type=torch.Tensor(index_th[2]).to(torch.int64),
        num_nodes=num_relation, 
        num_relations=num_node
    )
    relation_adj["tt"] = Data(
        edge_index=torch.Tensor(index_tt[0:2]).to(torch.int64), 
        edge_type=torch.Tensor(index_tt[2]).to(torch.int64),
        num_nodes=num_relation, 
        num_relations=num_node
    )


    graph.relation_adj = relation_adj

    return graph


