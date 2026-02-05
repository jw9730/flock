from typing import Optional, Tuple
import copy
import numpy as np
import torch
from torch import nn, Tensor
from torch_geometric.data import Data

import graph_walker

from . import tasks, util
from .consensus import consensus_mean, consensus_softmax_stable

torch.set_float32_matmul_precision("high")


def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def to_csr_tensor(edge_index: Tensor, edge_type: Tensor, num_nodes=None):
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1
    # get source and destination nodes
    src = edge_index[0]
    dst = edge_index[1]
    # sort edges by source node, then destination node
    # this is crucial for proper CSR format
    sort_idx = torch.argsort(src * num_nodes + dst)
    src_sorted = src[sort_idx]
    dst_sorted = dst[sort_idx]
    edge_type_sorted = edge_type[sort_idx]
    # count edges per source node
    src_counts = torch.bincount(src_sorted, minlength=num_nodes)
    # build indptr (crow_indices)
    indptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=edge_index.device)
    indptr[1:] = torch.cumsum(src_counts, dim=0)
    # convert to numpy arrays with specified dtype
    indptr_np = indptr.cpu().numpy().astype(np.uint32)
    indices_np = dst_sorted.cpu().numpy().astype(np.uint32)
    data_np = edge_type_sorted.cpu().numpy().astype(np.uint32)
    return indptr_np, indices_np, data_np


def balanced_sample(
    x: Tensor, x_types: Tensor, num_types: int, samples_per_type: int
) -> Tuple[Tensor, Tensor]:
    # num_types must be greater than or equal to unique type count shown in x_types
    # if num_types is smaller, than it means some types does not exist in x_types
    # assert num_types >= len(torch.unique(x_types)), "Number of types must be greater than or equal to unique type count shown in x_types"
    sampled_indices = []
    for i in range(num_types):
        # get indices for the current type
        type_indices = (x_types == i).nonzero(as_tuple=True)[0]
        # if type does not exist in x_types, skip it
        if len(type_indices) == 0:
            continue
        # randomly choose indices for this type
        num_repeats = (samples_per_type + len(type_indices) - 1) // len(type_indices)
        rand_indices = torch.cat(
            [
                torch.randperm(len(type_indices), device=x.device)
                for _ in range(num_repeats)
            ]
        )[:samples_per_type]
        assert len(rand_indices) == samples_per_type
        # collect sampled indices
        sampled_indices.append(type_indices[rand_indices])
    # concatenate indices
    indices = torch.cat(sampled_indices)
    return x.t()[indices].contiguous(), x_types[indices].contiguous()


def remove_easy_edges(data, remove_one_hop, h_index, t_index, r_index):
    # we remove training edges (we need to predict them at training time) from the edge index
    # think of it as a dynamic edge dropout
    h_index_ext = torch.cat([h_index, t_index], dim=-1)
    t_index_ext = torch.cat([t_index, h_index], dim=-1)
    r_index_ext = torch.cat([r_index, r_index + data.num_relations // 2], dim=-1)
    if remove_one_hop:
        # we remove all existing immediate edges between heads and tails in the batch
        edge_index = data.edge_index
        easy_edge = torch.stack([h_index_ext, t_index_ext]).flatten(1)
        index = tasks.edge_match(edge_index, easy_edge)[0]
        mask = ~index_to_mask(index, data.num_edges)
    else:
        # we remove existing immediate edges between heads and tails in the batch with the given relation
        edge_index = torch.cat([data.edge_index, data.edge_type.unsqueeze(0)])
        # note that here we add relation types r_index_ext to the matching query
        easy_edge = torch.stack([h_index_ext, t_index_ext, r_index_ext]).flatten(1)
        index = tasks.edge_match(edge_index, easy_edge)[0]
        mask = ~index_to_mask(index, data.num_edges)
    data = copy.copy(data)
    data.edge_index = data.edge_index[:, mask]
    data.edge_type = data.edge_type[mask]
    return data


def negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel):
    # convert p(h | t, r) to p(t' | h', r')
    # h' = t, r' = r^{-1}, t' = h
    is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
    new_h_index = torch.where(is_t_neg, h_index, t_index)
    new_t_index = torch.where(is_t_neg, t_index, h_index)
    new_r_index = torch.where(is_t_neg, r_index, r_index + num_direct_rel)
    return new_h_index, new_t_index, new_r_index


class RMSNorm(nn.Module):
    """https://github.com/meta-llama/llama3/blob/main/llama/model.py"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    """https://github.com/meta-llama/llama3/blob/main/llama/model.py"""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class BidirectionalGRU(nn.Module):
    def __init__(self, dim: int, n_layers: int, multiple_of: int, norm_eps: float):
        super().__init__()
        self.gru_norm = RMSNorm(dim, eps=norm_eps)
        self.gru = nn.GRU(
            dim, dim, num_layers=n_layers, batch_first=True, bidirectional=True
        )
        self.gru_out = nn.Linear(2 * dim, dim, bias=False)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.gru_out(self.gru(self.gru_norm(x))[0])
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Flock(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        self.verbose = model_cfg["verbose"]
        self.test_samples = model_cfg.get("test_samples", 1)

        # preprocessing
        self.drop_edge_rate = 0.0
        self.remove_one_hop = False

        # random walk
        self.N = model_cfg["walk_num"]
        self.L = model_cfg["walk_len"]
        self.record_neighbors = model_cfg["record_neighbors"]

        # localized walk
        self.use_local_walk = model_cfg.get("local_walk", False)
        self.restart_prob = model_cfg.get("restart_prob", 0.0)
        self.restart_period = model_cfg.get("restart_period", None)

        # refinements
        self.T = model_cfg["refinements"]
        self.attention_scatter = model_cfg["attention_scatter"]
        self.H = model_cfg["attention_scatter_n_heads"]
        self.additive_refinement = model_cfg["additive_refinement"]
        self.embed_first_only = model_cfg["embed_only_first_refinement"]
        self.embedding_tying = model_cfg["embedding_tying_across_refinements"]
        self.parameter_tying = model_cfg["parameter_tying_across_refinements"]

        # body
        self.D = model_cfg["hidden_dim"]
        self.dtype = getattr(torch, model_cfg["dtype"])

        # embeddings
        self.node_init = nn.Parameter(torch.randn(self.D, dtype=self.dtype))
        self.type_init = nn.Parameter(torch.randn(self.D, dtype=self.dtype))
        self.emb_anon_node = nn.ModuleList()
        self.emb_anon_type = nn.ModuleList()
        self.emb_restart = nn.ModuleList()
        self.emb_neighbor = nn.ModuleList()
        self.emb_direction = nn.ModuleList()
        self.emb_node_is_query = nn.ModuleList()
        self.emb_type_is_query = nn.ModuleList()

        # sequence network
        self.net = nn.ModuleList()
        self.from_node = nn.ModuleList()
        self.from_type = nn.ModuleList()
        self.to_node = nn.ModuleList()
        self.to_type = nn.ModuleList()
        self.node_logit = nn.ModuleList()
        self.type_logit = nn.ModuleList()

        for _ in range(self.T if not self.embedding_tying else 1):
            # anon_type     -1 (=self.walk_len) no type
            # *_is_query    0 not query, 1 query
            # restart       0 walk, 1 restart
            # neighbor      0 walk, 1 neighbor
            # direction     0 downstream, 1 upstream, 2 loop, 3 no direction
            self.emb_anon_node.append(nn.Embedding(self.L, self.D).to(self.dtype))
            self.emb_anon_type.append(nn.Embedding(self.L + 1, self.D).to(self.dtype))
            self.emb_restart.append(nn.Embedding(2, self.D).to(self.dtype))
            self.emb_neighbor.append(nn.Embedding(2, self.D).to(self.dtype))
            self.emb_direction.append(nn.Embedding(4, self.D).to(self.dtype))
            self.emb_node_is_query.append(nn.Embedding(2, self.D).to(self.dtype))
            self.emb_type_is_query.append(nn.Embedding(2, self.D).to(self.dtype))

        for _ in range(self.T if not self.parameter_tying else 1):
            if model_cfg["net"] == "gru":
                self.net.append(
                    BidirectionalGRU(
                        dim=self.D,
                        n_layers=model_cfg["n_layers"],
                        multiple_of=self.D,
                        norm_eps=1e-5,
                    ).to(self.dtype)
                )
            else:
                raise NotImplementedError(
                    f"Network {model_cfg['net']} is not implemented"
                )

            self.from_node.append(nn.Linear(self.D, self.D).to(self.dtype))
            self.from_type.append(nn.Linear(self.D, self.D).to(self.dtype))
            self.to_node.append(nn.Linear(self.D, self.D).to(self.dtype))
            self.to_type.append(nn.Linear(self.D, self.D).to(self.dtype))

            if self.attention_scatter:
                self.node_logit.append(nn.Linear(self.D, self.H).to(self.dtype))
                self.type_logit.append(nn.Linear(self.D, self.H).to(self.dtype))

        # head
        mlp_hidden_dim = 128
        self.head = nn.Sequential(
            nn.Linear(self.D, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, 1),
        ).to(self.dtype)

    def _walk(
        self,
        data: Data,
        prefix: Tensor,
        remove_loops: bool,
        localize: bool,
        prefix_types: Optional[Tensor] = None,
    ) -> Tuple:
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        num_types = data.num_relations.item()
        device = prefix.device

        assert isinstance(edge_index, Tensor)
        assert isinstance(num_nodes, int)
        assert isinstance(device, torch.device)

        # for random walks, treat edges as undirected, untyped and remove duplicates
        # remove loops if specified
        edge_index_ = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        edge_index_ = torch.unique(edge_index_, dim=1)
        if remove_loops:
            edge_index_ = edge_index_[:, edge_index_[0] != edge_index_[1]]

        # run random walks
        walks, restarts = graph_walker.random_walks_fast(
            graph=Data(
                edge_index=edge_index_, num_nodes=num_nodes, is_directed_hash=False
            ),
            n_walks=1,
            walk_len=self.L,
            p=1,
            q=1,
            alpha=self.restart_prob if localize else 0.0,
            k=self.restart_period if localize else None,
            no_backtrack=True,
            prefix=prefix.cpu(),
            verbose=False,
        )

        if not self.record_neighbors:
            # anonymize node
            named_walks = graph_walker._anonymize(walks)
            neighbors = np.zeros_like(restarts, dtype=np.int32)

            # parse edge types and directions
            # for this, use the original edge_index
            types, directions = graph_walker._parse_edge_types_and_directions(
                walks,
                restarts,
                *to_csr_tensor(edge_index, data.edge_type, num_nodes),
                *to_csr_tensor(edge_index[[1, 0]], data.edge_type, num_nodes),
                graph_walker._seed(None),
            )

        else:
            # anonymize node and record named neighbors
            named_walks, walks, restarts, neighbors = (
                graph_walker._anonymize_with_neighbors(
                    walks,
                    restarts,
                    *to_csr_tensor(
                        edge_index_, torch.ones_like(edge_index_[0]), num_nodes
                    )[:2],
                )
            )

            # parse edge types and directions
            # for this, use the original edge_index
            types, directions = (
                graph_walker._parse_edge_types_and_directions_with_neighbors(
                    walks,
                    restarts,
                    neighbors,
                    *to_csr_tensor(edge_index, data.edge_type, num_nodes),
                    *to_csr_tensor(edge_index[[1, 0]], data.edge_type, num_nodes),
                    graph_walker._seed(None),
                )
            )

        # fix first-step edge type (prefix) and direction (0; downstream)
        if prefix_types is not None:
            assert prefix.ndim == 2 and prefix.shape[1] == 2
            types[:, 1] = np.asarray(prefix_types.cpu(), dtype=np.int32)
            directions[:, 1] = 0

        # anonymize edge types, ignoring no-type markers (-1)
        named_types = graph_walker._anonymize_edge_types(types)

        # convert to tensors
        walks, named_walks, restarts, neighbors, types, named_types, directions = map(
            lambda x: torch.tensor(x.astype(np.int32), dtype=torch.long, device=device),
            (walks, named_walks, restarts, neighbors, types, named_types, directions),
        )

        # handle no edge types and directions
        types[types == -1] = num_types
        named_types[named_types == -1] = self.L + 1
        directions[directions == -1] = 3

        return walks, named_walks, restarts, neighbors, types, named_types, directions

    def walks(self, data: Data, head_index: Tensor, localize: bool = False) -> tuple:
        device = head_index.device
        num_nodes = data.num_nodes
        num_types = data.num_relations.item()
        bsize = head_index.shape[0]

        assert isinstance(data.edge_index, Tensor)
        assert isinstance(num_nodes, int)

        # random walks starting at: head nodes and random nodes
        # sample starting nodes
        start_nodes = torch.cat(
            [
                head_index.repeat(self.N * self.T),
                torch.randint(num_nodes, (self.N * self.T * bsize,), device=device),
            ],
            dim=0,
        )
        assert start_nodes.numel() == 2 * self.N * self.T * bsize
        # random walks
        walks, named_walks, restarts, neighbors, types, named_types, directions = (
            self._walk(data, start_nodes, remove_loops=True, localize=localize)
        )

        # random walks starting at: edge types
        # number of walks per edge type, iteration and batch instance
        n1 = (self.N // num_types) + 1
        # number of walks per edge type
        n2 = n1 * self.T * bsize
        # sample starting edges
        start_edges, start_types = balanced_sample(
            data.edge_index, data.edge_type, num_types, n2
        )
        # random walks
        (
            walks_,
            named_walks_,
            restarts_,
            neighbors_,
            types_,
            named_types_,
            directions_,
        ) = self._walk(
            data,
            start_edges,
            remove_loops=False,
            localize=localize,
            prefix_types=start_types,
        )

        # number of actually sampled walks, per iteration and batch instance
        # we generally expect _N >= self.N, but the opposite may happen (e.g., NELLInductive:v4)
        _N = start_edges.shape[0] // (self.T * bsize)

        # sample permutations for subsampling
        # same across attributes (walks, restarts, ...) and walk length
        # but different for each iteration and batch instance
        perms = torch.argsort(torch.rand(self.T * bsize, _N, device=device))
        perms = perms.view(self.T, bsize, _N).permute(2, 0, 1)
        perms = perms[..., None].expand(_N, self.T, bsize, self.L)

        # combine walks
        def combine(x: Tensor, y: Tensor) -> Tensor:
            assert x.ndim == y.ndim == 2
            x = x.view(2 * self.N, self.T, bsize, self.L)
            y = y.view(_N, self.T, bsize, self.L)
            # subsampling
            y = y.gather(0, perms)
            if _N >= self.N:
                y = y[: self.N]
            else:
                y = y.repeat(self.N // _N + 1, 1, 1, 1)[: self.N]
            return torch.cat([x, y], dim=0).permute(1, 2, 0, 3)

        return (
            combine(walks, walks_),
            combine(named_walks, named_walks_),
            combine(restarts, restarts_),
            combine(neighbors, neighbors_),
            combine(types, types_),
            combine(named_types, named_types_),
            combine(directions, directions_),
        )

    def _forward(
        self, data: Data, head_index: Tensor, tail_index: Tensor, type_index: Tensor
    ) -> Tensor:
        """
        Arguments:
            data:
                edge_index: [2, |E|]
                edge_type: [|E|,]
                target_edge_index: [2, |T|]
                target_edge_type: [|T|,]
                num_relations: [1,]
                device: torch.device,
                num_nodes: int
            head_index: [batch_size,]
            tail_index: [batch_size, num_negative + 1]
            type_index: [batch_size,]
        Return:
            score: [batch_size, num_negative + 1]
        """
        num_nodes = data.num_nodes
        num_types = data.num_relations.item()
        bsize, num_tails = tail_index.shape
        device = head_index.device

        assert isinstance(data.edge_index, Tensor)
        assert isinstance(num_nodes, int)
        assert isinstance(device, torch.device)

        # [T, bsize, samples, len]
        walks, named_walks, restarts, neighbors, types, named_types, directions = (
            self.walks(data, head_index)
        )
        if self.use_local_walk:
            results = self.walks(data, head_index, localize=True)
            walks, named_walks, restarts, neighbors, types, named_types, directions = (
                torch.cat([walks, results[0]], dim=2),
                torch.cat([named_walks, results[1]], dim=2),
                torch.cat([restarts, results[2]], dim=2),
                torch.cat([neighbors, results[3]], dim=2),
                torch.cat([types, results[4]], dim=2),
                torch.cat([named_types, results[5]], dim=2),
                torch.cat([directions, results[6]], dim=2),
            )
        samples = walks.shape[2]

        # initialize node and type states
        h_node = self.node_init[None, None].expand(bsize, num_nodes, self.D)
        h_type = self.type_init[None, None].expand(bsize, num_types + 1, self.D)

        # initialize visited masks
        node_mask = torch.zeros(bsize, num_nodes, device=device, dtype=torch.bool)
        type_mask = torch.zeros(bsize, num_types + 1, device=device, dtype=torch.bool)

        # main processing
        _node_id = torch.arange(num_nodes, device=device)[None].expand(bsize, num_nodes)
        _type_id = torch.arange(num_types + 1, device=device)[None].expand(
            bsize, num_types + 1
        )
        for t in range(self.T):
            # _walks, _types: [bsize, samples * len]
            _walks = walks[t].flatten(1, 2)
            _types = types[t].flatten(1, 2)

            # currently visited mask
            node_mask_now = torch.isin(_node_id, _walks)
            type_mask_now = torch.isin(_type_id, _types)

            # embedding [bsize, samples, len, dim]
            if self.embed_first_only and t > 0:
                x = torch.zeros(bsize, samples, self.L, self.D, device=device)
            else:
                # boolean marker for query nodes and types [bsize, samples, len]
                is_h = torch.zeros(bsize, num_nodes, dtype=torch.long, device=device)
                is_r = torch.zeros(
                    bsize, num_types + 1, dtype=torch.long, device=device
                )
                is_h[torch.arange(bsize), head_index] = 1
                is_r[torch.arange(bsize), type_index] = 1
                is_h = is_h.gather(1, _walks).view(bsize, samples, self.L)
                is_r = is_r.gather(1, _types).view(bsize, samples, self.L)

                # all embeddings
                i = 0 if self.embedding_tying else t
                x = (
                    self.emb_anon_node[i](named_walks[t] - 1)
                    + self.emb_anon_type[i](named_types[t] - 1)
                    + self.emb_restart[i](restarts[t])
                    + self.emb_neighbor[i](neighbors[t])
                    + self.emb_direction[i](directions[t])
                    + self.emb_node_is_query[i](is_h)
                    + self.emb_type_is_query[i](is_r)
                )

            # previous node and type states
            x_node = h_node.gather(1, _walks[:, :, None].expand(-1, -1, self.D))
            x_type = h_type.gather(1, _types[:, :, None].expand(-1, -1, self.D))
            x_node = x_node.view(bsize, samples, self.L, self.D)
            x_type = x_type.view(bsize, samples, self.L, self.D)

            # forward
            i = 0 if self.parameter_tying else t

            # sequence network
            x = x + self.from_node[i](x_node) + self.from_type[i](x_type)
            x = x.view(bsize * samples, self.L, self.D)
            x = self.net[i](x)

            # scatter
            if self.attention_scatter:
                x_node = consensus_softmax_stable(
                    self.to_node[i](x), self.node_logit[i](x), walks[t], num_nodes
                )
                x_type = consensus_softmax_stable(
                    self.to_type[i](x), self.type_logit[i](x), types[t], num_types + 1
                )
            else:
                x_node = consensus_mean(self.to_node[i](x), walks[t], num_nodes)
                x_type = consensus_mean(self.to_type[i](x), types[t], num_types + 1)

            # update node and type states
            if self.additive_refinement:
                h_node = h_node + x_node
                h_type = h_type + x_type
            else:
                h_node = torch.where(node_mask_now[:, :, None], x_node, h_node)
                h_type = torch.where(type_mask_now[:, :, None], x_type, h_type)

            # update visited mask
            node_mask = torch.logical_or(node_mask, node_mask_now)
            type_mask = torch.logical_or(type_mask, type_mask_now)

            if self.verbose and util.get_rank() == 0:
                node_hit = (node_mask.count_nonzero() / node_mask.numel() * 100).item()
                type_hit = (type_mask.count_nonzero() / type_mask.numel() * 100).item()
                print(
                    f"step {t + 1} / {self.T} node {node_hit:.2f}% ({num_nodes}) type {type_hit:.2f}% ({num_types})"
                )

        # query features [bsize, num_tails, dim]
        h_query = h_node.gather(
            1, tail_index[:, :, None].expand(bsize, num_tails, self.D)
        ) + h_type.gather(1, type_index[:, None, None].expand(bsize, 1, self.D))

        # probability logit for each query (batch_size, num_tails)
        return self.head(h_query).squeeze(-1).to(torch.float32)

    def forward(self, data: Data, batch: Tensor):
        """
        Arguments:
            data:
                edge_index: [2, |E|]
                edge_type: [|E|,]
                target_edge_index: [2, |T|]
                target_edge_type: [|T|,]
                num_relations: [1,]
                device: torch.device,
                num_nodes: int
            batch: [batch_size, num_negative + 1, 3]
        Return:
            score: [batch_size, num_negative + 1]
        """
        device = data.device

        h_index, t_index, r_index = batch.unbind(-1)

        if self.training:
            data = remove_easy_edges(
                data, self.remove_one_hop, h_index, t_index, r_index
            )
            if self.drop_edge_rate > 0:
                assert isinstance(data.edge_index, Tensor)
                drop_edge_prob = (1 - self.drop_edge_rate) * torch.ones(
                    len(data.edge_type), device=device
                )
                drop_edge_mask = torch.bernoulli(drop_edge_prob).to(torch.bool)
                data.edge_index, data.edge_type = (
                    data.edge_index[:, drop_edge_mask],
                    data.edge_type[drop_edge_mask],
                )

        h_index, t_index, r_index = negative_sample_to_tail(
            h_index, t_index, r_index, num_direct_rel=data.num_relations // 2
        )

        # second dimension is redundant for h_index and r_index
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        head_index = h_index[:, 0]  # [batch_size,]
        type_index = r_index[:, 0]  # [batch_size,]
        tail_index = t_index  # [batch_size, num_negative + 1]

        if self.training or self.test_samples == 1:
            # probability logit for each tail node (batch_size, num_negative + 1)
            return self._forward(data, head_index, tail_index, type_index)

        # prediction ensembling for testing
        bsize = head_index.shape[0]
        num_tails = tail_index.shape[1]
        head_index = head_index.repeat_interleave(self.test_samples, dim=0)
        type_index = type_index.repeat_interleave(self.test_samples, dim=0)
        tail_index = tail_index.repeat_interleave(self.test_samples, dim=0)
        logits = self._forward(data, head_index, tail_index, type_index)
        return logits.view(bsize, self.test_samples, num_tails).mean(1)
