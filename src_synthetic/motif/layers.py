import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree, scatter
from torch_geometric.utils import scatter

from motif.util import static_positional_encoding, coo_to_csr, preprocess_triton_hypergraph


class GeneralizedRelationalConv(MessagePassing):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="sum", layer_norm=False, activation="relu", dependent=False, project_relations=False, use_triton=True): 
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.project_relations = project_relations
        self.use_triton = use_triton

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)

        if dependent:
            # obtain relation embeddings as a projection of the query relation
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            if not self.project_relations:
                # relation embeddings as an independent embedding matrix per each layer
                self.relation = nn.Embedding(num_relation, input_dim)
            else:
                # will be initialized after the pass over relation graph
                self.relation = None
                self.relation_projection = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim)
                )


    def forward(self, input, query, boundary, edge_index, edge_type, size, edge_weight=None):
        batch_size = len(query)
        
        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            if not self.project_relations:
                # layer-specific relation features as a special embedding matrix unique to each layer
                relation = self.relation.weight.expand(batch_size, -1, -1)
            else:
                # NEW and only change: 
                # projecting relation features to unique features for this layer, then resizing for the current batch
                relation = self.relation_projection(self.relation)
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)

        # note that we send the initial boundary condition (node states at layer0) to the message passing
        # correspond to Eq.6 on p5 in https://arxiv.org/pdf/2106.06935.pdf
        output = self.propagate(edge_index=edge_index,
                                edge_type=edge_type, size=size, 
                                edge_weight=edge_weight, input=input, 
                                relation=relation, boundary=boundary)
        return output

    def propagate(self, edge_index, size=None, **kwargs):
        if kwargs["edge_weight"].requires_grad or self.message_func == "rotate":
            # the rspmm cuda kernel only works for TransE and DistMult message functions
            # otherwise we invoke separate message & aggregate functions
            return super(GeneralizedRelationalConv, self).propagate(edge_index, size, **kwargs)

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        # in newer PyG, 
        # __check_input__ -> _check_input()
        # __collect__ -> _collect()
        # __fused_user_args__ -> _fuser_user_args
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._fused_user_args, edge_index, size, kwargs)


        # TODO: use from packaging.version import parse as parse_version as by default 2.4 > 2.14 which is wrong
        pyg_version = [int(i) for i in torch_geometric.__version__.split(".")]
        col_fn = self.inspector.distribute if pyg_version[1] <= 4 else self.inspector.collect_param_data
        msg_aggr_kwargs = col_fn("message_and_aggregate", coll_dict)



        for hook in self._message_and_aggregate_forward_pre_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs))
            if res is not None:
                edge_index, msg_aggr_kwargs = res
        out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        for hook in self._message_and_aggregate_forward_hooks.values():
            res = hook(self, (edge_index, msg_aggr_kwargs), out)
            if res is not None:
                out = res


        update_kwargs = col_fn("update", coll_dict)

        out = self.update(out, **update_kwargs)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res

        return out

    def message(self, input_j, relation, boundary, edge_type):
        relation_j = relation.index_select(self.node_dim, edge_type)

        if self.message_func == "transe":
            message = input_j + relation_j
        elif self.message_func == "distmult":
            message = input_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # augment messages with the boundary condition
        message = torch.cat([message, boundary], dim=self.node_dim)  # (num_edges + num_nodes, batch_size, input_dim)

        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        # augment aggregation index with self-loops for the boundary condition
        index = torch.cat([index, torch.arange(dim_size, device=input.device)]) # (num_edges + num_nodes,)
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)

        if self.aggregate_func == "pna":
            mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            max = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="max")
            min = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="min")
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            output = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size,
                             reduce=self.aggregate_func)

        return output

    def message_and_aggregate(self, edge_index, input, relation, boundary, edge_type, edge_weight, index, dim_size):
        # fused computation of message and aggregate steps with the custom rspmm cuda kernel
        # speed up computation by several times
        batch_size, num_node = input.shape[:2]
        input = input.transpose(0, 1).flatten(1)
        relation = relation.transpose(0, 1).flatten(1)
        boundary = boundary.transpose(0, 1).flatten(1)
        degree_out = degree(index, dim_size).unsqueeze(-1) + 1
        if self.use_triton:
            from .rspmm.triton_rspmm import RelConvSumAggr
            rowptr, indices, etypes = coo_to_csr(edge_index[0], edge_index[1], edge_type, num_node)
            if self.aggregate_func == "sum":
                update = RelConvSumAggr.apply(input, rowptr, indices, num_node, etypes, relation, 0)
                update = update + boundary
            elif self.aggregate_func == "mean":
                update = RelConvSumAggr.apply(input, rowptr, indices, num_node, etypes, relation, 0)
                update = (update + boundary) / degree_out
            else:
                raise ValueError("For now, the Triton kernel only supports sum aggr. Unknown aggregation function `%s`" % self.aggregate_func)
        else:
            # reduce memory complexity from O(|E|d) to O(|V|d), so we can apply it to larger graphs
            from .rspmm import generalized_rspmm

            if self.message_func in self.message2mul:
                mul = self.message2mul[self.message_func]
            else:
                raise ValueError("Unknown message function `%s`" % self.message_func)
            if self.aggregate_func == "sum":
                update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
                update = update + boundary
            elif self.aggregate_func == "mean":
                update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
                update = (update + boundary) / degree_out
            elif self.aggregate_func == "max":
                update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="max", mul=mul)
                update = torch.max(update, boundary)
            elif self.aggregate_func == "pna":
                # we use PNA with 4 aggregators (mean / max / min / std)
                # and 3 scalars (identity / log degree / reciprocal of log degree)
                sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
                sq_sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation ** 2, input ** 2, sum="add",
                                        mul=mul)
                max = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="max", mul=mul)
                min = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="min", mul=mul)
                mean = (sum + boundary) / degree_out
                sq_mean = (sq_sum + boundary ** 2) / degree_out
                max = torch.max(max, boundary)
                min = torch.min(min, boundary) # (node, batch_size * input_dim)
                std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
                features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
                features = features.flatten(-2) # (node, batch_size * input_dim * 4)
                scale = degree_out.log()
                scale = scale / scale.mean()
                scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1) # (node, 3)
                update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2) # (node, batch_size * input_dim * 4 * 3)
            else:
                raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        update = update.view(num_node, batch_size, -1).transpose(0, 1)
        return update

    def update(self, update, input):
        # node update as a function of old states (input) and this layer output (update)
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


# Turn on triton if possible
class HypergraphLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_relation, max_arity = 6, dropout=0.2, aggregate_func = "sum", norm = "layer_norm", dependent = False, use_triton = True):
        super(HypergraphLayer, self).__init__()
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels * 2, out_channels)
        self.num_relation = num_relation
        self.norm_type = norm
        self.dependent = dependent
        self.use_triton = use_triton
        self.aggregate_func = aggregate_func
        if norm == "layer_norm":
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if self.dependent:
            self.relation_linear = nn.Linear(in_channels, num_relation * in_channels)
        else:
            self.rel_embedding = nn.Embedding(num_relation, in_channels)

        
        self.dropout = nn.Dropout(p = dropout,inplace=False)
        
        self.pos_embedding = nn.Embedding(max_arity + 1, in_channels)
        
        
       
        

    def forward(self, node_features, query, edge_list, rel):
        self.pos_embedding.weight.data[0] = torch.ones(self.in_channels)
        batch_size, node_size, input_dim = node_features.shape
        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation_vector = self.relation_linear(query).view(batch_size, self.num_relation, input_dim)
        else:
            relation_vector = None
             
        node_features[:, 0, :] = 0 # Clear the padding node for message agg

        if self.use_triton:
            from .rspmm.triton_rspmm import HyperRelConvSumAggr, HyperRelConvMeanAggr
            pos_embedding = self.pos_embedding.weight.unsqueeze(1).expand(-1, batch_size, -1).flatten(1) # expand the positional encoding for batch, and compress with the feature size
            relation_vector = self.rel_embedding.weight.unsqueeze(1).expand(-1, batch_size, -1).flatten(1).transpose(0,1) # expand the relation embedding for batch, and compress with the feature size
            edge_list_trans = edge_list.transpose(0,1)
            node_features_flatten = node_features.transpose(0,1).flatten(1)
            rowptr, indices, etypes, pos_index, _ = preprocess_triton_hypergraph(edge_list_trans, rel, num_node = node_size)
            if self.aggregate_func == "sum":
                out  = HyperRelConvSumAggr.apply(node_features_flatten, rowptr, indices, node_size, etypes, relation_vector, pos_embedding, pos_index, 0)
            elif self.aggregate_func == "mean":
                out  = HyperRelConvMeanAggr.apply(node_features_flatten, rowptr, indices, node_size, etypes, relation_vector, pos_embedding, pos_index, 0)
            else:
                raise ValueError("For now, the Triton kernel only supports sum and mean aggr. Unknown aggregation function `%s`" % self.aggregate_func)
            out = out.view(node_size, batch_size, -1).transpose(0,1)
        else:
            message = self.messages(node_features, relation_vector, edge_list, rel)
            out = self.aggregates(message, edge_list, rel, node_features)
            out[:, 0, :] = 0 # Clear the padding node for learning

        out = (self.linear(torch.cat([out, node_features], dim=-1)))
        out = self.dropout(out)
    
        if self.norm_type == "layer_norm":
            out = self.norm(out)

        return out
    
    def messages(self, node_features, relation_vector, hyperedges, relations):
        device = node_features.device
        # Set the node feature of node 0 to be always 0 so that it does not contribute to the messages

        batch_size, _ , input_dim = node_features.shape
        edge_size, max_arity = hyperedges.shape

        # Create a batch index array
        batch_indices = torch.arange(batch_size, device=hyperedges.device)[:, None, None]  

        # Repeat batch indices to match the shape of hyperedges
        batch_indices = batch_indices.repeat(1, hyperedges.shape[0], hyperedges.shape[1]) 

        # Use advanced indexing to gather node features
        sum_node_positional = node_features[batch_indices, hyperedges]


        # Compute positional encodings for nodes in each hyperedge
        positional_encodings = self.computer_pos_encoding(hyperedges, batch_size, device)

        sum_node_positional = sum_node_positional + positional_encodings

        messages = self.all_but_one_trick(sum_node_positional, batch_size, edge_size, input_dim, device)
        
        # Get relation vectors for each edge and expand
        if relation_vector is not None:
            assert self.dependent
            relation_vectors = relation_vector.index_select(1, relations)
            relation_vectors = relation_vectors.unsqueeze(2).expand(-1, -1, max_arity, -1)
        else:
            assert not self.dependent
            relation_vectors = self.rel_embedding(relations).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, max_arity, -1)
        
        messages = messages * relation_vectors

        # shape: [batch_size,  edge_size, max_arity, input_dim]
        return messages

    def aggregates(self, messages, hyperedges, relations, node_features):
        batch_size, node_size, input_dim = node_features.shape
        edge_size, max_arity = hyperedges.shape
        messages_expanded = messages.view(batch_size, edge_size * max_arity, input_dim)
        node_aggregate = scatter(messages_expanded, hyperedges.flatten(), dim = 1, reduce = "sum", dim_size=node_size)
        
        return node_aggregate


        
    def all_but_one_trick(self, sum_node_positional, batch_size, edge_size, input_dim, device):
        cumprod_forward = torch.cumprod(sum_node_positional, dim=2)
        cumprod_backward = torch.cumprod(sum_node_positional.flip(dims=[2]), dim=2).flip(dims=[2])

        # Shift and combine
        shifted_forward = torch.cat([torch.ones(batch_size, edge_size, 1, input_dim).to(device), cumprod_forward[:, :, :-1, :]], dim=2)
        shifted_backward = torch.cat([cumprod_backward[:, :, 1:, :], torch.ones(batch_size, edge_size, 1, input_dim).to(device)], dim=2)

        # Combine the two shifted products
        return shifted_forward * shifted_backward

    def computer_pos_encoding(self, hyperedges, batch_size, device):
        
        sequence_tensor = torch.arange(1, hyperedges.size(1) + 1, device = device).unsqueeze(0)
        # Apply the sequence tensor to the non-zero elements
        pos_node_in_edge = torch.where(hyperedges != 0, sequence_tensor, torch.zeros_like(hyperedges, device = device))

        return self.pos_embedding(pos_node_in_edge).unsqueeze(0).expand(batch_size, -1, -1, -1)
    

