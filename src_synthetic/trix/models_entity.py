import math
import torch
from torch import nn

from . import tasks, layers
from trix.base_nbfnet import BaseNBFNet

class TRIX(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_1_cfg, entity_model_2_cfg, num_test_samples=1):
        super(TRIX, self).__init__()

        self.relation_model = RelNet(**rel_model_cfg)
        self.entity_model_1 = EntityNet(**entity_model_1_cfg)
        self.entity_model_2 = EntityNet(**entity_model_2_cfg)
        self.num_test_samples = num_test_samples
        
    def forward(self, data, batch):
        if not self.training and self.num_test_samples > 1:
            original_batch_size = batch.shape[0]
            batch = batch.repeat_interleave(self.num_test_samples, dim=0)

            # iterative updates: relation 1 - entity 1 - relation 1 - entity 3
            relation_representations = self.relation_model(data, batch, self.entity_model_1)

            # data is ground truth graph
            # batch is n by 3
            # relation representation is n by node by dim
            score = self.entity_model_2(data, relation_representations, batch)["score"]
            score = score.view(original_batch_size, self.num_test_samples, -1).mean(1)
            return score
        else:
            # iterative updates: relation 1 - entity 1 - relation 1 - entity 3
            relation_representations = self.relation_model(data, batch, self.entity_model_1)

            # data is ground truth graph
            # batch is n by 3
            # relation representation is n by node by dim
            score = self.entity_model_2(data, relation_representations, batch)["score"]
            return score
    
    def relation(self, data, batch):
        relation_representations = self.relation_model(data, batch, self.entity_model_1)
        return relation_representations


# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations 
# of shape [bs, num_rel, hidden]
class RelNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, random_feature_ratio=0, noise_magnitude=0, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)
            
        self.node_mlp = torch.nn.Linear(self.dims[-1] * 2, self.dims[-1])
        self.num_random_features = math.floor(self.dims[0] * random_feature_ratio)
        self.noise_magnitude = noise_magnitude

        self.layers_hh = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers_hh.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
                )
            
        self.layers_ht = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers_ht.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
                )
            
        self.layers_th = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers_th.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
                )
            
        self.layers_tt = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers_tt.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
                )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    def forward(self, data, batch, entity_model_1):
        rel_graph = data.relation_adj
        h_index = batch[:, 0, 2]

        # inject noise to the initial node representations
        if self.num_random_features == 0:
            # no random features
            node_representations = torch.ones(len(h_index), rel_graph["hh"].num_relations, self.layers_hh[0].input_dim).to(h_index.device)
        elif self.num_random_features < self.layers_hh[0].input_dim:
            num_deterministic_features = self.layers_hh[0].input_dim - self.num_random_features
            deterministic_representations = torch.ones(len(h_index), rel_graph["hh"].num_relations, num_deterministic_features).to(h_index.device)
            random_representations = torch.rand(len(h_index), rel_graph["hh"].num_relations, self.num_random_features).to(h_index.device) * self.noise_magnitude
            node_representations = torch.cat([deterministic_representations, random_representations], dim=-1)
        else:
            # 100% random features
            node_representations = torch.rand(len(h_index), rel_graph["hh"].num_relations, self.layers_hh[0].input_dim).to(h_index.device) * self.noise_magnitude

        batch_size = len(h_index)
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)
        
        boundary = torch.zeros(batch_size, rel_graph["hh"].num_nodes, self.dims[0], device=h_index.device)
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (rel_graph["hh"].num_nodes, rel_graph["hh"].num_nodes)
        edge_weight_hh = torch.ones(rel_graph["hh"].num_edges, device=h_index.device)
        edge_weight_ht = torch.ones(rel_graph["ht"].num_edges, device=h_index.device)
        edge_weight_th = torch.ones(rel_graph["th"].num_edges, device=h_index.device)
        edge_weight_tt = torch.ones(rel_graph["tt"].num_edges, device=h_index.device)

        hiddens = []
        layer_input = boundary

        for i in range(len(self.layers_hh)):
            self.layers_hh[i].relation = node_representations
            self.layers_ht[i].relation = node_representations
            self.layers_th[i].relation = node_representations
            self.layers_tt[i].relation = node_representations

            hidden_hh = self.layers_hh[i](layer_input, query, boundary, rel_graph["hh"].edge_index, rel_graph["hh"].edge_type, size, edge_weight_hh)
            hidden_ht = self.layers_ht[i](layer_input, query, boundary, rel_graph["ht"].edge_index, rel_graph["ht"].edge_type, size, edge_weight_ht)
            hidden_th = self.layers_th[i](layer_input, query, boundary, rel_graph["th"].edge_index, rel_graph["th"].edge_type, size, edge_weight_th)
            # hidden_tt = self.layers_tt[i](layer_input, query, boundary, rel_graph["tt"].edge_index, rel_graph["tt"].edge_type, size, edge_weight_tt)

            hidden = hidden_hh + hidden_ht + hidden_th
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)

            layer_input = hidden
            relation_representations = layer_input
            
            if i == 0:
                node_representations = self.node_mlp(entity_model_1(data, relation_representations, batch)["feature"]).reshape(len(h_index), rel_graph["hh"].num_relations, -1)


        node_query = query.unsqueeze(1).expand(-1, rel_graph["hh"].num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]
     
        return output
    

class EntityNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, random_feature_ratio=0, noise_magnitude=0, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.num_random_features = math.floor(self.dims[0] * random_feature_ratio)
        self.noise_magnitude = noise_magnitude

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
            )

        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    
    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        # inject noise to the initial node representations
        if self.num_random_features > 0:
            boundary[..., -self.num_random_features:] = boundary[..., -self.num_random_features:] - torch.rand_like(boundary[..., -self.num_random_features:]) * self.noise_magnitude
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, batch):
        h_index, t_index, r_index = batch.unbind(-1)

        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return {"score": score, "feature": output["node_feature"]}