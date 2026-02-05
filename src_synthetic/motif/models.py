import torch
from torch import nn
from torch.nn import functional as F
from motif.util import static_positional_encoding
from . import tasks, layers
from motif.base_nbfnet import BaseNBFNet

class Ultra(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg):
        super(Ultra, self).__init__()

        self.relation_model = RelNBFNet(**rel_model_cfg)
        self.entity_model = EntityNBFNet(**entity_model_cfg)

        
    def forward(self, data, batch):
        
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs

        score = self.entity_model(data, batch,  self.relation_model,  relation_hyper_flag = False)
        return score




# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations 
# of shape [bs, num_rel, hidden]
class RelNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=4, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
                )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    
    def bellmanford(self, data, h_index, separate_grad=False):
        batch_size = len(h_index)

        # initialize initial nodes (relations of interest in the batcj) with all ones
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        #boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
        # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
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
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, rel_graph, query):
        # message passing and updated node representations (that are in fact relations)
        output = self.bellmanford(rel_graph, h_index=query)["node_feature"]  # (batch_size, num_nodes, hidden_dim）
        
        return output



class MOTIF(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg):
        super(MOTIF, self).__init__()

        self.relation_model = RelHCNet(num_relation=7,**rel_model_cfg)
        self.entity_model = EntityNBFNet(**entity_model_cfg)

        
    def forward(self, data, batch):
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        score = self.entity_model(data, batch,  self.relation_model,  relation_hyper_flag = True)
        
        return score

class RelHCNet(nn.Module):
    def __init__(self, input_dim, num_relation, hidden_dims,
                 short_cut=True,  num_mlp_layer=2, max_arity=3, dropout=0.2,norm = "layer_norm", padding_idx = 0,dependent = False,aggregate_func = "sum", drop_edge_rate = 0.0,**kwargs):
        super(RelHCNet,self).__init__()
        self.name = "RelHCNet"
        self.aggregate_func = aggregate_func
        input_dim = input_dim
        self.drop_edge_rate = drop_edge_rate
        assert self.drop_edge_rate >= 0.0 and self.drop_edge_rate < 1.0
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut  # whether to use residual connections between layers
        
        self.max_arity  = max_arity
        self.padding_idx = padding_idx
        
        self.max_considered_arity = kwargs.get("max_considered_arity", 2) # for synthetic experiments
        
        static_encodings = static_positional_encoding(max_arity + 1, input_dim)
        # Fix the encoding
        self.position = nn.Embedding.from_pretrained(static_encodings, freeze=True)

        self.position.weight.data[self.padding_idx] = torch.ones(input_dim)


        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1): # num of hidden layers
            self.layers.append(layers.HypergraphLayer(self.dims[i], self.dims[i + 1], self.num_relation, dropout=dropout, norm = norm, dependent = dependent, aggregate_func=aggregate_func))

        self.feature_dim = input_dim

        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(self.feature_dim, self.feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(self.feature_dim, self.feature_dim))
        self.mlp = nn.Sequential(*mlp)

    def inference(self,  query_idx, edge_list, rel_list, num_nodes):
        batch_size = len(query_idx)
        
        query =  torch.ones(query_idx.shape[0], self.dims[0], device=query_idx.device, dtype=torch.float)
        index = query_idx.unsqueeze(-1).expand_as(query)
        query_feature = torch.zeros(batch_size, num_nodes, self.dims[0], device=query_idx.device)

        query_feature.scatter_add_(dim=1, 
                                   index = index.unsqueeze(1),
                                    src = query.unsqueeze(1)
                                )
        
        init_feature = query_feature

        init_feature[:, self.padding_idx, :] = 0 # clear the padding node

        # Passing in the layer:
        layer_input = init_feature

        for layer in self.layers:
            hidden = F.relu(layer(layer_input, query, edge_list, rel_list))
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden
        output = layer_input

        # Remind the model which query we are looking for
        score = self.mlp(output)

        return score

    def forward(self,  relation_hypergraph, query):
        edge_list, rel_list, num_nodes = relation_hypergraph.edge_index, relation_hypergraph.edge_type, relation_hypergraph.num_nodes
        
        if self.training and self.drop_edge_rate >= 0:
            drop_edge_mask = torch.bernoulli((1-self.drop_edge_rate) * torch.ones(len(rel_list), device = edge_list.device)).to(bool)
            edge_list = edge_list[:,drop_edge_mask]
            rel_list = rel_list[drop_edge_mask]

        
        # Shift edge_list by 1 to avoid the padding node
        edge_list = torch.transpose(edge_list,0,1) + 1
        query += 1
        num_nodes +=1

        relation_feature = self.inference(query, edge_list, rel_list, num_nodes)

        query -= 1
        return relation_feature[:,1:,:]



class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, drop_edge_rate = 0, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)
        assert drop_edge_rate >= 0 and drop_edge_rate < 1
        self.drop_edge_rate = drop_edge_rate
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

        # for synthetic experiments:
        self.synthetic = kwargs.get("synthetic", False)

    
    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        
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

    def forward(self, data, batch, relation_model, relation_hyper_flag = False):
        h_index, t_index, r_index = batch.unbind(-1)

        shape = h_index.shape
        
        # initial query representations are those from the relation graph
        if self.training and not self.synthetic:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)
            if self.drop_edge_rate > 0:
                drop_edge_mask = torch.bernoulli((1-self.drop_edge_rate) * torch.ones(len(data.edge_type), device = h_index.device)).to(bool)
                data.edge_index, data.edge_type = data.edge_index[:,drop_edge_mask], data.edge_type[drop_edge_mask]
                
         
            
            # turn all triples in a batch into a tail prediction mode
            if relation_hyper_flag:
                # build relation hypergraph
                data = tasks.build_relation_hypergraph(data)
            else:
                data = tasks.build_relation_graph(data)
        
        
        if not self.synthetic:
            h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        if relation_hyper_flag:
            if not self.synthetic:
                relation_representations = relation_model(data.relation_hypergraph, query=r_index[:, 0])
            else:
                index_to_use = relation_model.max_considered_arity-2
                relation_representations = relation_model(data.relation_hypergraph[0][index_to_use], query=r_index[:, 0])
        else:
            relation_representations = relation_model(data.relation_graph, query=r_index[:, 0])






        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        

        

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)

    

    


