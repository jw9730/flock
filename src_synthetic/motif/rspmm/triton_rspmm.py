import torch

import triton
import triton.language as tl


@triton.jit
def relconv_aggregate_sum_kernel_forward(
      out_h,         # h_out output tensor of shape[num_nodes_dst,IN_CHAN] ; This stores places for the output 
      adj_rowptr,    # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1] ; a cumulative count in CSR for each row. This is the pointer pointing the start of the memory
      adj_indices,   # graph adjacency matrix (indices) of shape[num_edges]; the "column index" count, position for each nodes
      h_src,         # h_src tensor of shape[num_nodes_src, IN_CHAN] ; This is the input x
      rel_indices,   # edge types of shape [num_edges]; relation type for each edge
      rels,          # relations matrix of shaoe [num_rels, IN_CHAN];  actual relation matrix. 
      IN_CHAN        : tl.constexpr, # number of features per node we are considering in this thread
      WG_SIZE        : tl.constexpr, # workgroup size
    ):
    # Pseudo-code (functional):
    #   for node_i in range(out_h.shape[0]):
    #     out_h[node_i,:] = 0
    #     col_start = adj_rowptr[node_i]
    #     col_count = adj_rowptr[node_i+1] – col_start
    #     for icol in range(col_count):
    #       node_j = adj_indices[col_start + icol]
    #       rel_j = rel_indices[col_start + icol]
    #       out_h[node_i,:] += h_src[node_j,:] * rels[rel_j, :] 

    # (Triton implementation for float32)
    # get current node_index and features to process
    node_index_i = tl.program_id(0) # Get which node we are operating one
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE # tl.program_id(1) = which feature group we are working on. There might be multiple groups 
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    # identity list of node indices to aggregate. Note these are sorted in the CSR format
    col_start = tl.load(adj_rowptr + node_index_i) # To extract the row[node_index_i], we want to get the index of starting column and ending column
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start # The sliced indexes for the extracted row (indicated by the column index).
    # aggregate neighboring features
    aggr_sum = feat_zeros
    for index in range(col_count):
        node_index_j = tl.load(adj_indices + col_start + index) # adj_indices points the stored total column index, then col_start is the start of the current considered row, index is the current edges in the considered row, which is what we want
        edge_type_j = tl.load(rel_indices + col_start + index) # same here, extract the correct [row, column] position for edges. 

        neighbor_feat_j = tl.load(h_src + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        rel_feat_j = tl.load(rels + edge_type_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        aggr_sum += rel_feat_j * neighbor_feat_j
    # col_count = tl.where(col_count == 0, 1, col_count)
    # aggr_mean = aggr_sum / col_count
    # store aggregator output
    tl.store(out_h + node_index_i * IN_CHAN + feat_offsets, aggr_sum, feat_valid_mask)

@triton.jit
def relconv_aggregate_sum_kernel_forward_v2u(
    out_h,         # h_out output tensor of shape[num_nodes_dst,IN_CHAN]
    adj_rowptr,    # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1]
    adj_indices,   # graph adjacency matrix (indices) of shape[num_edges]
    h_src,         # h_src tensor of shape[num_nodes_src, IN_CHAN]
    rel_indices,   # edge types of shape [num_edges]
    rels,          # relations matrix of shaoe [num_rels, IN_CHAN]
    IN_CHAN: tl.constexpr,  # number of features per head
    batch_size: tl.constexpr,  # Batch size
    n_nodes: tl.constexpr,
    WG_SIZE_NODES: tl.constexpr,  # workgroup size
    WG_SIZE_BATCH: tl.constexpr,  # workgroup size
    WG_SIZE_FEATS: tl.constexpr,  # workgroup size
):

    feat_offset = tl.arange(0, WG_SIZE_FEATS) + tl.program_id(0) * WG_SIZE_FEATS
    feat_offset_mask = feat_offset < IN_CHAN

    batch_offset = tl.arange(0, WG_SIZE_BATCH) + tl.program_id(1) * WG_SIZE_BATCH
    batch_offset_mask = batch_offset < batch_size

    batch_feat_mask = batch_offset_mask[:, None] and feat_offset_mask[None, :]
    batch_feat_offset = batch_offset[:, None] * IN_CHAN + feat_offset[None, :]

    row_start = tl.program_id(2) * WG_SIZE_NODES
    row_end = tl.minimum(row_start + WG_SIZE_NODES, n_nodes)

    for node_index in range(row_start, row_end):
        col_start = tl.load(adj_rowptr + node_index)
        col_end = tl.load(adj_rowptr + node_index + 1)
        col_count = col_end - col_start
        # aggregate neighboring features
        aggr_sum = tl.zeros((WG_SIZE_BATCH, WG_SIZE_FEATS), dtype=tl.float32)
        for index in range(col_count):
            node_index_j = tl.load(adj_indices + col_start + index)
            edge_type_j = tl.load(rel_indices + col_start + index)

            # Load the 2D rel feat vector. 
            rel_feat_j = tl.load(rels + edge_type_j * IN_CHAN * batch_size + batch_feat_offset, mask = batch_feat_mask)
            # Load a neighbor_feat_j for multiple minibatch elements. A 2D tensor
            neighbor_feat_j = tl.load(h_src + node_index_j * IN_CHAN * batch_size + batch_feat_offset,
                                      mask=batch_feat_mask)
            aggr_sum += rel_feat_j * neighbor_feat_j  # no broadcasting

        tl.store(out_h + node_index * batch_size * IN_CHAN +
                 batch_feat_offset, aggr_sum, mask=batch_feat_mask)
    
@triton.jit
def relconv_aggregate_sum_kernel_backward(
      dh_src,        # pre-initialized dh_src output tensor of shape[num_nodes_src,IN_CHAN]
      drel_src,      # pre-initialized drel output tensor of shape [num_relations, IN_CHAN]
      adj_rowptr,    # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1]
      adj_indices,   # graph adjacency matrix (indices) of shape[num_edges]
      h_in,          # original input features of shape [num_nodes, IN_CHAN]
      dh_out,        # MAIN GRADIENT: dh_out tensor of shape[num_nodes_src,IN_CHAN]
      rel_indices,   # edge types of shape [num_edges]
      rels,          # relations matrix of shape [num_rels, IN_CHAN]
      IN_CHAN        : tl.constexpr, # number of features per head
      WG_SIZE        : tl.constexpr, # workgroup size
    ):
    # Pseudo-code (functional):
    #   for node_i in range(dh_out.shape[0]):
    #     col_start = adj_rowptr[node_i]
    #     col_count = adj_rowptr[node_i+1] – col_start
    #     for icol in range(col_count):
    #       node_j = adj_indices[col_start + icol]
    #       rel_j = rel_indices[col_start + icol]
    #       nodej_feat = hin[node_j]
    #       relj_feat = rels[rel_j]
    #       dh_src[node_i,:] += dh * relj_feat
    #       drel_src[rel_j, :] += dh * nodej_feat

    # (Triton implementation for float32)
    # get current node_index and features to process
    node_index_i = tl.program_id(0)
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    # identify list of node indices to aggregate
    col_start = tl.load(adj_rowptr + node_index_i)
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start
    # load output gradient and scale
    h_out_grad = tl.load(dh_out + node_index_i * IN_CHAN + feat_offsets,
            feat_valid_mask, feat_zeros)
    #h_out_grad = h_out_grad / tl.where(col_count == 0, 1, col_count)
    # accumulate gradient to corresponding inputs
    for index in range(col_count):
        node_index_j = tl.load(adj_indices + col_start + index)
        edge_type_j = tl.load(rel_indices + col_start + index)
        
        neighbor_feat_j = tl.load(h_in + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        rel_feat_j = tl.load(rels + edge_type_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        
        tl.atomic_add(dh_src + node_index_j * IN_CHAN + feat_offsets, h_out_grad * rel_feat_j, feat_valid_mask)
        tl.atomic_add(drel_src + edge_type_j * IN_CHAN + feat_offsets, h_out_grad * neighbor_feat_j, feat_valid_mask)



@triton.jit
def hyper_relconv_aggregate_sum_kernel_forward(
      out_h,         # h_out output tensor of shape[num_nodes_dst,IN_CHAN] ; This stores places for the output 
      adj_rowptr,    # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1] ; This stores the destination row index
      adj_indices,   # graph adjacency matrix (indices) of shape[num_edges, max_arity - 1]; the "column index" count, store all the source nodes 
      h_src,         # h_src tensor of shape[num_nodes_src, IN_CHAN] ; This is the input x
      rel_indices,   # edge types of shape [num_edges]; relation type for each edge
      rels,          # relations matrix of shaoe [num_rels, IN_CHAN];  actual relation matrix. 
      pos_encodings, # position matrix of shape [max_arity, IN_CHAN]
      pos_index,     # position index of shape [num_edges, max_arity - 1], indicates for all the source nodes, which position is considered;
      MAX_ARITY,      # maximum arity for each node 
      NUM_EDGE,       # number of edges in total
      IN_CHAN        : tl.constexpr, # number of features per node we are considering in this thread 
      WG_SIZE        : tl.constexpr, # workgroup size
    ):
    # Pseudo-code (functional):
    #   for node_i in range(out_h.shape[0]):
    #     out_h[node_i,:] = 0
    #     col_start = adj_rowptr[node_i]
    #     col_count = adj_rowptr[node_i+1] – col_start
    #     for icol in range(col_count):
    #       rel_j = rel_indices[col_start + icol]
    #       for arity in pos_index:
    #           node_j = adj_indices[col_start + icol, arity]
    #           pos_j = tl.load(pos_encodings + pos_index[col_start + icol, arity])
    #           out_h[node_i,:] *= (h_src[node_j,:] + pos_j)
    #       out_h[node_i,:] *=  rels[rel_j, :]

    node_index_i = tl.program_id(0) # Get which node we are operating one
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE # tl.program_id(1) = which feature group we are working on. There might be multiple groups 
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    feat_ones = tl.full((WG_SIZE,),1, dtype=tl.float32)

    # identity list of node indices to aggregate
    col_start = tl.load(adj_rowptr + node_index_i) # To extract the row[node_index_i], we want to get the index of starting column and ending column
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start # The sliced indexes for the extracted row (indicated by the column index).
    # aggregate neighboring features
    aggr_sum = feat_zeros
    # Per edges
    for index in range(col_count):
        edge_type_j = tl.load(rel_indices + col_start + index) # same here, extract the correct [row, column] position for edges. 
        rel_feat_j = tl.load(rels + edge_type_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        # Per position, i.e. per nodes
        aggr_mul = feat_ones
        for place in range(MAX_ARITY-1):
            pos_index_j = pos_index + col_start + index + place * NUM_EDGE
            arity = tl.load(pos_index_j) # extract the correct arity for current node in this edge. 
            node_index_j = tl.load(adj_indices + col_start + index  + place * NUM_EDGE)
            if node_index_j != 0:
                # Need a mask here to see if node_index_j is 0 or not. If it is 0, which is padding node, we should stop the for loop
                pos_j = tl.load(pos_encodings +  IN_CHAN * arity  + feat_offsets, feat_valid_mask, feat_zeros)
                neighbor_feat_j = tl.load(h_src + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
                aggr_mul *= (neighbor_feat_j + pos_j) 
        aggr_mul *= rel_feat_j
        aggr_sum += aggr_mul 

    # store aggregator output
    tl.store(out_h + node_index_i * IN_CHAN + feat_offsets, aggr_sum, feat_valid_mask)

@triton.jit
def relconv_aggregate_mean_kernel_forward(
      out_h,         # h_out output tensor of shape[num_nodes_dst,IN_CHAN] ; This stores places for the output 
      adj_rowptr,    # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1] ; a cumulative count in CSR for each row. This is the pointer pointing the start of the memory
      adj_indices,   # graph adjacency matrix (indices) of shape[num_edges]; the "column index" count, position for each nodes
      h_src,         # h_src tensor of shape[num_nodes_src, IN_CHAN] ; This is the input x
      rel_indices,   # edge types of shape [num_edges]; relation type for each edge
      rels,          # relations matrix of shaoe [num_rels, IN_CHAN];  actual relation matrix. 
      IN_CHAN        : tl.constexpr, # number of features per node we are considering in this thread
      WG_SIZE        : tl.constexpr, # workgroup size
    ):
    # Pseudo-code (functional):
    #   for node_i in range(out_h.shape[0]):
    #     out_h[node_i,:] = 0
    #     col_start = adj_rowptr[node_i]
    #     col_count = adj_rowptr[node_i+1] – col_start
    #     for icol in range(col_count):
    #       node_j = adj_indices[col_start + icol]
    #       rel_j = rel_indices[col_start + icol]
    #       out_h[node_i,:] += h_src[node_j,:] * rels[rel_j, :] 

    # (Triton implementation for float32)
    # get current node_index and features to process
    node_index_i = tl.program_id(0) # Get which node we are operating one
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE # tl.program_id(1) = which feature group we are working on. There might be multiple groups 
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    # identity list of node indices to aggregate. Note these are sorted in the CSR format
    col_start = tl.load(adj_rowptr + node_index_i) # To extract the row[node_index_i], we want to get the index of starting column and ending column
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start # The sliced indexes for the extracted row (indicated by the column index).
    # aggregate neighboring features
    aggr_sum = feat_zeros
    for index in range(col_count):
        node_index_j = tl.load(adj_indices + col_start + index) # adj_indices points the stored total column index, then col_start is the start of the current considered row, index is the current edges in the considered row, which is what we want
        edge_type_j = tl.load(rel_indices + col_start + index) # same here, extract the correct [row, column] position for edges. 

        neighbor_feat_j = tl.load(h_src + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        rel_feat_j = tl.load(rels + edge_type_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        aggr_sum += rel_feat_j * neighbor_feat_j
    col_count = tl.where(col_count == 0, 1, col_count)
    aggr_mean = aggr_sum / col_count
    # store aggregator output
    tl.store(out_h + node_index_i * IN_CHAN + feat_offsets, aggr_mean, feat_valid_mask)

@triton.jit
def relconv_aggregate_mean_kernel_forward_v2u(
    out_h,         # h_out output tensor of shape[num_nodes_dst,IN_CHAN]
    adj_rowptr,    # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1]
    adj_indices,   # graph adjacency matrix (indices) of shape[num_edges]
    h_src,         # h_src tensor of shape[num_nodes_src, IN_CHAN]
    rel_indices,   # edge types of shape [num_edges]
    rels,          # relations matrix of shaoe [num_rels, IN_CHAN]
    IN_CHAN: tl.constexpr,  # number of features per head
    batch_size: tl.constexpr,  # Batch size
    n_nodes: tl.constexpr,
    WG_SIZE_NODES: tl.constexpr,  # workgroup size
    WG_SIZE_BATCH: tl.constexpr,  # workgroup size
    WG_SIZE_FEATS: tl.constexpr,  # workgroup size
):

    feat_offset = tl.arange(0, WG_SIZE_FEATS) + tl.program_id(0) * WG_SIZE_FEATS
    feat_offset_mask = feat_offset < IN_CHAN

    batch_offset = tl.arange(0, WG_SIZE_BATCH) + tl.program_id(1) * WG_SIZE_BATCH
    batch_offset_mask = batch_offset < batch_size

    batch_feat_mask = batch_offset_mask[:, None] and feat_offset_mask[None, :]
    batch_feat_offset = batch_offset[:, None] * IN_CHAN + feat_offset[None, :]

    row_start = tl.program_id(2) * WG_SIZE_NODES
    row_end = tl.minimum(row_start + WG_SIZE_NODES, n_nodes)

    for node_index in range(row_start, row_end):
        col_start = tl.load(adj_rowptr + node_index)
        col_end = tl.load(adj_rowptr + node_index + 1)
        col_count = col_end - col_start
        # aggregate neighboring features
        aggr_sum = tl.zeros((WG_SIZE_BATCH, WG_SIZE_FEATS), dtype=tl.float32)
        for index in range(col_count):
            node_index_j = tl.load(adj_indices + col_start + index)
            edge_type_j = tl.load(rel_indices + col_start + index)

            # Load the 2D rel feat vector. 
            rel_feat_j = tl.load(rels + edge_type_j * IN_CHAN * batch_size + batch_feat_offset, mask = batch_feat_mask)
            # Load a neighbor_feat_j for multiple minibatch elements. A 2D tensor
            neighbor_feat_j = tl.load(h_src + node_index_j * IN_CHAN * batch_size + batch_feat_offset,
                                      mask=batch_feat_mask)
            aggr_sum += rel_feat_j * neighbor_feat_j  # no broadcasting
        col_count = tl.where(col_count == 0, 1, col_count)
        aggr_mean = aggr_sum / col_count
        tl.store(out_h + node_index * batch_size * IN_CHAN +
                 batch_feat_offset, aggr_mean, mask=batch_feat_mask)
    
@triton.jit
def relconv_aggregate_mean_kernel_backward(
      dh_src,        # pre-initialized dh_src output tensor of shape[num_nodes_src,IN_CHAN]
      drel_src,      # pre-initialized drel output tensor of shape [num_relations, IN_CHAN]
      adj_rowptr,    # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1]
      adj_indices,   # graph adjacency matrix (indices) of shape[num_edges]
      h_in,          # original input features of shape [num_nodes, IN_CHAN]
      dh_out,        # MAIN GRADIENT: dh_out tensor of shape[num_nodes_src,IN_CHAN]
      rel_indices,   # edge types of shape [num_edges]
      rels,          # relations matrix of shape [num_rels, IN_CHAN]
      IN_CHAN        : tl.constexpr, # number of features per head
      WG_SIZE        : tl.constexpr, # workgroup size
    ):
    # Pseudo-code (functional):
    #   for node_i in range(dh_out.shape[0]):
    #     col_start = adj_rowptr[node_i]
    #     col_count = adj_rowptr[node_i+1] – col_start
    #     for icol in range(col_count):
    #       node_j = adj_indices[col_start + icol]
    #       rel_j = rel_indices[col_start + icol]
    #       nodej_feat = hin[node_j]
    #       relj_feat = rels[rel_j]
    #       dh_src[node_i,:] += dh * relj_feat
    #       drel_src[rel_j, :] += dh * nodej_feat

    # (Triton implementation for float32)
    # get current node_index and features to process
    node_index_i = tl.program_id(0)
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    # identify list of node indices to aggregate
    col_start = tl.load(adj_rowptr + node_index_i)
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start
    # load output gradient and scale
    h_out_grad = tl.load(dh_out + node_index_i * IN_CHAN + feat_offsets,
            feat_valid_mask, feat_zeros)
    h_out_grad = h_out_grad / tl.where(col_count == 0, 1, col_count)
    # accumulate gradient to corresponding inputs
    for index in range(col_count):
        node_index_j = tl.load(adj_indices + col_start + index)
        edge_type_j = tl.load(rel_indices + col_start + index)
        
        neighbor_feat_j = tl.load(h_in + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        rel_feat_j = tl.load(rels + edge_type_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)


        tl.atomic_add(dh_src + node_index_j * IN_CHAN + feat_offsets, (h_out_grad * rel_feat_j), feat_valid_mask)
        tl.atomic_add(drel_src + edge_type_j * IN_CHAN + feat_offsets, (h_out_grad * neighbor_feat_j), feat_valid_mask)



@triton.jit
def hyper_relconv_aggregate_mean_kernel_forward(
      out_h,         # h_out output tensor of shape[num_nodes_dst,IN_CHAN] ; This stores places for the output 
      adj_rowptr,    # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1] ; This stores the destination row index
      adj_indices,   # graph adjacency matrix (indices) of shape[num_edges, max_arity - 1]; the "column index" count, store all the source nodes 
      h_src,         # h_src tensor of shape[num_nodes_src, IN_CHAN] ; This is the input x
      rel_indices,   # edge types of shape [num_edges]; relation type for each edge
      rels,          # relations matrix of shaoe [num_rels, IN_CHAN];  actual relation matrix. 
      pos_encodings, # position matrix of shape [max_arity, IN_CHAN]
      pos_index,     # position index of shape [num_edges, max_arity - 1], indicates for all the source nodes, which position is considered;
      MAX_ARITY      : tl.constexpr, # maximum arity for each node 
      NUM_EDGE       : tl.constexpr, # number of edges in total
      IN_CHAN        : tl.constexpr, # number of features per node we are considering in this thread 
      WG_SIZE        : tl.constexpr, # workgroup size
    ):
    # Pseudo-code (functional):
    #   for node_i in range(out_h.shape[0]):
    #     out_h[node_i,:] = 0
    #     col_start = adj_rowptr[node_i]
    #     col_count = adj_rowptr[node_i+1] – col_start
    #     for icol in range(col_count):
    #       rel_j = rel_indices[col_start + icol]
    #       for arity in pos_index:
    #           node_j = adj_indices[col_start + icol, arity]
    #           pos_j = tl.load(pos_encodings + pos_index[col_start + icol, arity])
    #           out_h[node_i,:] *= (h_src[node_j,:] + pos_j)
    #       out_h[node_i,:] *=  rels[rel_j, :]

    node_index_i = tl.program_id(0) # Get which node we are operating one
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE # tl.program_id(1) = which feature group we are working on. There might be multiple groups 
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    feat_ones = tl.full((WG_SIZE,),1, dtype=tl.float32)

    # identity list of node indices to aggregate
    col_start = tl.load(adj_rowptr + node_index_i) # To extract the row[node_index_i], we want to get the index of starting column and ending column
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start # The sliced indexes for the extracted row (indicated by the column index).
    # aggregate neighboring features
    aggr_sum = feat_zeros
    # Per edges
    for index in range(col_count):
        edge_type_j = tl.load(rel_indices + col_start + index) # same here, extract the correct [row, column] position for edges. 
        rel_feat_j = tl.load(rels + edge_type_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        # Per position, i.e. per nodes
        aggr_mul = feat_ones
        for place in range(MAX_ARITY-1):
            pos_index_j = pos_index + col_start + index + place * NUM_EDGE
            arity = tl.load(pos_index_j) # extract the correct arity for current node in this edge. 
            node_index_j = tl.load(adj_indices + col_start + index  + place * NUM_EDGE)
            if node_index_j != 0:
                # Need a mask here to see if node_index_j is 0 or not. If it is 0, which is padding node, we should stop the for loop
                pos_j = tl.load(pos_encodings +  IN_CHAN * arity  + feat_offsets, feat_valid_mask, feat_zeros)
                neighbor_feat_j = tl.load(h_src + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
                aggr_mul *= (neighbor_feat_j + pos_j) 
        aggr_mul *= rel_feat_j
        aggr_sum += aggr_mul 
    col_count = tl.where(col_count == 0, 1, col_count)
    aggr_mean = aggr_sum / col_count
        

    # store aggregator output
    tl.store(out_h + node_index_i * IN_CHAN + feat_offsets, aggr_mean, feat_valid_mask)

@triton.jit
def multiply(a, b):
    return a * b

@triton.jit
def hyper_relconv_aggregate_sum_kernel_v2u_forward(
      out_h,         # h_out output tensor of shape[num_nodes_dst,IN_CHAN] ; This stores places for the output 
      adj_rowptr,    # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1] ; This stores the destination row index
      adj_indices,   # graph adjacency matrix (indices) of shape[num_edges, max_arity - 1]; the "column index" count, store all the source nodes 
      h_src,         # h_src tensor of shape[num_nodes_src, IN_CHAN] ; This is the input x
      rel_indices,   # edge types of shape [num_edges]; relation type for each edge
      rels,          # relations matrix of shaoe [num_rels, IN_CHAN];  actual relation matrix. 
      pos_encodings, # position matrix of shape [max_arity, IN_CHAN]
      pos_index,     # position index of shape [num_edges, max_arity - 1], indicates for all the source nodes, which position is considered;
      MAX_ARITY      : tl.constexpr, # maximum arity for each node 
      NUM_EDGE       : tl.constexpr, # number of edges in total
      IN_CHAN        : tl.constexpr, # number of features per node we are considering in this thread 
      WG_SIZE        : tl.constexpr, # workgroup size
      ARITY_SIZE     : tl.constexpr, # maximum arity size
    ):
    # Pseudo-code (functional):
    #   for node_i in range(out_h.shape[0]):
    #     out_h[node_i,:] = 0
    #     col_start = adj_rowptr[node_i]
    #     col_count = adj_rowptr[node_i+1] – col_start
    #     range_pos = tl.arange(0, MAX_ARITY - 1) * NUM_EDGE
    #     for icol in range(col_count):
    #       rel_j = rel_indices[col_start + icol]
    #       pos_index_array = pos_index + col_start + index + range_pos
    #       arity = tl.load(pos_index_array, 0)
    #       node_index_j = tl.load(adj_indices + col_start + index  + range_pos)
    #       non_zero_mask = node_index_j != 0
    #       pos_j = tl.load(pos_encodings +  IN_CHAN * arity  + feat_offsets, feat_valid_mask, feat_ones)
    #       neighbor_feat_j = tl.load(h_src + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask & non_zero_mask, feat_zeros)
    #       out_h[node_i,:] *= tl.reduce(neighbor_feat_j + pos_j, multiply)
    #       out_h[node_i,:] *=  rels[rel_j, :]

    node_index_i = tl.program_id(0) # Get which node we are operating one
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE # tl.program_id(1) = which feature group we are working on. There might be multiple groups 
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    feat_ones = tl.full((WG_SIZE,),1, dtype=tl.float32)

    feat_offsets_array = tl.broadcast_to(tl.expand_dims(feat_offsets, 0), (ARITY_SIZE, WG_SIZE))
    dummy_arity_mask = tl.broadcast_to(tl.expand_dims(tl.arange(0, ARITY_SIZE), 1), (ARITY_SIZE, WG_SIZE)) < MAX_ARITY - 1 # mask for the maximum arity; 2D array that assigns all the row greater than MAX_ARITY - 1 to 0
    feat_valid_mask_array = (feat_offsets_array < IN_CHAN) and dummy_arity_mask # By default the feat_offsets_array < IN_CHAN cut off the columns laters, and by land we cut off the rows that are greater than MAX_ARITY - 1
    feat_zeros_array = tl.zeros((ARITY_SIZE, WG_SIZE), dtype=tl.float32)
    feat_ones_array = tl.full((ARITY_SIZE, WG_SIZE),1, dtype=tl.float32)

    # identity list of node indices to aggregate
    col_start = tl.load(adj_rowptr + node_index_i) # To extract the row[node_index_i], we want to get the index of starting column and ending column
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start # The sliced indexes for the extracted row (indicated by the column index).

    mask_for_arity = tl.arange(0, ARITY_SIZE) < MAX_ARITY - 1
    range_pos = tl.arange(0, ARITY_SIZE) * NUM_EDGE
    feat_zeros_arity = tl.zeros((ARITY_SIZE,), dtype=tl.float32)

    # aggregate neighboring features
    aggr_sum = feat_zeros
    # Per edges
    for index in range(col_count):
        edge_type_j = tl.load(rel_indices + col_start + index) # same here, extract the correct [row, column] position for edges. 
        rel_feat_j = tl.load(rels + edge_type_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        # Per position, i.e. per nodes
        aggr_mul = feat_ones
        
        pos_index_array = pos_index + col_start + index + range_pos
        arity = tl.load(pos_index_array, mask_for_arity, feat_zeros_arity)
        node_index_j = tl.load(adj_indices + col_start + index  + range_pos, mask_for_arity, feat_zeros_arity)
        node_index_j_reshape = tl.broadcast_to(tl.expand_dims(node_index_j, 1), (ARITY_SIZE, WG_SIZE))
        non_zero_mask = node_index_j_reshape != 0
        
        arity_reshape = tl.broadcast_to(tl.expand_dims(arity, 1), (ARITY_SIZE, WG_SIZE))

        pos_j = tl.load(pos_encodings + IN_CHAN * arity_reshape  + feat_offsets_array , feat_valid_mask_array & non_zero_mask, feat_ones_array)
        neighbor_feat_j = tl.load(h_src + node_index_j_reshape * IN_CHAN + feat_offsets_array, feat_valid_mask_array & non_zero_mask, feat_zeros_array)

        aggr_mul *= tl.reduce(neighbor_feat_j + pos_j, axis=0, combine_fn=multiply) # Take the last element of the cumprod

        aggr_mul *= rel_feat_j 
        aggr_sum += aggr_mul 

    # store aggregator output
    tl.store(out_h + node_index_i * IN_CHAN + feat_offsets, aggr_sum, feat_valid_mask)
    

@triton.jit
def hyper_relconv_aggregate_sum_kernel_backward(
      dh_src,        # pre-initialized dh_src output tensor of shape[num_nodes_src,IN_CHAN]
      drel_src,      # pre-initialized drel output tensor of shape [num_relations, IN_CHAN]
      adj_rowptr,    # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1]
      adj_indices,   # graph adjacency matrix (indices) of shape[num_edges, max_arity - 1]
      h_in,          # original input features of shape [num_nodes, IN_CHAN]
      dh_out,        # MAIN GRADIENT: dh_out tensor of shape[num_nodes_src,IN_CHAN]
      rel_indices,   # edge types of shape [num_edges]
      rels,          # relations matrix of shape [num_rels, IN_CHAN]
      pos_encodings, # position matrix of shape [max_arity, IN_CHAN]
      pos_index,     # position index of shape [num_edges, max_arity - 1], indicates for all the source nodes, which position is considered;
      MAX_ARITY, # maximum arity for each node 
      NUM_EDGE, # number of edges in total
      IN_CHAN        : tl.constexpr, # number of features per head
      WG_SIZE        : tl.constexpr, # workgroup size
    ):
    # Pseudo-code (functional):
    #   for node_i in range(out_h.shape[0]):
    #     out_h[node_i,:] = 0
    #     col_start = adj_rowptr[node_i]
    #     col_count = adj_rowptr[node_i+1] – col_start
    #     for icol in range(col_count):
    #       aggr_mul = 1
    #       rel_j = rel_indices[col_start + icol]
    #       for arity in pos_index:
    #           node_j = adj_indices[col_start + icol, arity]
    #           pos_j = tl.load(pos_encodings + pos_index[col_start + icol, arity])
    #           aggr_mul *=  (h_src[node_j,:] + pos_j) 
    #       for arity in pos_index:
    #           node_j = adj_indices[col_start + icol, arity]
    #           pos_j = tl.load(pos_encodings + pos_index[col_start + icol, arity])
    #           out_h[node_j,:] = aggr_mul / (h_src[node_j,:] + pos_j) # 
    #       rel_j[,:] *=  rels[rel_j, :] * aggr_mul
    node_index_i = tl.program_id(0) # Get which node we are operating one
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE #
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    feat_ones = tl.full((WG_SIZE,),1, dtype=tl.float32)

    # identity list of node indices to aggregate
    col_start = tl.load(adj_rowptr + node_index_i) # To extract the row[node_index_i], we want to get the index of starting column and ending column
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start # The sliced indexes for the extracted row (indicated by the column index).
    
    # load output gradient and scale
    h_out_grad = tl.load(dh_out + node_index_i * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
    # Per edges
    for index in range(col_count):
        edge_type_j = tl.load(rel_indices + col_start + index) # same here, extract the correct [row, column] position for edges. 
        rel_feat_j = tl.load(rels + edge_type_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        # Per position, i.e. per nodes
        aggr_mul = feat_ones
        for place in range(MAX_ARITY-1):
            pos_index_j = pos_index + col_start + index + place * NUM_EDGE
            arity = tl.load(pos_index_j) # extract the correct arity for current node in this edge. 
            node_index_j = tl.load(adj_indices + col_start + index  + place * NUM_EDGE)
            if node_index_j != 0:
                # Need a mask here to see if node_index_j is 0 or not. If it is 0, which is padding node, we should stop the for loop
                pos_j = tl.load(pos_encodings +  IN_CHAN * arity  + feat_offsets, feat_valid_mask, feat_ones)
                neighbor_feat_j = tl.load(h_in + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
                aggr_mul *= (neighbor_feat_j + pos_j) 
        # Compute everything again excepts one. A dummy way of doing this
        for place in range(MAX_ARITY - 1):
            pos_index_j = pos_index + col_start + index + place * NUM_EDGE
            arity = tl.load(pos_index_j) 
            node_index_j = tl.load(adj_indices + col_start + index  + place * NUM_EDGE)
            if node_index_j != 0:
                pos_j = tl.load(pos_encodings +  IN_CHAN * arity  + feat_offsets, feat_valid_mask, feat_ones)
                neighbor_feat_j = tl.load(h_in + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
                rest_j = aggr_mul / (neighbor_feat_j + pos_j + 1e-7) # Avoid division by zero. TODO: Need further improvement
                tl.atomic_add(dh_src + node_index_j * IN_CHAN + feat_offsets, h_out_grad * rel_feat_j * rest_j, feat_valid_mask)
        # Here, we store the rel_feature_j for this edge, which is the agg_mul now. 
        tl.atomic_add(drel_src + edge_type_j * IN_CHAN + feat_offsets, h_out_grad * aggr_mul, feat_valid_mask)


@triton.jit
def hyper_relconv_aggregate_sum_kernel_v2u_backward(
      dh_src,        # pre-initialized dh_src output tensor of shape[num_nodes_src,IN_CHAN]
      drel_src,      # pre-initialized drel output tensor of shape [num_relations, IN_CHAN]
      adj_rowptr,    # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1]
      adj_indices,   # graph adjacency matrix (indices) of shape[num_edges, max_arity - 1]
      h_in,          # original input features of shape [num_nodes, IN_CHAN]
      dh_out,        # MAIN GRADIENT: dh_out tensor of shape[num_nodes_src,IN_CHAN]
      rel_indices,   # edge types of shape [num_edges]
      rels,          # relations matrix of shape [num_rels, IN_CHAN]
      pos_encodings, # position matrix of shape [max_arity, IN_CHAN]
      pos_index,     # position index of shape [num_edges, max_arity - 1], indicates for all the source nodes, which position is considered;
      MAX_ARITY      : tl.constexpr, # maximum arity for each node 
      NUM_EDGE       : tl.constexpr, # number of edges in total
      IN_CHAN        : tl.constexpr, # number of features per head
      WG_SIZE        : tl.constexpr, # workgroup size
      ARITY_SIZE     : tl.constexpr, # maximum arity size
    ):
    # Pseudo-code (functional):
    #   for node_i in range(out_h.shape[0]):
    #     out_h[node_i,:] = 0
    #     col_start = adj_rowptr[node_i]
    #     col_count = adj_rowptr[node_i+1] – col_start
    #     for icol in range(col_count):
    #       aggr_mul = 1
    #       rel_j = rel_indices[col_start + icol]
    #       for arity in pos_index:
    #           node_j = adj_indices[col_start + icol, arity]
    #           pos_j = tl.load(pos_encodings + pos_index[col_start + icol, arity])
    #           aggr_mul *=  (h_src[node_j,:] + pos_j) 
    #       for arity in pos_index:
    #           node_j = adj_indices[col_start + icol, arity]
    #           pos_j = tl.load(pos_encodings + pos_index[col_start + icol, arity])
    #           out_h[node_j,:] = aggr_mul / (h_src[node_j,:] + pos_j) # 
    #       rel_j[,:] *=  rels[rel_j, :] * aggr_mul
    node_index_i = tl.program_id(0) # Get which node we are operating one
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE #
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    feat_ones = tl.full((WG_SIZE,),1, dtype=tl.float32)
    
    feat_offsets_array = tl.broadcast_to(tl.expand_dims(feat_offsets, 0), (ARITY_SIZE, WG_SIZE))
    dummy_arity_mask = tl.broadcast_to(tl.expand_dims(tl.arange(0, ARITY_SIZE), 1), (ARITY_SIZE, WG_SIZE)) < MAX_ARITY - 1 # mask for the maximum arity; 2D array that assigns all the row greater than MAX_ARITY - 1 to 0
    feat_valid_mask_array = (feat_offsets_array < IN_CHAN) and dummy_arity_mask # By default the feat_offsets_array < IN_CHAN cut off the columns laters, and by land we cut off the rows that are greater than MAX_ARITY - 1
    feat_zeros_array = tl.zeros((ARITY_SIZE, WG_SIZE), dtype=tl.float32)
    feat_ones_array = tl.full((ARITY_SIZE, WG_SIZE),1, dtype=tl.float32)

    # identity list of node indices to aggregate
    col_start = tl.load(adj_rowptr + node_index_i) # To extract the row[node_index_i], we want to get the index of starting column and ending column
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start # The sliced indexes for the extracted row (indicated by the column index).
    
    mask_for_arity = tl.arange(0, ARITY_SIZE) < MAX_ARITY - 1
    range_pos = tl.arange(0, ARITY_SIZE) * NUM_EDGE
    feat_zeros_arity = tl.zeros((ARITY_SIZE,), dtype=tl.float32)

    # load output gradient and scale
    h_out_grad = tl.load(dh_out + node_index_i * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
    h_out_grad_reshape = tl.broadcast_to(tl.expand_dims(h_out_grad, 0), (ARITY_SIZE, WG_SIZE))
    # Per edges
    for index in range(col_count):
        edge_type_j = tl.load(rel_indices + col_start + index) # same here, extract the correct [row, column] position for edges. 
        rel_feat_j = tl.load(rels + edge_type_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        rel_feat_j_reshape = tl.broadcast_to(tl.expand_dims(rel_feat_j, 0), (ARITY_SIZE, WG_SIZE))
        # Per position, i.e. per nodes
        aggr_mul = feat_ones

        
        pos_index_array = pos_index + col_start + index + range_pos
        arity = tl.load(pos_index_array, mask_for_arity, feat_zeros_arity)
        node_index_j = tl.load(adj_indices + col_start + index  + range_pos, mask_for_arity, feat_zeros_arity)
        node_index_j_reshape = tl.broadcast_to(tl.expand_dims(node_index_j, 1), (ARITY_SIZE, WG_SIZE))
        non_zero_mask = node_index_j_reshape != 0
        
        arity_reshape = tl.broadcast_to(tl.expand_dims(arity, 1), (ARITY_SIZE, WG_SIZE))

        pos_j = tl.load(pos_encodings + IN_CHAN * arity_reshape  + feat_offsets_array , feat_valid_mask_array & non_zero_mask, feat_ones_array)
        neighbor_feat_j = tl.load(h_in + node_index_j_reshape * IN_CHAN + feat_offsets_array, feat_valid_mask_array & non_zero_mask, feat_zeros_array)

        pos_add_neighbor_j = neighbor_feat_j + pos_j
        left_prod = tl.cumprod(pos_add_neighbor_j, axis=0)
        right_prod = tl.cumprod(pos_add_neighbor_j, axis=0, reverse=True)
        # right_prod = tl.cumprod(pos_add_neighbor_j.flip(axis = 0), axis = 0)
        # right_prod = tl.cumprod(tl.flip(pos_add_neighbor_j, axis = 0), axis = 0)
        rest_j = left_prod * right_prod
        tl.atomic_add(dh_src + node_index_j_reshape * IN_CHAN + feat_offsets, h_out_grad_reshape * rel_feat_j_reshape * rest_j, feat_valid_mask_array)
        tl.atomic_add(drel_src + edge_type_j * IN_CHAN + feat_offsets, h_out_grad * aggr_mul, feat_valid_mask)

        # aggr_mul *= tl.reduce(neighbor_feat_j + pos_j, axis=0, combine_fn=multiply) # Take the last element of the cumprod
        #tl.atomic_add(dh_src + node_index_j * IN_CHAN + feat_offsets, h_out_grad * rel_feat_j * rest_j, feat_valid_mask)
        # for place in range(MAX_ARITY-1):
        #     pos_index_j = pos_index + col_start + index + place * NUM_EDGE
        #     arity = tl.load(pos_index_j) # extract the correct arity for current node in this edge. 
        #     node_index_j = tl.load(adj_indices + col_start + index  + place * NUM_EDGE)
        #     if node_index_j != 0:
        #         # Need a mask here to see if node_index_j is 0 or not. If it is 0, which is padding node, we should stop the for loop
        #         pos_j = tl.load(pos_encodings +  IN_CHAN * arity  + feat_offsets, feat_valid_mask, feat_ones)
        #         neighbor_feat_j = tl.load(h_in + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        #         aggr_mul *= (neighbor_feat_j + pos_j) 
        # # Compute everything again excepts one. A dummy way of doing this
        # for place in range(MAX_ARITY - 1):
        #     pos_index_j = pos_index + col_start + index + place * NUM_EDGE
        #     arity = tl.load(pos_index_j) 
        #     node_index_j = tl.load(adj_indices + col_start + index  + place * NUM_EDGE)
        #     if node_index_j != 0:
        #         pos_j = tl.load(pos_encodings +  IN_CHAN * arity  + feat_offsets, feat_valid_mask, feat_ones)
        #         neighbor_feat_j = tl.load(h_in + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        #         rest_j = aggr_mul / (neighbor_feat_j + pos_j + 1e-7) # Avoid division by zero. TODO: Need further improvement
        #         tl.atomic_add(dh_src + node_index_j * IN_CHAN + feat_offsets, h_out_grad * rel_feat_j * rest_j, feat_valid_mask)
        # # Here, we store the rel_feature_j for this edge, which is the agg_mul now. 
        
@triton.jit
def hyper_relconv_aggregate_mean_kernel_v2u_forward(
      out_h,         # h_out output tensor of shape[num_nodes_dst,IN_CHAN] ; This stores places for the output 
      adj_rowptr,    # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1] ; This stores the destination row index
      adj_indices,   # graph adjacency matrix (indices) of shape[num_edges, max_arity - 1]; the "column index" count, store all the source nodes 
      h_src,         # h_src tensor of shape[num_nodes_src, IN_CHAN] ; This is the input x
      rel_indices,   # edge types of shape [num_edges]; relation type for each edge
      rels,          # relations matrix of shaoe [num_rels, IN_CHAN];  actual relation matrix. 
      pos_encodings, # position matrix of shape [max_arity, IN_CHAN]
      pos_index,     # position index of shape [num_edges, max_arity - 1], indicates for all the source nodes, which position is considered;
      MAX_ARITY      : tl.constexpr, # maximum arity for each node 
      NUM_EDGE       : tl.constexpr, # number of edges in total
      IN_CHAN        : tl.constexpr, # number of features per node we are considering in this thread 
      WG_SIZE        : tl.constexpr, # workgroup size
      ARITY_SIZE     : tl.constexpr, # maximum arity size
    ):
    # Pseudo-code (functional):
    #   for node_i in range(out_h.shape[0]):
    #     out_h[node_i,:] = 0
    #     col_start = adj_rowptr[node_i]
    #     col_count = adj_rowptr[node_i+1] – col_start
    #     range_pos = tl.arange(0, MAX_ARITY - 1) * NUM_EDGE
    #     for icol in range(col_count):
    #       rel_j = rel_indices[col_start + icol]
    #       pos_index_array = pos_index + col_start + index + range_pos
    #       arity = tl.load(pos_index_array, 0)
    #       node_index_j = tl.load(adj_indices + col_start + index  + range_pos)
    #       non_zero_mask = node_index_j != 0
    #       pos_j = tl.load(pos_encodings +  IN_CHAN * arity  + feat_offsets, feat_valid_mask, feat_ones)
    #       neighbor_feat_j = tl.load(h_src + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask & non_zero_mask, feat_zeros)
    #       out_h[node_i,:] *= tl.reduce(neighbor_feat_j + pos_j, multiply)
    #       out_h[node_i,:] *=  rels[rel_j, :]

    node_index_i = tl.program_id(0) # Get which node we are operating one
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE # tl.program_id(1) = which feature group we are working on. There might be multiple groups 
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    feat_ones = tl.full((WG_SIZE,),1, dtype=tl.float32)

    feat_offsets_array = tl.broadcast_to(tl.expand_dims(feat_offsets, 0), (ARITY_SIZE, WG_SIZE))
    dummy_arity_mask = tl.broadcast_to(tl.expand_dims(tl.arange(0, ARITY_SIZE), 1), (ARITY_SIZE, WG_SIZE)) < MAX_ARITY - 1 # mask for the maximum arity; 2D array that assigns all the row greater than MAX_ARITY - 1 to 0
    feat_valid_mask_array = (feat_offsets_array < IN_CHAN) and dummy_arity_mask # By default the feat_offsets_array < IN_CHAN cut off the columns laters, and by land we cut off the rows that are greater than MAX_ARITY - 1
    feat_zeros_array = tl.zeros((ARITY_SIZE, WG_SIZE), dtype=tl.float32)
    feat_ones_array = tl.full((ARITY_SIZE, WG_SIZE),1, dtype=tl.float32)

    # identity list of node indices to aggregate
    col_start = tl.load(adj_rowptr + node_index_i) # To extract the row[node_index_i], we want to get the index of starting column and ending column
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start # The sliced indexes for the extracted row (indicated by the column index).

    mask_for_arity = tl.arange(0, ARITY_SIZE) < MAX_ARITY - 1
    range_pos = tl.arange(0, ARITY_SIZE) * NUM_EDGE
    feat_zeros_arity = tl.zeros((ARITY_SIZE,), dtype=tl.float32)

    # aggregate neighboring features
    aggr_sum = feat_zeros
    # Per edges
    for index in range(col_count):
        edge_type_j = tl.load(rel_indices + col_start + index) # same here, extract the correct [row, column] position for edges. 
        rel_feat_j = tl.load(rels + edge_type_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        # Per position, i.e. per nodes
        aggr_mul = feat_ones
        
        pos_index_array = pos_index + col_start + index + range_pos
        arity = tl.load(pos_index_array, mask_for_arity, feat_zeros_arity)
        node_index_j = tl.load(adj_indices + col_start + index  + range_pos, mask_for_arity, feat_zeros_arity)
        node_index_j_reshape = tl.broadcast_to(tl.expand_dims(node_index_j, 1), (ARITY_SIZE, WG_SIZE))
        non_zero_mask = node_index_j_reshape != 0
        
        arity_reshape = tl.broadcast_to(tl.expand_dims(arity, 1), (ARITY_SIZE, WG_SIZE))

        pos_j = tl.load(pos_encodings + IN_CHAN * arity_reshape  + feat_offsets_array , feat_valid_mask_array & non_zero_mask, feat_ones_array)
        neighbor_feat_j = tl.load(h_src + node_index_j_reshape * IN_CHAN + feat_offsets_array, feat_valid_mask_array & non_zero_mask, feat_zeros_array)

        aggr_mul *= tl.reduce(neighbor_feat_j + pos_j, axis=0, combine_fn=multiply) # Take the last element of the cumprod
        
        aggr_mul *= rel_feat_j 
        aggr_sum += aggr_mul 
    col_count = tl.where(col_count == 0, 1, col_count)
    aggr_mean = aggr_sum / col_count
    # store aggregator output
    tl.store(out_h + node_index_i * IN_CHAN + feat_offsets, aggr_mean, feat_valid_mask)
    

@triton.jit
def hyper_relconv_aggregate_mean_kernel_backward(
      dh_src,        # pre-initialized dh_src output tensor of shape[num_nodes_src,IN_CHAN]
      drel_src,      # pre-initialized drel output tensor of shape [num_relations, IN_CHAN]
      adj_rowptr,    # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1]
      adj_indices,   # graph adjacency matrix (indices) of shape[num_edges, max_arity - 1]
      h_in,          # original input features of shape [num_nodes, IN_CHAN]
      dh_out,        # MAIN GRADIENT: dh_out tensor of shape[num_nodes_src,IN_CHAN]
      rel_indices,   # edge types of shape [num_edges]
      rels,          # relations matrix of shape [num_rels, IN_CHAN]
      pos_encodings, # position matrix of shape [max_arity, IN_CHAN]
      pos_index,     # position index of shape [num_edges, max_arity - 1], indicates for all the source nodes, which position is considered;
      MAX_ARITY      : tl.constexpr, # maximum arity for each node 
      NUM_EDGE       : tl.constexpr, # number of edges in total
      IN_CHAN        : tl.constexpr, # number of features per head
      WG_SIZE        : tl.constexpr, # workgroup size
    ):
    # Pseudo-code (functional):
    #   for node_i in range(out_h.shape[0]):
    #     out_h[node_i,:] = 0
    #     col_start = adj_rowptr[node_i]
    #     col_count = adj_rowptr[node_i+1] – col_start
    #     for icol in range(col_count):
    #       aggr_mul = 1
    #       rel_j = rel_indices[col_start + icol]
    #       for arity in pos_index:
    #           node_j = adj_indices[col_start + icol, arity]
    #           pos_j = tl.load(pos_encodings + pos_index[col_start + icol, arity])
    #           aggr_mul *=  (h_src[node_j,:] + pos_j) 
    #       for arity in pos_index:
    #           node_j = adj_indices[col_start + icol, arity]
    #           pos_j = tl.load(pos_encodings + pos_index[col_start + icol, arity])
    #           out_h[node_j,:] = aggr_mul / (h_src[node_j,:] + pos_j) # 
    #       rel_j[,:] *=  rels[rel_j, :] * aggr_mul
    node_index_i = tl.program_id(0) # Get which node we are operating one
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE #
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    feat_ones = tl.full((WG_SIZE,),1, dtype=tl.float32)

    # identity list of node indices to aggregate
    col_start = tl.load(adj_rowptr + node_index_i) # To extract the row[node_index_i], we want to get the index of starting column and ending column
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start # The sliced indexes for the extracted row (indicated by the column index).
    
    # load output gradient and scale
    h_out_grad = tl.load(dh_out + node_index_i * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
    h_out_grad = h_out_grad / tl.where(col_count == 0, 1, col_count)
    # Per edges
    for index in range(col_count):
        edge_type_j = tl.load(rel_indices + col_start + index) # same here, extract the correct [row, column] position for edges. 
        rel_feat_j = tl.load(rels + edge_type_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        # Per position, i.e. per nodes
        aggr_mul = feat_ones
        for place in range(MAX_ARITY-1):
            pos_index_j = pos_index + col_start + index + place * NUM_EDGE
            arity = tl.load(pos_index_j) # extract the correct arity for current node in this edge. 
            node_index_j = tl.load(adj_indices + col_start + index  + place * NUM_EDGE)
            if node_index_j != 0:
                # Need a mask here to see if node_index_j is 0 or not. If it is 0, which is padding node, we should stop the for loop
                pos_j = tl.load(pos_encodings +  IN_CHAN * arity  + feat_offsets, feat_valid_mask, feat_ones)
                neighbor_feat_j = tl.load(h_in + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
                aggr_mul *= (neighbor_feat_j + pos_j) 
        # Compute everything again excepts one. A dummy way of doing this
        for place in range(MAX_ARITY - 1):
            pos_index_j = pos_index + col_start + index + place * NUM_EDGE
            arity = tl.load(pos_index_j) 
            node_index_j = tl.load(adj_indices + col_start + index  + place * NUM_EDGE)
            if node_index_j != 0:
                pos_j = tl.load(pos_encodings +  IN_CHAN * arity  + feat_offsets, feat_valid_mask, feat_ones)
                neighbor_feat_j = tl.load(h_in + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
                rest_j = aggr_mul / (neighbor_feat_j + pos_j + 1e-7) # Avoid division by zero. TODO: Need further improvement
                tl.atomic_add(dh_src + node_index_j * IN_CHAN + feat_offsets, h_out_grad * rel_feat_j * rest_j, feat_valid_mask)
        # Here, we store the rel_feature_j for this edge, which is the agg_mul now. 
        tl.atomic_add(drel_src + edge_type_j * IN_CHAN + feat_offsets, h_out_grad * aggr_mul, feat_valid_mask)


@triton.jit
def hyper_relconv_aggregate_mean_kernel_v2u_backward(
      dh_src,        # pre-initialized dh_src output tensor of shape[num_nodes_src,IN_CHAN]
      drel_src,      # pre-initialized drel output tensor of shape [num_relations, IN_CHAN]
      adj_rowptr,    # graph adjacency matrix (rowptr) of shape[num_nodes_dst+1]
      adj_indices,   # graph adjacency matrix (indices) of shape[num_edges, max_arity - 1]
      h_in,          # original input features of shape [num_nodes, IN_CHAN]
      dh_out,        # MAIN GRADIENT: dh_out tensor of shape[num_nodes_src,IN_CHAN]
      rel_indices,   # edge types of shape [num_edges]
      rels,          # relations matrix of shape [num_rels, IN_CHAN]
      pos_encodings, # position matrix of shape [max_arity, IN_CHAN]
      pos_index,     # position index of shape [num_edges, max_arity - 1], indicates for all the source nodes, which position is considered;
      MAX_ARITY      : tl.constexpr, # maximum arity for each node 
      NUM_EDGE       : tl.constexpr, # number of edges in total
      IN_CHAN        : tl.constexpr, # number of features per head
      WG_SIZE        : tl.constexpr, # workgroup size
      ARITY_SIZE     : tl.constexpr, # maximum arity size
    ):
    # Pseudo-code (functional):
    #   for node_i in range(out_h.shape[0]):
    #     out_h[node_i,:] = 0
    #     col_start = adj_rowptr[node_i]
    #     col_count = adj_rowptr[node_i+1] – col_start
    #     for icol in range(col_count):
    #       aggr_mul = 1
    #       rel_j = rel_indices[col_start + icol]
    #       for arity in pos_index:
    #           node_j = adj_indices[col_start + icol, arity]
    #           pos_j = tl.load(pos_encodings + pos_index[col_start + icol, arity])
    #           aggr_mul *=  (h_src[node_j,:] + pos_j) 
    #       for arity in pos_index:
    #           node_j = adj_indices[col_start + icol, arity]
    #           pos_j = tl.load(pos_encodings + pos_index[col_start + icol, arity])
    #           out_h[node_j,:] = aggr_mul / (h_src[node_j,:] + pos_j) # 
    #       rel_j[,:] *=  rels[rel_j, :] * aggr_mul
    node_index_i = tl.program_id(0) # Get which node we are operating one
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE #
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    feat_ones = tl.full((WG_SIZE,),1, dtype=tl.float32)
    
    feat_offsets_array = tl.broadcast_to(tl.expand_dims(feat_offsets, 0), (ARITY_SIZE, WG_SIZE))
    dummy_arity_mask = tl.broadcast_to(tl.expand_dims(tl.arange(0, ARITY_SIZE), 1), (ARITY_SIZE, WG_SIZE)) < MAX_ARITY - 1 # mask for the maximum arity; 2D array that assigns all the row greater than MAX_ARITY - 1 to 0
    feat_valid_mask_array = (feat_offsets_array < IN_CHAN) and dummy_arity_mask # By default the feat_offsets_array < IN_CHAN cut off the columns laters, and by land we cut off the rows that are greater than MAX_ARITY - 1
    feat_zeros_array = tl.zeros((ARITY_SIZE, WG_SIZE), dtype=tl.float32)
    feat_ones_array = tl.full((ARITY_SIZE, WG_SIZE),1, dtype=tl.float32)

    # identity list of node indices to aggregate
    col_start = tl.load(adj_rowptr + node_index_i) # To extract the row[node_index_i], we want to get the index of starting column and ending column
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start # The sliced indexes for the extracted row (indicated by the column index).
    
    mask_for_arity = tl.arange(0, ARITY_SIZE) < MAX_ARITY - 1
    range_pos = tl.arange(0, ARITY_SIZE) * NUM_EDGE
    feat_zeros_arity = tl.zeros((ARITY_SIZE,), dtype=tl.float32)

    # load output gradient and scale
    h_out_grad = tl.load(dh_out + node_index_i * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
    h_out_grad = h_out_grad / tl.where(col_count == 0, 1, col_count)
    h_out_grad_reshape = tl.broadcast_to(tl.expand_dims(h_out_grad, 0), (ARITY_SIZE, WG_SIZE))
    # Per edges
    for index in range(col_count):
        edge_type_j = tl.load(rel_indices + col_start + index) # same here, extract the correct [row, column] position for edges. 
        rel_feat_j = tl.load(rels + edge_type_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        rel_feat_j_reshape = tl.broadcast_to(tl.expand_dims(rel_feat_j, 0), (ARITY_SIZE, WG_SIZE))
        # Per position, i.e. per nodes
        aggr_mul = feat_ones

        
        pos_index_array = pos_index + col_start + index + range_pos
        arity = tl.load(pos_index_array, mask_for_arity, feat_zeros_arity)
        node_index_j = tl.load(adj_indices + col_start + index  + range_pos, mask_for_arity, feat_zeros_arity)
        node_index_j_reshape = tl.broadcast_to(tl.expand_dims(node_index_j, 1), (ARITY_SIZE, WG_SIZE))
        non_zero_mask = node_index_j_reshape != 0
        
        arity_reshape = tl.broadcast_to(tl.expand_dims(arity, 1), (ARITY_SIZE, WG_SIZE))

        pos_j = tl.load(pos_encodings + IN_CHAN * arity_reshape  + feat_offsets_array , feat_valid_mask_array & non_zero_mask, feat_ones_array)
        neighbor_feat_j = tl.load(h_in + node_index_j_reshape * IN_CHAN + feat_offsets_array, feat_valid_mask_array & non_zero_mask, feat_zeros_array)

        pos_add_neighbor_j = neighbor_feat_j + pos_j
        left_prod = tl.cumprod(pos_add_neighbor_j, axis=0)
        right_prod = tl.cumprod(pos_add_neighbor_j, axis=0, reverse=True)
        # right_prod = tl.cumprod(pos_add_neighbor_j.flip(axis = 0), axis = 0)
        # right_prod = tl.cumprod(tl.flip(pos_add_neighbor_j, axis = 0), axis = 0)
        rest_j = left_prod * right_prod
        tl.atomic_add(dh_src + node_index_j_reshape * IN_CHAN + feat_offsets, h_out_grad_reshape * rel_feat_j_reshape * rest_j, feat_valid_mask_array)
        tl.atomic_add(drel_src + edge_type_j * IN_CHAN + feat_offsets, h_out_grad * aggr_mul, feat_valid_mask)

        # aggr_mul *= tl.reduce(neighbor_feat_j + pos_j, axis=0, combine_fn=multiply) # Take the last element of the cumprod
        #tl.atomic_add(dh_src + node_index_j * IN_CHAN + feat_offsets, h_out_grad * rel_feat_j * rest_j, feat_valid_mask)
        # for place in range(MAX_ARITY-1):
        #     pos_index_j = pos_index + col_start + index + place * NUM_EDGE
        #     arity = tl.load(pos_index_j) # extract the correct arity for current node in this edge. 
        #     node_index_j = tl.load(adj_indices + col_start + index  + place * NUM_EDGE)
        #     if node_index_j != 0:
        #         # Need a mask here to see if node_index_j is 0 or not. If it is 0, which is padding node, we should stop the for loop
        #         pos_j = tl.load(pos_encodings +  IN_CHAN * arity  + feat_offsets, feat_valid_mask, feat_ones)
        #         neighbor_feat_j = tl.load(h_in + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        #         aggr_mul *= (neighbor_feat_j + pos_j) 
        # # Compute everything again excepts one. A dummy way of doing this
        # for place in range(MAX_ARITY - 1):
        #     pos_index_j = pos_index + col_start + index + place * NUM_EDGE
        #     arity = tl.load(pos_index_j) 
        #     node_index_j = tl.load(adj_indices + col_start + index  + place * NUM_EDGE)
        #     if node_index_j != 0:
        #         pos_j = tl.load(pos_encodings +  IN_CHAN * arity  + feat_offsets, feat_valid_mask, feat_ones)
        #         neighbor_feat_j = tl.load(h_in + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        #         rest_j = aggr_mul / (neighbor_feat_j + pos_j + 1e-7) # Avoid division by zero. TODO: Need further improvement
        #         tl.atomic_add(dh_src + node_index_j * IN_CHAN + feat_offsets, h_out_grad * rel_feat_j * rest_j, feat_valid_mask)
        # # Here, we store the rel_feature_j for this edge, which is the agg_mul now. 




class RelConvSumAggr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_in, rowptr, indices, out_node_count, edge_types, rel_features, work_group_size = None):
        # need to set the current CUDA device to avoid the error
        # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
        # https://github.com/openai/triton/issues/2925 
        torch.cuda.set_device(h_in.device)
        # get node feature count
        num_features_per_node = h_in.shape[1]
        # if work_group_size is not specified, pick a default value based on node feature count
        if not work_group_size:
            work_group_size = 32
            while work_group_size < num_features_per_node:
                work_group_size *= 2
        # calculate kernel configuration
        num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
        num_nodes = out_node_count
        # invoke triton forward kernel
        h_out = torch.empty((out_node_count, h_in.shape[1]),  dtype=h_in.dtype, layout=h_in.layout, device=h_in.device)
        relconv_aggregate_sum_kernel_forward[(num_nodes, num_work_groups)]( 
                    h_out, rowptr, indices, h_in, edge_types, rel_features,
                    num_features_per_node, work_group_size, num_warps=32)  # fixing num_warps to 32 as in the cuda rspmm kernel
        # save parameters for backward
        h_in_grad = torch.zeros_like(h_in, requires_grad=False)
        rel_in_grad = torch.zeros_like(rel_features, requires_grad=False)
        work_group_size_shaped_dummy = torch.empty(work_group_size, dtype=torch.int8)
        ctx.save_for_backward(rowptr, indices, h_in_grad, h_in,
                              edge_types, rel_features, rel_in_grad, work_group_size_shaped_dummy)
        return h_out
    
    # # Alternative v2u kernel
    # @staticmethod
    # def forward(ctx, h_in, rowptr, indices, out_node_count, edge_types, rel_features, work_group_size = None):
    #     num_nodes, bsz, feat_dim = h_in.size()
    #     WG_SIZE_BATCH = 1  # min(4, bsz)
    #     WG_SIZE_FEATS = triton.next_power_of_2(feat_dim)
    #     WG_SIZE_NODES = 4

    #     if not work_group_size:
    #         work_group_size = 32
    #         while work_group_size < feat_dim:
    #             work_group_size *= 2

    #     h_out = torch.empty_like(h_in)
    #     grid = (triton.cdiv(feat_dim, WG_SIZE_FEATS),
    #             triton.cdiv(bsz, WG_SIZE_BATCH),
    #             triton.cdiv(num_nodes, WG_SIZE_NODES))
    #     # invoke triton forward kernel
    #     #h_out = torch.empty((out_node_count, h_in.shape[1]),  dtype=h_in.dtype, layout=h_in.layout, device=h_in.device)
    #     relconv_aggregate_sum_kernel_forward_v2u[grid]( 
    #                 h_out, rowptr, indices, h_in, edge_types, rel_features,
    #                 feat_dim, bsz, num_nodes, WG_SIZE_NODES, WG_SIZE_BATCH, WG_SIZE_FEATS,)  # fixing num_warps to 32 as in the cuda rspmm kernel
    #     # save parameters for backward
    #     h_in_grad = torch.zeros_like(h_in, requires_grad=False)
    #     rel_in_grad = torch.zeros_like(rel_features, requires_grad=False)
    #     work_group_size_shaped_dummy = torch.empty(work_group_size, dtype=torch.int8)
    #     ctx.save_for_backward(rowptr, indices, h_in_grad, h_in,
    #                           edge_types, rel_features, rel_in_grad, work_group_size_shaped_dummy)
    #     return h_out
    
    @staticmethod
    def backward(ctx, h_out_grad):
        # get saved variables from forward
        rowptr, indices, h_in_grad, h_in, edge_types, rel_features, rel_in_grad, work_group_size_shape_dummy = ctx.saved_tensors
        work_group_size = work_group_size_shape_dummy.shape[0]
        # calculate kernel configuration
        num_features_per_node = h_out_grad.shape[1]
        num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
        num_nodes = h_out_grad.shape[0]
        # invoke triton backward kernel
        relconv_aggregate_sum_kernel_backward[(num_nodes,num_work_groups)](
                    h_in_grad, rel_in_grad, rowptr, indices, h_in, h_out_grad, edge_types, rel_features,
                    num_features_per_node, work_group_size, num_warps=32)
        return h_in_grad, None, None, None, None, rel_in_grad, None



class HyperRelConvSumAggr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_in, rowptr, indices, out_node_count, edge_types, rel_features, pos_encodings, pos_index, work_group_size = None):
        # need to set the current CUDA device to avoid the error
        # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
        # https://github.com/openai/triton/issues/2925 
        torch.cuda.set_device(h_in.device)
        # Note that h_in does not have padding node h_in
        max_arity = pos_index.shape[0] + 1 # Real max arity
        num_edge = indices.shape[1] 
        # get node feature count
        num_features_per_node = h_in.shape[1] # base_num_features * batch_size
        
        # if work_group_size is not specified, pick a default value based on node feature count
        if not work_group_size:
            work_group_size = 32
            while work_group_size < num_features_per_node:
                work_group_size *= 2
        # calculate kernel configuration
        num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
        num_nodes = out_node_count # Number of nodes is one more as there are padding nodes x = 0; Remember to reset the padding node. We could ignore this node in the computation
        # invoke triton forward kernel
        h_out = torch.empty((out_node_count, h_in.shape[1]),  dtype=h_in.dtype, layout=h_in.layout, device=h_in.device)
        hyper_relconv_aggregate_sum_kernel_forward[(num_nodes, num_work_groups)]( 
                    h_out, rowptr, indices, h_in, edge_types, rel_features, pos_encodings, pos_index, max_arity, num_edge,
                    num_features_per_node, work_group_size, num_warps=32)  # fixing num_warps to 32 as in the cuda rspmm kernel
        # save parameters for backward
        h_in_grad = torch.zeros_like(h_in, requires_grad=False)
        rel_in_grad = torch.zeros_like(rel_features, requires_grad=False)
        work_group_size_shaped_dummy = torch.empty(work_group_size, dtype=torch.int8)
        ctx.save_for_backward(rowptr, indices, h_in_grad, h_in,
                              edge_types, rel_features, rel_in_grad, pos_encodings, pos_index,
                              work_group_size_shaped_dummy)
        return h_out
    
    # Alternative v2u kernel
    # def forward(ctx, h_in, rowptr, indices, out_node_count, edge_types, rel_features, pos_encodings, pos_index, work_group_size = None):
    #     # need to set the current CUDA device to avoid the error
    #     # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    #     # https://github.com/openai/triton/issues/2925 
    #     torch.cuda.set_device(h_in.device)
    #     # Note that h_in does not have padding node h_in
    #     max_arity = pos_index.shape[0] + 1 # Real max arity
    #     num_edge = indices.shape[1] 
    #     # get node feature count
    #     num_features_per_node = h_in.shape[1] # base_num_features * batch_size
        
    #     # if work_group_size is not specified, pick a default value based on node feature count
    #     if not work_group_size:
    #         work_group_size = 32
    #         while work_group_size < num_features_per_node:
    #             work_group_size *= 2
    #     # calculate kernel configuration
    #     num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
    #     num_nodes = out_node_count # Number of nodes is one more as there are padding nodes x = 0; Remember to reset the padding node. We could ignore this node in the computation
    #     next_max_arity = triton.next_power_of_2(max_arity - 1)
    #     # invoke triton forward kernel
    #     h_out = torch.empty((out_node_count, h_in.shape[1]),  dtype=h_in.dtype, layout=h_in.layout, device=h_in.device)
    #     hyper_relconv_aggregate_sum_kernel_v2u_forward[(num_nodes, num_work_groups)]( 
    #                 h_out, rowptr, indices, h_in, edge_types, rel_features, pos_encodings, pos_index, max_arity, num_edge,
    #                 num_features_per_node, work_group_size, next_max_arity, num_warps=32)  # fixing num_warps to 32 as in the cuda rspmm kernel
    #     # save parameters for backward
    #     h_in_grad = torch.zeros_like(h_in, requires_grad=False)
    #     rel_in_grad = torch.zeros_like(rel_features, requires_grad=False)
    #     work_group_size_shaped_dummy = torch.empty(work_group_size, dtype=torch.int8)
    #     ctx.save_for_backward(rowptr, indices, h_in_grad, h_in,
    #                           edge_types, rel_features, rel_in_grad, pos_encodings, pos_index,
    #                           work_group_size_shaped_dummy)
    #     return h_out
    
    @staticmethod
    def backward(ctx, h_out_grad):
        # # get saved variables from forward
        rowptr, indices, h_in_grad, h_in, edge_types, rel_features, rel_in_grad, pos_encodings, pos_index, work_group_size_shaped_dummy = ctx.saved_tensors    
        max_arity = pos_index.shape[0] + 1 # Real max arity
        num_edge = indices.shape[1] 
        work_group_size = work_group_size_shaped_dummy.shape[0]
        # calculate kernel configuration
        num_features_per_node = h_out_grad.shape[1]
        num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
        num_nodes = h_out_grad.shape[0]
        # invoke triton backward kernel
        hyper_relconv_aggregate_sum_kernel_backward[(num_nodes,num_work_groups)](
                    h_in_grad, rel_in_grad, rowptr, indices, h_in, h_out_grad, edge_types, rel_features, pos_encodings, pos_index, max_arity, num_edge,
                    num_features_per_node, work_group_size, num_warps=32)
        return h_in_grad, None, None, None, None, rel_in_grad, None, None, None

    # Alternative v2u kernel
    # @staticmethod
    # def backward(ctx, h_out_grad):
    #     # # get saved variables from forward
    #     rowptr, indices, h_in_grad, h_in, edge_types, rel_features, rel_in_grad, pos_encodings, pos_index, work_group_size_shaped_dummy = ctx.saved_tensors    
    #     max_arity = pos_index.shape[0] + 1 # Real max arity
    #     num_edge = indices.shape[1] 
    #     next_max_arity = triton.next_power_of_2(max_arity - 1)
    #     work_group_size = work_group_size_shaped_dummy.shape[0]
    #     # calculate kernel configuration
    #     num_features_per_node = h_out_grad.shape[1]
    #     num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
    #     num_nodes = h_out_grad.shape[0]
    #     # invoke triton backward kernel
    #     hyper_relconv_aggregate_sum_kernel_v2u_backward[(num_nodes,num_work_groups)](
    #                 h_in_grad, rel_in_grad, rowptr, indices, h_in, h_out_grad, edge_types, rel_features, pos_encodings, pos_index, max_arity, num_edge,
    #                 num_features_per_node, work_group_size, next_max_arity, num_warps=32)
    #     return h_in_grad, None, None, None, None, rel_in_grad, None, None, None



class RelConvMeanAggr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_in, rowptr, indices, out_node_count, edge_types, rel_features, work_group_size = None):
        # need to set the current CUDA device to avoid the error
        # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
        # https://github.com/openai/triton/issues/2925 
        torch.cuda.set_device(h_in.device)
        # get node feature count
        num_features_per_node = h_in.shape[1]
        # if work_group_size is not specified, pick a default value based on node feature count
        if not work_group_size:
            work_group_size = 32
            while work_group_size < num_features_per_node:
                work_group_size *= 2
        # calculate kernel configuration
        num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
        num_nodes = out_node_count
        # invoke triton forward kernel
        h_out = torch.empty((out_node_count, h_in.shape[1]),  dtype=h_in.dtype, layout=h_in.layout, device=h_in.device)
        relconv_aggregate_mean_kernel_forward[(num_nodes, num_work_groups)]( 
                    h_out, rowptr, indices, h_in, edge_types, rel_features,
                    num_features_per_node, work_group_size, num_warps=32)  # fixing num_warps to 32 as in the cuda rspmm kernel
        # save parameters for backward
        h_in_grad = torch.zeros_like(h_in, requires_grad=False)
        rel_in_grad = torch.zeros_like(rel_features, requires_grad=False)
        work_group_size_shaped_dummy = torch.empty(work_group_size, dtype=torch.int8)
        ctx.save_for_backward(rowptr, indices, h_in_grad, h_in,
                              edge_types, rel_features, rel_in_grad, work_group_size_shaped_dummy)
        return h_out
    
    # # Alternative v2u kernel
    # @staticmethod
    # def forward(ctx, h_in, rowptr, indices, out_node_count, edge_types, rel_features, work_group_size = None):
    #     num_nodes, bsz, feat_dim = h_in.size()
    #     WG_SIZE_BATCH = 1  # min(4, bsz)
    #     WG_SIZE_FEATS = triton.next_power_of_2(feat_dim)
    #     WG_SIZE_NODES = 4

    #     if not work_group_size:
    #         work_group_size = 32
    #         while work_group_size < feat_dim:
    #             work_group_size *= 2

    #     h_out = torch.empty_like(h_in)
    #     grid = (triton.cdiv(feat_dim, WG_SIZE_FEATS),
    #             triton.cdiv(bsz, WG_SIZE_BATCH),
    #             triton.cdiv(num_nodes, WG_SIZE_NODES))
    #     # invoke triton forward kernel
    #     #h_out = torch.empty((out_node_count, h_in.shape[1]),  dtype=h_in.dtype, layout=h_in.layout, device=h_in.device)
    #     relconv_aggregate_mean_kernel_forward_v2u[grid]( 
    #                 h_out, rowptr, indices, h_in, edge_types, rel_features,
    #                 feat_dim, bsz, num_nodes, WG_SIZE_NODES, WG_SIZE_BATCH, WG_SIZE_FEATS,)  # fixing num_warps to 32 as in the cuda rspmm kernel
    #     # save parameters for backward
    #     h_in_grad = torch.zeros_like(h_in, requires_grad=False)
    #     rel_in_grad = torch.zeros_like(rel_features, requires_grad=False)
    #     work_group_size_shaped_dummy = torch.empty(work_group_size, dtype=torch.int8)
    #     ctx.save_for_backward(rowptr, indices, h_in_grad, h_in,
    #                           edge_types, rel_features, rel_in_grad, work_group_size_shaped_dummy)
    #     return h_out
    
    @staticmethod
    def backward(ctx, h_out_grad):
        # get saved variables from forward
        rowptr, indices, h_in_grad, h_in, edge_types, rel_features, rel_in_grad, work_group_size_shape_dummy = ctx.saved_tensors
        work_group_size = work_group_size_shape_dummy.shape[0]
        # calculate kernel configuration
        num_features_per_node = h_out_grad.shape[1]
        num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
        num_nodes = h_out_grad.shape[0]
        # invoke triton backward kernel
        relconv_aggregate_mean_kernel_backward[(num_nodes,num_work_groups)](
                    h_in_grad, rel_in_grad, rowptr, indices, h_in, h_out_grad, edge_types, rel_features,
                    num_features_per_node, work_group_size, num_warps=32)
        return h_in_grad, None, None, None, None, rel_in_grad, None



class HyperRelConvMeanAggr(torch.autograd.Function):
    @staticmethod
    # def forward(ctx, h_in, rowptr, indices, out_node_count, edge_types, rel_features, pos_encodings, pos_index, work_group_size = None):
    #     # need to set the current CUDA device to avoid the error
    #     # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    #     # https://github.com/openai/triton/issues/2925 
    #     torch.cuda.set_device(h_in.device)
    #     # Note that h_in does not have padding node h_in
    #     max_arity = pos_index.shape[0] + 1 # Real max arity
    #     num_edge = indices.shape[1] 
    #     # get node feature count
    #     num_features_per_node = h_in.shape[1] # base_num_features * batch_size
        
    #     # if work_group_size is not specified, pick a default value based on node feature count
    #     if not work_group_size:
    #         work_group_size = 32
    #         while work_group_size < num_features_per_node:
    #             work_group_size *= 2
    #     # calculate kernel configuration
    #     num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
    #     num_nodes = out_node_count # Number of nodes is one more as there are padding nodes x = 0; Remember to reset the padding node. We could ignore this node in the computation
    #     # invoke triton forward kernel
    #     h_out = torch.empty((out_node_count, h_in.shape[1]),  dtype=h_in.dtype, layout=h_in.layout, device=h_in.device)
    #     hyper_relconv_aggregate_mean_kernel_forward[(num_nodes, num_work_groups)]( 
    #                 h_out, rowptr, indices, h_in, edge_types, rel_features, pos_encodings, pos_index, max_arity, num_edge,
    #                 num_features_per_node, work_group_size, num_warps=32)  # fixing num_warps to 32 as in the cuda rspmm kernel
    #     # save parameters for backward
    #     h_in_grad = torch.zeros_like(h_in, requires_grad=False)
    #     rel_in_grad = torch.zeros_like(rel_features, requires_grad=False)
    #     work_group_size_shaped_dummy = torch.empty(work_group_size, dtype=torch.int8)
    #     ctx.save_for_backward(rowptr, indices, h_in_grad, h_in,
    #                           edge_types, rel_features, rel_in_grad, pos_encodings, pos_index,
    #                           work_group_size_shaped_dummy)
    #     return h_out
    
    # Alternative v2u kernel
    def forward(ctx, h_in, rowptr, indices, out_node_count, edge_types, rel_features, pos_encodings, pos_index, work_group_size = None):
        # need to set the current CUDA device to avoid the error
        # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
        # https://github.com/openai/triton/issues/2925 
        torch.cuda.set_device(h_in.device)
        # Note that h_in does not have padding node h_in
        max_arity = pos_index.shape[0] + 1 # Real max arity
        num_edge = indices.shape[1] 
        # get node feature count
        num_features_per_node = h_in.shape[1] # base_num_features * batch_size
        
        # if work_group_size is not specified, pick a default value based on node feature count
        if not work_group_size:
            work_group_size = 32
            while work_group_size < num_features_per_node:
                work_group_size *= 2
        # calculate kernel configuration
        num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
        num_nodes = out_node_count # Number of nodes is one more as there are padding nodes x = 0; Remember to reset the padding node. We could ignore this node in the computation
        next_max_arity = triton.next_power_of_2(max_arity - 1)
        # invoke triton forward kernel
        h_out = torch.empty((out_node_count, h_in.shape[1]),  dtype=h_in.dtype, layout=h_in.layout, device=h_in.device)
        hyper_relconv_aggregate_mean_kernel_v2u_forward[(num_nodes, num_work_groups)]( 
                    h_out, rowptr, indices, h_in, edge_types, rel_features, pos_encodings, pos_index, max_arity, num_edge,
                    num_features_per_node, work_group_size, next_max_arity, num_warps=32)  # fixing num_warps to 32 as in the cuda rspmm kernel
        # save parameters for backward
        h_in_grad = torch.zeros_like(h_in, requires_grad=False)
        rel_in_grad = torch.zeros_like(rel_features, requires_grad=False)
        work_group_size_shaped_dummy = torch.empty(work_group_size, dtype=torch.int8)
        ctx.save_for_backward(rowptr, indices, h_in_grad, h_in,
                              edge_types, rel_features, rel_in_grad, pos_encodings, pos_index,
                              work_group_size_shaped_dummy)
        return h_out
    
    # @staticmethod
    # def backward(ctx, h_out_grad):
    #     # # get saved variables from forward
    #     rowptr, indices, h_in_grad, h_in, edge_types, rel_features, rel_in_grad, pos_encodings, pos_index, work_group_size_shaped_dummy = ctx.saved_tensors    
    #     max_arity = pos_index.shape[0] + 1 # Real max arity
    #     num_edge = indices.shape[1] 
    #     work_group_size = work_group_size_shaped_dummy.shape[0]
    #     # calculate kernel configuration
    #     num_features_per_node = h_out_grad.shape[1]
    #     num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
    #     num_nodes = h_out_grad.shape[0]
    #     # invoke triton backward kernel
    #     hyper_relconv_aggregate_mean_kernel_backward[(num_nodes,num_work_groups)](
    #                 h_in_grad, rel_in_grad, rowptr, indices, h_in, h_out_grad, edge_types, rel_features, pos_encodings, pos_index, max_arity, num_edge,
    #                 num_features_per_node, work_group_size, num_warps=32)
    #     return h_in_grad, None, None, None, None, rel_in_grad, None, None, None

    # Alternative v2u kernel
    @staticmethod
    def backward(ctx, h_out_grad):
        # # get saved variables from forward
        rowptr, indices, h_in_grad, h_in, edge_types, rel_features, rel_in_grad, pos_encodings, pos_index, work_group_size_shaped_dummy = ctx.saved_tensors    
        max_arity = pos_index.shape[0] + 1 # Real max arity
        num_edge = indices.shape[1] 
        next_max_arity = triton.next_power_of_2(max_arity - 1)
        work_group_size = work_group_size_shaped_dummy.shape[0]
        # calculate kernel configuration
        num_features_per_node = h_out_grad.shape[1]
        num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
        num_nodes = h_out_grad.shape[0]
        # invoke triton backward kernel
        hyper_relconv_aggregate_mean_kernel_v2u_backward[(num_nodes,num_work_groups)](
                    h_in_grad, rel_in_grad, rowptr, indices, h_in, h_out_grad, edge_types, rel_features, pos_encodings, pos_index, max_arity, num_edge,
                    num_features_per_node, work_group_size, next_max_arity, num_warps=32)
        return h_in_grad, None, None, None, None, rel_in_grad, None, None, None


