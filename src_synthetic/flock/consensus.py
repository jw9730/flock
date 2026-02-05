import torch
from torch import Tensor


def consensus_mean(x: Tensor, x_ids: Tensor, num_ids: int) -> Tensor:
    """
    Scatter sequence features
    Arguments:
        x:          [bsize * samples, walk_len, dim]
        x_ids:      [bsize, samples, walk_len]
        num_ids:    int
    Return:
        out:        [bsize, num_ids, dim]
    """
    bsize, samples, walk_len = x_ids.shape
    _, _, dim = x.shape

    out = torch.zeros(bsize * num_ids, dim, device=x.device, dtype=x.dtype)
    src = x.view(bsize * samples * walk_len, dim)

    batch_offset = torch.arange(bsize, device=x.device) * num_ids
    index = x_ids + batch_offset[:, None, None]
    index = index.reshape(bsize * samples * walk_len)[:, None].expand(-1, dim)

    out.scatter_reduce_(0, index, src, reduce="mean", include_self=False)
    return out.view(bsize, num_ids, dim)


def consensus_softmax_stable(
    x: Tensor, logits: Tensor, x_ids: Tensor, num_ids: int
) -> Tensor:
    """
    Scatter sequence features with attention weights (numerically stable version)
    Arguments:
        x:          [bsize * samples, walk_len, dim]
        logits:     [bsize * samples, walk_len, heads]
        x_ids:      [bsize, samples, walk_len]
        num_ids:    int
    Return:
        out:        [bsize, num_ids, dim]
    """
    bsize, samples, walk_len = x_ids.shape
    _, _, dim = x.shape
    _, _, heads = logits.shape
    assert (
        dim % heads == 0
    ), f"Dimension {dim} must be divisible by number of heads {heads}"

    # Reshape for processing
    logits_flat = logits.view(bsize * samples * walk_len, heads)
    x_flat = x.view(bsize * samples * walk_len, heads, dim // heads)

    # Find max logits for numerical stability (log-sum-exp trick)
    max_logits = torch.zeros(bsize * num_ids, heads, device=x.device, dtype=x.dtype)
    batch_offset = torch.arange(bsize, device=x.device) * num_ids
    index = x_ids + batch_offset[:, None, None]
    index_flat = index.reshape(bsize * samples * walk_len)[:, None]

    # Compute max logits for each group
    max_logits.scatter_reduce_(
        0, index_flat.expand(-1, heads), logits_flat, reduce="amax", include_self=False
    )

    # Subtract max for stability and compute exp
    logits_stable = logits_flat - max_logits.gather(0, index_flat.expand(-1, heads))
    exp_stable = torch.exp(logits_stable)

    # Prepare output tensors
    out_weighted = torch.zeros(bsize * num_ids, dim, device=x.device, dtype=x.dtype)
    out_weights = torch.zeros(bsize * num_ids, heads, device=x.device, dtype=x.dtype)

    # Compute weighted features and sum of weights
    x_weighted = (x_flat * exp_stable[:, :, None]).flatten(1, 2)

    # Scatter weighted features and weights
    out_weighted.scatter_reduce_(
        0, index_flat.expand(-1, dim), x_weighted, reduce="sum", include_self=False
    )
    out_weights.scatter_reduce_(
        0, index_flat.expand(-1, heads), exp_stable, reduce="sum", include_self=False
    )

    # Compute final result with numerical stability
    out_weighted = out_weighted.view(bsize * num_ids, heads, dim // heads)
    out_weights_safe = out_weights[:, :, None] + 1e-8
    out = out_weighted / out_weights_safe

    return out.view(bsize, num_ids, dim)
