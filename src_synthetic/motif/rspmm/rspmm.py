import os
import sys

import torch.backends.openmp
from torch import autograd
from torch.utils import cpp_extension

module = sys.modules[__name__]


class RSPMMAddMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_add_mul_forward_cuda
        else:
            forward = rspmm.rspmm_add_mul_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_add_mul_backward_cuda
        else:
            backward = rspmm.rspmm_add_mul_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMinMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_min_mul_forward_cuda
        else:
            forward = rspmm.rspmm_min_mul_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_min_mul_backward_cuda
        else:
            backward = rspmm.rspmm_min_mul_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMaxMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_max_mul_forward_cuda
        else:
            forward = rspmm.rspmm_max_mul_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_max_mul_backward_cuda
        else:
            backward = rspmm.rspmm_max_mul_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMAddAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_add_add_forward_cuda
        else:
            forward = rspmm.rspmm_add_add_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_add_add_backward_cuda
        else:
            backward = rspmm.rspmm_add_add_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMinAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_min_add_forward_cuda
        else:
            forward = rspmm.rspmm_min_add_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_min_add_backward_cuda
        else:
            backward = rspmm.rspmm_min_add_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


class RSPMMMaxAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, edge_index, edge_type, edge_weight, relation, input):
        node_in, node_out = edge_index
        key = node_in * (node_out.max() + 1) + node_out
        assert (key.diff() >= 0).all(), "Expect sorted `edge_index`"

        if input.device.type == "cuda":
            forward = rspmm.rspmm_max_add_forward_cuda
        else:
            forward = rspmm.rspmm_max_add_forward_cpu
        output = forward(edge_index, edge_type, edge_weight, relation, input)
        ctx.save_for_backward(edge_index, edge_type, edge_weight, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = rspmm.rspmm_max_add_backward_cuda
        else:
            backward = rspmm.rspmm_max_add_backward_cpu
        weight_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        return None, None, weight_grad, relation_grad, input_grad


def generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul="mul"):
    name = "RSPMM%s%sFunction" % (sum.capitalize(), mul.capitalize())
    if not hasattr(module, name):
        raise ValueError("No generalized rspmm implementation found for summation `%s` and multiplication `%s`"
                         % (sum, mul))
    Function = getattr(module, name)

    node_in, node_out = edge_index
    key = node_in * (node_out.max() + 1) + node_out
    order = key.argsort()

    return Function.apply(edge_index[:, order], edge_type[order], edge_weight[order], relation, input)


def load_extension(name, sources, extra_cflags=None, extra_cuda_cflags=None, **kwargs):
    if extra_cflags is None:
        extra_cflags = ["-Ofast"]
        if torch.backends.openmp.is_available():
            extra_cflags += ["-fopenmp", "-DAT_PARALLEL_OPENMP"]
        else:
            extra_cflags.append("-DAT_PARALLEL_NATIVE")
    if extra_cuda_cflags is None:
        if torch.cuda.is_available():
            extra_cuda_cflags = ["-O3"]
            extra_cflags.append("-DCUDA_OP")
        else:
            new_sources = []
            for source in sources:
                if not cpp_extension._is_cuda_file(source):
                    new_sources.append(source)
            sources = new_sources

    return cpp_extension.load(name, sources, extra_cflags, extra_cuda_cflags, **kwargs)


print("Load rspmm extension. This may take a while...")
path = os.path.join(os.path.dirname(__file__), "source")
rspmm = load_extension("rspmm", [os.path.join(path, "rspmm.cpp"), os.path.join(path, "rspmm.cu")])



if __name__ == '__main__':

    # A test for RSPMM
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]]) # This means from 1 to 0, 2 to 1, 3 to 2
    edge_type = torch.tensor([0, 1, 2])
    num_node = 4
    print(edge_index)

    # Generate sample input data; here they see each batches as a separate feature dim
    in_channels = 8
    relation = torch.tensor([[10 * (i+1) for _ in range(in_channels)] for i in range(3)], dtype=torch.float32) # the relation embeddings

    x = torch.zeros(1, num_node, in_channels)
    # Assign values
    for i in range(0, num_node ):
        x[:, i, :] = i + 1
    x = x.transpose(0, 1).flatten(1)
    # relation = relation.transpose(0, 1).flatten(1) # wait what?


    edge_weight = torch.ones(len(edge_type), device=x.device, dtype=torch.float32)

    # print(edge_index.size(), edge_type.size(), edge_weight.size(), relation.size(), x.size())

    update = generalized_rspmm(edge_index.to("cuda:0"), edge_type.to("cuda:0"), edge_weight.to("cuda:0"), relation.to("cuda:0"), x.to("cuda:0"), sum="add", mul="mul")
    
    
    print(update)
    assert torch.equal(update, torch.tensor(
        [[ 20.,  20.,  20.,  20.,  20.,  20.,  20.,  20.],
         [ 60.,  60.,  60.,  60.,  60.,  60.,  60.,  60.],
         [120., 120., 120., 120., 120., 120., 120., 120.],
         [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]]
    ).to("cuda:0")), "update is wrong for rspmm"
    "test passed!"

