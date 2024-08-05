import importlib
import numbers

import torch
from torch.nn import init
from torch.nn.parameter import Parameter

global cryofold_layernorm_cuda
cryofold_layernorm_cuda = None


class FusedLayerNormAffineFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):

        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = cryofold_layernorm_cuda.forward_affine(
            input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input, grad_weight, grad_bias \
            = cryofold_layernorm_cuda.backward_affine(
                grad_output.contiguous(), mean, invvar,
                input_, ctx.normalized_shape,
                weight_, bias_, ctx.eps)

        return grad_input, grad_weight, grad_bias, None, None


class MixedFusedLayerNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-5, dtype=torch.bfloat16):
        super(MixedFusedLayerNorm, self).__init__()

        global cryofold_layernorm_cuda
        if cryofold_layernorm_cuda is None:
            try:
                cryofold_layernorm_cuda = importlib.import_module("cryofold_layernorm_cuda")
            except ImportError:
                raise RuntimeError('MixedFusedLayerNorm requires cuda extensions')

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        # self.weight = self.weight.to(dtype)
        # self.bias = self.bias.to(dtype)
        self.reset_parameters()

    def reset_parameters(self):

        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input):

        return FusedLayerNormAffineFunction.apply(input, self.weight, self.bias,
                                                  self.normalized_shape, self.eps)
