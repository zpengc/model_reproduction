import math
import torch
import torch.nn as nn


# Function需要定义三个方法：__init__, forward, backward（需要自己写求导公式）
# Module：只需定义__init__和forward，而backward的计算由自动求导机制构成
# https://zhuanlan.zhihu.com/p/344802526
class SpLinearFunc(torch.autograd.Function):
    """
    自定义一个操作
    step1：首先你要让它继承这个class：torch.autograd.Function
    step2：同时，实现forward和backward两个函数
    Note that both forward and backward are @staticmethods
    ctx:context
    """
    @staticmethod
    def forward(ctx, x, weight, bias=None):  # 线性层操作
        ctx.save_for_backward(x, weight, bias)
        output = x.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # 计算导数
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors  # 获得之前forward存的数据
        grad_input = grad_weight = grad_bias = None  # 初始化导数结果
        # ctx.needs_input_grad as a tuple of booleans representing whether each input needs gradient
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            # grad_weight = (input.t().mm(grad_output)).t()
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


# use it by calling the apply method:
# splinear = SpLinearFunc.apply


class SpLinear(nn.Module):

    def __init__(self, input_features, output_features, bias=True):
        super(SpLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        # TODO write a default initialization
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return SpLinearFunc.apply(x, self.weight, self.bias)