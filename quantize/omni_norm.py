import torch
import torch.nn as nn


'''
Modify normalization layer to adapt the training of learnable equivalent transformation
'''


class OmniLayerNorm(nn.Module):
    def __init__(self, ori_layer_norm) -> None:
        super().__init__()
        self.use_act_quant = True
        self.weight = nn.Parameter(ori_layer_norm.weight.clone())
        self.bias = nn.Parameter(ori_layer_norm.bias.clone()) if ori_layer_norm.bias is not None else None
        self.eps = ori_layer_norm.eps
        self.norm_func = nn.functional.layer_norm
        self.normalized_shape = ori_layer_norm.normalized_shape
        self.use_temporary_parameter = False

    def forward(self, x):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias

        out = self.norm_func(x, self.normalized_shape, weight, bias, eps=self.eps)
        return out

    def set_quant_state(self, use_weight_quant, use_act_quant):
        self.use_act_quant = use_act_quant

    def extra_repr(self):
        return f"{tuple(self.normalized_shape)}, eps={self.eps}"


class OmniLlamaRMSNorm(nn.Module):
    def __init__(self, ori_norm, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(ori_norm.weight.clone())
        self.variance_epsilon = eps
        self.use_temporary_parameter = False

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        return (weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


