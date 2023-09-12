import torch
from torch import nn
from typing import Optional, Tuple, List
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
import torch.nn.functional as F
from collections import OrderedDict
import math
from transformers.models.falcon.configuration_falcon import FalconConfig
from transformers.models.falcon.modeling_falcon import FalconAttention, dropout_add
import pdb
import copy
from models.transformation import *
from quantize.omni_norm import OmniLayerNorm




    
class QuantFalconMLP(nn.Module):
    def __init__(self, org_module: nn.Module,args=None):
        super().__init__()

        self.dense_h_to_4h = QuantLinear(org_module.dense_h_to_4h,args.weight_quant_params,args.act_quant_params)
        self.act = nn.GELU()
        self.dense_4h_to_h = QuantLinear(org_module.dense_4h_to_h,args.weight_quant_params,args.act_quant_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x


                
class QuantFalconAttention(nn.Module):
    def __init__(self,  config: FalconConfig, org_module: nn.Module, args=None):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.maybe_rotary = copy.deepcopy(org_module.maybe_rotary)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor
        self.query_key_value = QuantLinear(org_module.query_key_value,args.weight_quant_params,args.act_quant_params)
        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query
        self.dense =QuantLinear(org_module.dense,args.weight_quant_params,args.act_quant_params)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.num_kv_heads = config.num_kv_heads if (self.new_decoder_architecture or not self.multi_query) else 1


    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        if self.new_decoder_architecture:
            batch, seq_len, _ = fused_qkv.shape
            qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
            query = qkv[:, :, :, :-2]
            key = qkv[:, :, :, [-2]]
            value = qkv[:, :, :, [-1]]
            key = torch.broadcast_to(key, query.shape)
            value = torch.broadcast_to(value, query.shape)

            query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
            return query, key, value
        elif not self.multi_query:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        else:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads + 2, self.head_dim)
            return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]

    # Copied from transformers.models.bloom.modeling_bloom.BloomAttention._merge_heads
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, query_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(
            batch_size * num_kv_heads,
            query_length,
            self.head_dim,
        )
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * num_kv_heads, query_length, self.head_dim)

        past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]
        query_layer, key_layer = self.maybe_rotary(query_layer, key_layer, past_kv_length)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, kv_length, head_dim]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, kv_length, _ = key_layer.shape
        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None

        attention_mask_float = (attention_mask * 1.0).masked_fill(attention_mask, float("-1e9")).to(query_layer.dtype)

        query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)

        if alibi is None:
            if hasattr(F, "scaled_dot_product_attention") and not output_attentions:

                attn_output = F.scaled_dot_product_attention(
                    query_layer_, key_layer_, value_layer_, attention_mask_float, 0.0, is_causal=False
                )
                attention_scores = None
            else:
                attention_scores = query_layer_ @ key_layer_.transpose(-1, -2)
                attention_scores /= math.sqrt(self.head_dim)

                attention_scores = F.softmax(
                    attention_scores + attention_mask_float, dim=-1, dtype=hidden_states.dtype
                )
                attn_output = attention_scores @ value_layer_

            attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

            output_tensor = self.dense(attn_output)

            if output_attentions:
                return output_tensor, present, attention_scores
            else:
                return output_tensor, present

        else:
            matmul_result = query_layer_ @ key_layer_.transpose(-1, -2)

            # change view to [batch_size, num_heads, q_length, kv_length]
            attention_scores = matmul_result.view(batch_size, self.num_heads, query_length, kv_length)

            # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
            input_dtype = attention_scores.dtype
            # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
            if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                attention_scores = attention_scores.to(torch.float32)
            # Matt (HF) note: We could possibly use F.scaled_dot_product_attention here too, by
            # adding (alibi * self.inv_norm_factor) to attention_mask_float. I think this would be mathematically
            # equivalent and more performant, but there might be a numerical difference. If you're reading this
            # and you'd like to experiment and maybe file a PR, feel free!
            attention_logits = attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)
            attention_logits *= self.inv_norm_factor
            attention_probs = F.softmax(attention_logits + attention_mask_float, dim=-1, dtype=hidden_states.dtype)
            # [batch_size, num_heads, q_length, kv_length]
            attention_probs = self.attention_dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # change view [batch_size, num_heads, q_length, kv_length]
            attention_probs_reshaped = attention_probs.view(batch_size, self.num_heads, query_length, kv_length)

            # matmul: [batch_size * num_heads, q_length, head_dim]
            context_layer = (attention_probs_reshaped @ value_layer_).flatten(0, 1)

            # change view [batch_size, q_length, num_heads * head_dim]
            context_layer = self._merge_heads(context_layer)

            output_tensor = self.dense(context_layer)

            if output_attentions:
                return output_tensor, present, attention_probs
            else:
                return output_tensor, present


    
class QuantFalconDecoderLayer(nn.Module):
    def __init__(self, config: FalconConfig,
                 ori_layer,
                 args):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.self_attention = QuantFalconAttention(config, ori_layer.self_attention, args)
        self.mlp = QuantFalconMLP(ori_layer.mlp, args)
        self.hidden_dropout = config.hidden_dropout
        self.config = config

        if config.new_decoder_architecture:
            # The layer norm before self-attention
            self.ln_attn = OmniLayerNorm(ori_layer.ln_attn)
            # The layer norm before the MLP
            self.ln_mlp =  OmniLayerNorm(ori_layer.ln_mlp)
        else:
            self.input_layernorm = OmniLayerNorm(ori_layer.input_layernorm)
            if not config.parallel_attn:
                self.post_attention_layernorm = OmniLayerNorm(ori_layer.post_attention_layernorm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        alibi: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs
    ):
        residual = hidden_states
        if self.config.new_decoder_architecture:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)

        # Self attention.
        attn_outputs = self.self_attention(
            attention_layernorm_out,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(
                    attention_output, residual, self.config.attention_dropout, training=self.training
                )
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        outputs = attn_outputs[1:]

        # MLP.
        mlp_output = self.mlp(mlp_layernorm_out)

        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attention_output

        output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        if self.let:
            raise ValueError("falcon not yet support let")
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter=False
                

    def clear_temp_variable(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    def smooth_and_quant_temporary(self):
        if self.let:
            raise ValueError("falcon not yet support let")
        else:
            for name, module in self.named_modules():
                if isinstance(module, QuantLinear):
                    module.temp_weight = module.weight
        # quant
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter=True


    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)  

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
        return iter(params)  

    def omni_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)  
    
    def omni_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination
    
    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()
    