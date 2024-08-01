#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2024-01-23] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import torch
import math

from auto_gptq.quantization.gptq import GPTQ
from auto_gptq.modeling._utils import find_layers

from quantize.quantizer import FixedScaleQuantizer
from quantize.int_linear import QuantLinear
from quantize.utils import set_quant_state, smooth_and_quant_inplace


def gptq(lm, args, dataloader, logger):
    logger.info("Starting GPTQ...")

    # 1. move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    logger.info("model.device: {}".format(dev))
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    # TODO(xcsong): support other archs
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layers_block_name = "model.layers"
        inside_layer_modules = [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"]
        ]
    else:
        raise ValueError("Only support for Llama-2 now")
    layers[0] = layers[0].to(dev)

    # 2. catch the first layer input
    dtype = next(iter(model.parameters())).dtype
    logger.info("model.dtype: {}".format(dtype))
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size),
        dtype=dtype, device=dev
    )
    outs = torch.zeros_like(inps)
    cache = {"i": 0, 'attention_mask': None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    for batch in dataloader:
        if cache["i"] >= args.nsamples:
            break
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass

    # 3. move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    else:
        raise ValueError("Only support for Llama-2 now")
    torch.cuda.empty_cache()

    # 4. get additional inputs (mask, pos, ..., etc)
    attention_mask = cache["attention_mask"]
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    # 5. start gptq quantization
    quantizers = {}
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} with GPTQ ===")
        layer = layers[i].to(dev)

        # 5.1 get layers which should be quantized
        full = find_layers(layer, layers=[QuantLinear])
        for names in inside_layer_modules:
            # NOTE(xcsong): type(subset[name]) == QuantLinear
            #   i.e. subset["self_attn.k_proj"] = QuantLinear(**someargs)
            #               type(gptq[name]) == GPTQ
            #   i.e. gptq["self_attn.k_proj"] = GPTQ(subset["self_attn.k_proj"])  # noqa
            subset = {n: full[n] for n in names if n in full}
            gptq = {}
            # 5.1.1 init gptq
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                # NOTE(xcsong): Overwrite GPTQ().quantizer, use fixed scale
                #   and zero obtained from omniquant's quantizer
                _ = subset[name].weight_quantizer(subset[name].weight)
                subset[name].weight_quantizer.register_scales_and_zeros()
                scale = subset[name].weight_quantizer.scales
                zero = subset[name].weight_quantizer.zeros \
                    if subset[name].weight_quantizer.zeros is not None \
                    else torch.zeros_like(scale)
                gptq[name].quantizer = FixedScaleQuantizer(
                    scale=scale, zero=zero,
                    **args.weight_quant_params,
                    shape=subset[name].weight.shape
                )

            # 5.1.2 init gptq.H
            # NOTE(xcsong): Overwrite GPTQ().add_batch(), since
            #   1. type(gptq[name].layer) is QuantLinear, not nn.Linear,
            #       making it incompatible with the original implementation.
            #   2. We might consider utilizing fake quantized activations
            #       for the calculation of H.
            def add_batch(name):
                def tmp(_, inp, out):
                    # apply fake_quant to actiavtion
                    inp = subset[name].act_quantizer(inp[0].data)
                    if len(inp.shape) == 2:
                        inp = inp.unsqueeze(0)
                    batch = inp.shape[0]
                    if isinstance(gptq[name].layer, QuantLinear):
                        if len(inp.shape) == 3:
                            inp = inp.reshape((-1, inp.shape[-1]))
                        inp = inp.t()
                    else:
                        raise NotImplementedError()
                    gptq[name].H *= gptq[name].nsamples / (gptq[name].nsamples + batch)
                    gptq[name].nsamples += batch
                    inp = math.sqrt(2 / gptq[name].nsamples) * inp.float()
                    gptq[name].H += inp.matmul(inp.t())
                return tmp

            handles = []
            for name in subset:
                set_quant_state(subset[name], weight_quant=False,
                                act_quant=False)
                subset[name].use_temporary_parameter = False
                handles.append(subset[name].register_forward_hook(
                    add_batch(name)))
            for j in range(args.nsamples):
                layer(inps[j].unsqueeze(0),
                      attention_mask=attention_mask,
                      position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            # 5.1.3 do gptq-algorithm and update weight in-place
            for name in subset:
                logger.info(f'Quantize {name} in layer {i + 1}/{len(layers)}')
                scale, zero, g_idx = gptq[name].fasterquant(
                    blocksize=128,  # same as gptq
                    percdamp=0.01,  # same as gptq, always choose 1% of the average diagonal value  # noqa
                    group_size=-1 if args.group_size is None else args.group_size,  # group = None means per-channel  # noqa
                    actorder=False,
                    static_groups=False
                )
                quantizers[f'{layers_block_name}.{i}.{name}'] = (
                    gptq[name].quantizer.cpu(), scale.cpu(),
                    zero.cpu(), g_idx.cpu()
                )
                gptq[name].free()
                torch.cuda.empty_cache()

        # 5.2 quantize weight optimized by gptq
        # NOTE(xcsong): After GPTQ quantization, we do
        #   online fake_quantize for activation (via set_quant_state)
        #       and
        #   offline in-place fake_quantize for weights (via smooth_and_quant_inplace)
        layer.half()
        set_quant_state(layer, weight_quant=False, act_quant=True)
        prev_let, prev_gptq = args.let, args.gptq
        args.let, args.gptq = False, False
        smooth_and_quant_inplace(layer, args)
        args.let, args.gptq = prev_let, prev_gptq

        # 5.3 get output of current layer, treat it as input for next layer
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids)[0]
        inps, outs = outs, inps
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
