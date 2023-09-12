import torch
import torch.nn as nn
from typing import List
from functools import partial
import subprocess
import re
import os
import time
import pdb


def nvidia_smi_memory_info():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    output = result.stdout.split("\n")[:-1]

    gpu_memory_info = []
    for line in output:
        gpu_id, total_memory, used_memory, free_memory = map(int, re.split(",\s", line))
        gpu_memory_info.append(
            {
                "id": gpu_id,
                "total_memory": total_memory,
                "used_memory": used_memory,
                "free_memory": free_memory,
            }
        )

    return gpu_memory_info


num_gpus = torch.cuda.device_count()


def get_gpu_memory():
    memory_info = []
    gpu_memory_info = nvidia_smi_memory_info()

    try:
        gpu_index = [int(k) for k in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    except KeyError:
        gpu_index = [x["id"] for x in gpu_memory_info]

    for gpu_id, i in enumerate( gpu_index):
        gpu = gpu_memory_info[i]
        total_memory = gpu["total_memory"]
        used_memory = gpu["used_memory"]
        memory_info.append((gpu_id, total_memory, used_memory))
    return memory_info


def get_lowest_occupied_gpu(wait_memory=1000):

    now_lowest_memory = 1e9
    while now_lowest_memory > wait_memory:
        if not now_lowest_memory == 1e9:
            time.sleep(10)
        memory_info = get_gpu_memory()
        gpu_id, tot_mem, used_mem = sorted(
            memory_info, key=lambda x: x[2], reverse=False
        )[0]
        now_lowest_memory = used_mem

    return gpu_id


def sort_layers_by_params(layers: List[nn.Module]):
    return sorted(
        layers, key=lambda m: sum(p.numel() for p in m.parameters()), reverse=True
    )


def get_all_gpu_free_memory():
    return sum(
        [
            total_memory - used_memory
            for gpu_id, total_memory, used_memory in get_gpu_memory()
        ]
    )


def assign_layers_to_gpus(layers: List[nn.Module]):
    layer_gpu_map = {}
    prev_gpu_id = None
    weight_num = 0
    for module in layers:
        if hasattr(module, "weight"):
            weight_num += module.weight.numel()
    weight_mb = weight_num * 2 / 1024 / 1024
    all_gpu_mems = get_all_gpu_free_memory()
    while all_gpu_mems < weight_mb * 1.3:
        time.sleep(10)
        all_gpu_mems = get_all_gpu_free_memory()

    for i, layer in enumerate(layers):
        if i == len(layers) - 1:
            layer_gpu_map[layer] = layer_gpu_map[layers[0]]
            layer.to(layers[0].device)
            layer.device = layers[0].device
            print(f"map last layer {i} to gpu {layer_gpu_map[layer]}")
            continue
        layer_memory = (
            sum(p.element_size() * p.numel() for p in layer.parameters()) / 1024**2
        )
        available_gpus = get_gpu_memory()
        if prev_gpu_id is None:
            gpus = sorted(available_gpus, key=lambda x: x[2])
        else:
            pre_gpu_info = available_gpus[prev_gpu_id]
            gpus = [pre_gpu_info] + sorted(available_gpus, key=lambda x: x[2])
        mapped = False
        for gpu_id, tot_memory, allocated_memory in gpus:
            if (tot_memory - allocated_memory * 1.35) > layer_memory:
                layer_gpu_map[layer] = gpu_id
                layer.to(f"cuda:{gpu_id}")
                layer.device = f"cuda:{gpu_id}"
                print(f"map layer {i} to gpu {gpu_id}, {available_gpus}")
                mapped = True
                prev_gpu_id = gpu_id
                break
        if not mapped:
            raise RuntimeError(f"memory not enough {available_gpus}")

    return layer_gpu_map


# forward hook
def forward_hook_wrapper(gpu_id):
    def forward_hook(module, input, kwargs):
        # breakpoint()
        input = tuple(_.to(f"cuda:{gpu_id}") for _ in input)
        kwargs = {
            k: v.to(f"cuda:{gpu_id}") if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        return input, kwargs

    return forward_hook


def add_forward_hooks(layer_gpu_map):
    prev_gpu_id = None
    for layer, gpu_id in layer_gpu_map.items():
        layer: nn.Module
        if prev_gpu_id is None:
            prev_gpu_id = gpu_id
        # if gpu_id != prev_gpu_id:
        layer.register_forward_pre_hook(forward_hook_wrapper(gpu_id), with_kwargs=True)
        prev_gpu_id = gpu_id


def map_layers_to_multi_gpus(layers):

    layer_gpu_map = assign_layers_to_gpus(layers)

    add_forward_hooks(layer_gpu_map)


if __name__ == "__main__":
    info = get_gpu_memory()
    print(info)
