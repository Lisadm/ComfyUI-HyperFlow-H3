import torch
from safetensors import safe_open


def expand_qkv(tensor):
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError('Compact QKV requires three diagonal blocks')
    return torch.block_diag(*tensor.unbind(0))


def load_compact_lora(path):
    with safe_open(path, framework='pt', device='cpu') as handle:
        metadata = handle.metadata() or {}
        storage = metadata.get('qkv_storage')
        if storage not in (None, 'diagonal_blocks_v1'):
            raise ValueError('Expected a standard or diagonal_blocks_v1 LoRA')
        weights = {}
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            weights[key] = expand_qkv(tensor) if storage and key.endswith('attn.qkv_proj.lora_B.weight') else tensor
    return weights
