import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

spec = importlib.util.spec_from_file_location('compact_lora_tested', Path(__file__).resolve().parents[1] / 'hyperflow_h3' / 'compact_lora.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_compact_lora_preserves_dense_weights(tmp_path):
    blocks = torch.randn(3, 8, 4, dtype=torch.bfloat16)
    a = torch.randn(12, 8, dtype=torch.bfloat16)
    key = 'diffusion_model.blocks.0.attn.qkv_proj'
    path = tmp_path / 'compact.safetensors'
    save_file({key + '.lora_A.weight': a, key + '.lora_B.weight': blocks}, str(path),
              metadata={'qkv_storage': 'diagonal_blocks_v1'})
    result = module.load_compact_lora(path)
    assert torch.equal(result[key + '.lora_A.weight'], a)
    assert torch.equal(result[key + '.lora_B.weight'].view(torch.uint8), torch.block_diag(*blocks.unbind()).view(torch.uint8))


def test_unknown_format_is_rejected(tmp_path):
    path = tmp_path / 'plain.safetensors'
    save_file({'weight': torch.ones(2)}, str(path), metadata={'qkv_storage': 'unsupported'})
    with pytest.raises(ValueError, match='diagonal_blocks_v1'):
        module.load_compact_lora(path)


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match='three diagonal blocks'):
        module.expand_qkv(torch.ones(2, 4, 4))


def test_standard_lora_remains_supported(tmp_path):
    path = tmp_path / 'standard.safetensors'
    tensor = torch.randn(12, 4)
    key = 'diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight'
    save_file({key: tensor}, str(path))
    assert torch.equal(module.load_compact_lora(path)[key], tensor)
