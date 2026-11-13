import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

spec = importlib.util.spec_from_file_location('baked_curve_tested', Path(__file__).resolve().parents[1] / 'hyperflow_h3' / 'baked_curve.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fit_must_match_the_baked_recipe(tmp_path):
    fit = tmp_path / 'fit.safetensors'
    fit.write_bytes(b'original curve')
    checkpoint = tmp_path / 'model.safetensors'
    metadata = {'format': 'h3-static-bake-v1', 'recipe': 'hyperflow_lms_ic21',
                'curve_fit_name': fit.name, 'curve_fit_sha256': hashlib.sha256(fit.read_bytes()).hexdigest()}
    save_file({'weight': torch.ones(1)}, str(checkpoint), metadata={'exact_h3_bake': json.dumps(metadata)})
    assert module.curve_for_checkpoint(checkpoint, [tmp_path]) == fit
    fit.write_bytes(b'changed curve')
    with pytest.raises(ValueError, match='original HyperFlow curve'):
        module.curve_for_checkpoint(checkpoint, [tmp_path])
    metadata['curve_fit_name'] = '../fit.safetensors'
    save_file({'weight': torch.ones(1)}, str(checkpoint), metadata={'exact_h3_bake': json.dumps(metadata)})
    with pytest.raises(ValueError, match='Invalid curve fit filename'):
        module.curve_for_checkpoint(checkpoint, [tmp_path])


def test_ordinary_checkpoint_cannot_receive_a_baked_curve(tmp_path):
    checkpoint = tmp_path / 'model.safetensors'
    save_file({'weight': torch.ones(1)}, str(checkpoint))
    with pytest.raises(ValueError, match='baked checkpoint'):
        module.curve_for_checkpoint(checkpoint, [tmp_path])
