"""Restore runtime conditioning that cannot be stored in diffusion weights."""
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


def curve_for_checkpoint(source, directories):
    with safe_open(source, framework='pt', device='cpu') as handle:
        metadata = handle.metadata() or {}
    recipe = json.loads(metadata.get('exact_h3_bake', '{}'))
    if recipe.get('format') != 'h3-static-bake-v1' or not str(recipe.get('recipe', '')).startswith('hyperflow'):
        raise ValueError('Expected an H3 HyperFlow baked checkpoint')
    name = recipe.get('curve_fit_name', '')
    if not name or '/' in name or '\\' in name or name in ('.', '..'):
        raise ValueError('Invalid curve fit filename in checkpoint')
    for directory in directories:
        path = Path(directory) / name
        if path.is_file():
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest == recipe.get('curve_fit_sha256'):
                return path
    raise ValueError('The baked checkpoint requires its original HyperFlow curve fit')
