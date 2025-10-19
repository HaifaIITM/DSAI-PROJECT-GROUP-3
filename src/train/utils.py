# src/train/utils.py
import os, hashlib, json, random
import numpy as np

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def dict_product(grid: dict):
    """Cartesian product over a dict of lists -> yields dicts."""
    import itertools
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def param_slug(params: dict, maxlen: int = 60) -> str:
    """Make a short, filesystem-safe experiment ID from params."""
    # stable JSON then hash to avoid super long names
    canon = json.dumps(params, sort_keys=True, default=str)
    h = hashlib.md5(canon.encode()).hexdigest()[:8]
    # include a few key fields for readability
    head = []
    for k in ["seq_len","hidden","d_model","channels","hidden_size","spectral_radius","leak_rate","ridge_alpha","epochs"]:
        if k in params:
            v = str(params[k]).replace(" ","").replace("(","").replace(")","").replace(",","_")
            head.append(f"{k}{v}")
    name = "_".join(head) if head else "exp"
    slug = f"{name}_{h}"
    return slug[:maxlen]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
