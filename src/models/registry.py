from .ridge_readout import RidgeReadout
from .esn import EchoStateNetwork
from .lstm import LSTMRegressor
from .transformer import TransformerRegressor
from .tcn import TCNRegressor
from .hybrid_esn_ridge import HybridESNRidge

MODEL_REGISTRY = {
    "ridge": RidgeReadout,
    "esn": EchoStateNetwork,
    "lstm": LSTMRegressor,
    "transformer": TransformerRegressor,
    "tcn": TCNRegressor,
    "hybrid": HybridESNRidge,
}

def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
