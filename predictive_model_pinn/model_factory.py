import torch
from torch import nn

from predictive_model_pinn.models.model_attn import SinCosAttentionModel
from predictive_model_pinn.models.model_multi import MultiModel
from predictive_model_pinn.models.model_naive import NaiveModel
from predictive_model_pinn.models.model_phase import PhaseModel
from predictive_model_pinn.models.model_refiner import RefinerSinCosModel
from predictive_model_pinn.models.model_resid import ResidualModel
from predictive_model_pinn.models.model_resid_conv import ResNetPhasePredictor
from predictive_model_pinn.models.model_sin_cos import SinCosModel
from predictive_model_pinn.models.model_unet import ConvModel
from predictive_model_pinn.models.model_vae_v4 import PatchVAETransformer

# Mapping names to classes for easy selection in experiments
MODEL_MAP = {
    "SinCosModel": SinCosModel,
    "ConvModel": ConvModel,
    "SinCosAttentionModel": SinCosAttentionModel,
    "RefinerSinCosModel": RefinerSinCosModel,
    "ResidualModel": ResidualModel,
    "MultiModel": MultiModel,
    "PatchVAETransformer": PatchVAETransformer,
    "NaiveModel": NaiveModel,
    "ResNetPhasePredictor": ResNetPhasePredictor,
    "PhaseModel": PhaseModel,
}


def get_model_by_name(name, **kwargs):
    if name not in MODEL_MAP:
        raise ValueError(
            f"Model {name} not found in MODEL_MAP. Available: {list(MODEL_MAP.keys())}"
        )
    return MODEL_MAP[name](**kwargs)
