from typing import Callable

import torch
import torch.nn as nn

from cryofold.model.sm import CF
from cryofold.model.sed import (
    StructureDecoder,
    StructureEncoder,
)
from cryofold.tokens import get_model_tokenizers
# from cryofold.diff.constants.models import (
#     CF_MODEL, CF_STRUCTURE_ENCODER, CF_STRUCTURE_DECODER
# )
ModelBuilder = Callable[[torch.device | str], nn.Module]

CF_MODEL = "cf_v1.0"
CF_STRUCTURE_ENCODER = "cf_structure_encoder"
CF_STRUCTURE_DECODER = "cf_structure_decoder"



weights_path = "data/weights/cf_v1.0.pth"
weights_path = "data/weights/cf_v1.pth"
weights1_path = "data/weights/cf_v1.pth"
weights1_path = "data/weights/cf_structure_encoder.pth"
weights2_path = "data/weights/cf_v1.pth"
weights2_path = "data/weights/cf_structure_decoder.pth"



weights_path  = "params/cf_v1.pth"
weights1_path = "params/cf_structure_encoder.pth"
weights2_path = "params/cf_structure_decoder.pth"


# "data/weights/structure_encoder.pth"
# "data/weights/structure_decoder.pth"
# "data/weights/cf_v1.pth"
def structure_encoder(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        ).eval()
    state_dict = torch.load(
         weights1_path, map_location=device, weights_only=False
    )
    model.load_state_dict(state_dict, strict=False)
    return model


def structure_decoder(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    state_dict = torch.load(
        weights2_path , map_location=device, weights_only=False
    )
    model.load_state_dict(state_dict, strict=False)
    return model


def cf_model(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = CF(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_fn=structure_encoder,
            structure_decoder_fn=structure_decoder,
            tokenizers=get_model_tokenizers(),
        ).eval()
    state_dict = torch.load(
        weights_path , map_location=device, weights_only=False
    )
    model.load_state_dict(state_dict, strict=False)
    return model


LOCAL_MODEL_REGISTRY: dict[str, ModelBuilder] = {
    CF_MODEL: cf_model,
    # CF_STRUCTURE_ENCODER: structure_encoder,
    # CF_STRUCTURE_DECODER: structure_decoder,
}


def load_local_model(
    model_name: str, device: torch.device = torch.device("cpu")
) -> nn.Module:
    if model_name not in LOCAL_MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in local model registry.")
    return LOCAL_MODEL_REGISTRY[model_name](device)


def register_local_model(model_name: str, model_builder: ModelBuilder) -> None:
    LOCAL_MODEL_REGISTRY[model_name] = model_builder
