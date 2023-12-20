import torch 
import torch.nn as nn

from .vit import (
    _make_pretrained_vitb_rn50_384,
    _make_pretrained_vitl16_384,
    _make_pretrained_vitb16_384,
    forward_vit,
)

def _make_encoder(
        backbone,
        features,
        use_pretrained,
        groups=1,
        expand=False,
        exportable=True,
        hooks=None,
        use_vit_only=False,
        use_readout="ignore",
        enable_attention_hooks=False,
):
    
    if backbone == "resnext101_wsl":
        pretrained = make_pretrained_resnext101_wsl(use_pretrained)
        scratch = _make_scratch(
            [256.512,1024,2048], features, groups=groups, expand=expand
        )

    elif backbone=="vitl16_384":
