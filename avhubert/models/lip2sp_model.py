import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import GradMultiply, LayerNorm
from fairseq import checkpoint_utils, options, tasks, utils
from copy import deepcopy


@dataclass
class Lip2SpConfig(FairseqDataclass):
    out_channels: int = 80


@register_model('lip2sp', dataclass=Lip2SpConfig)
class Lip2SpModel(BaseFairseqModel):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict
    
    @classmethod
    def build_model(cls, cfg, task):
        ckpt_path = '/home/minami/av_hubert_data/base_vox_433h.pt'
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        model = models[0]
        if hasattr(models[0], 'decoder'):
            print(f"Checkpoint: fine-tuned")
            model = models[0].encoder.w2v_model
        else:
            print(f"Checkpoint: pre-trained w/o fine-tuning")
        return model