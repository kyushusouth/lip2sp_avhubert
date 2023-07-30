import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from fairseq import metrics, search
from fairseq.data import Dictionary, encoders
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING, II
import numpy as np
from argparse import Namespace

sys.path.append(str(Path('~/lip2sp_av').expanduser()))
from dataset import Lip2SpDataset


@dataclass
class Lip2SpTrainingConfig(FairseqDataclass):
    data_path: str = 'test'
    

@register_task('lip2sp_training', dataclass=Lip2SpTrainingConfig)
class Lip2SpTrainingTask(FairseqTask):

    cfg: Lip2SpTrainingConfig

    def __init__(
        self,
        cfg,    
    ):
        super().__init__(cfg)

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        return cls(cfg)
    
    def load_dataset(self, split, **kwargs):
        self.datasets[split] = Lip2SpDataset()