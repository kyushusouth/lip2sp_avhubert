import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@register_criterion('lip2sp')
class Lip2SpCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
    ):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, log_pred=False):
        loss = 1
        sample_size = 10
        logging_output = {
            'loss': loss if reduce else loss,
        }
        return loss, sample_size, logging_output