import itertools
import logging
import os
import sys
import time
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from python_speech_features import logfbank
from scipy.io import wavfile


class Lip2SpDataset(FairseqDataset):
    def __init__(
        self,
    ):
        pass

    def __len__(self):
        return

    def __getitem__(self, index):
        return
    
    def collater(self, samples):
        return