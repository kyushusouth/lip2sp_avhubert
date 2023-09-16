import sys
from pathlib import Path
sys.path.append(str(Path('~/lip2sp_avhubert/avhubert').expanduser()))

import cv2
import tempfile
import torch
import utils as avhubert_utils
from argparse import Namespace
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
from IPython.display import HTML
import inspect

import hubert_asr

from avhubert_load import MyAVHubertModel
from collections import OrderedDict


def main():
    ckpt_path = '/home/minami/av_hubert_data/base_vox_433h.pt'
    mymodel = MyAVHubertModel()
    state = checkpoint_utils.load_checkpoint_to_cpu(ckpt_path)
    pretrained_dict = state['model']
    mymodel_dict = mymodel.state_dict()
    mymodel_dict = {'encoder.w2v_model.' + key: value for key, value in mymodel_dict.items()}
    match_dict = {k: v for k, v in pretrained_dict.items() if k in mymodel_dict}
    match_dict = {key.replace('encoder.w2v_model.', ''): value for key, value in match_dict.items()}
    mymodel.load_state_dict(match_dict, strict=True)
    mymodel.cuda()
    mymodel.eval()