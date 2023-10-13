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

from avhubert_copied import MyAVHubertModel, Config
from collections import OrderedDict


def load_avhubert(ckpt_path, finetuned, model_size):
    cfg = Config(model_size)
    avhubert = MyAVHubertModel(cfg)
    if finetuned:
        state = torch.load(ckpt_path, map_location=torch.device("cpu"))
        pretrained_dict = state['model']
        avhubert_dict = avhubert.state_dict()
        avhubert_dict = {'encoder.w2v_model.' + key: value for key, value in avhubert_dict.items()}
        match_dict = {k: v for k, v in pretrained_dict.items() if k in avhubert_dict}
        match_dict = {key.replace('encoder.w2v_model.', ''): value for key, value in match_dict.items()}
        avhubert.load_state_dict(match_dict, strict=True)
    else:
        state = torch.load(ckpt_path, map_location=torch.device("cpu"))
        pretrained_dict = state['model']
        avhubert_dict = avhubert.state_dict()
        match_dict = {k: v for k, v in pretrained_dict.items() if k in avhubert_dict}
        avhubert.load_state_dict(match_dict, strict=True)
    return avhubert


def main():
    # ckpt_path = '/home/minami/av_hubert_data/base_vox_433h.pt'      # finetuning
    # ckpt_path = '/home/minami/av_hubert_data/base_vox_iter5.pt'     # pretrained
    ckpt_path = '/home/minami/av_hubert_data/large_vox_iter5.pt'    # pretrained

    avhubert = load_avhubert(ckpt_path, finetuned=False, model_size='large')

    ckpt_path_new = Path(ckpt_path)
    new_filename = ckpt_path_new.stem + '_torch'
    ckpt_path_new = str(ckpt_path_new).replace(ckpt_path_new.stem, new_filename).replace(ckpt_path_new.suffix, '.ckpt')
    torch.save({'avhubert': avhubert.state_dict(),}, ckpt_path_new)
    

if __name__ == '__main__':
    main()