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


def count_params(module, attr):
    """
    モデルパラメータを計算
    """
    params = 0
    for p in module.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"{attr}_parameter = {params}")


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def extract_visual_feature(video_path, ckpt_path):
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    transform = avhubert_utils.Compose([
        avhubert_utils.Normalize(0.0, 255.0),
        avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
        avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std)])
    frames = avhubert_utils.load_video(video_path)
    print(f"Load video {video_path}: shape {frames.shape}")
    frames = transform(frames)
    print(f"Center crop video to: {frames.shape}")
    frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
    model = models[0]
    if hasattr(models[0], 'decoder'):
        print(f"Checkpoint: fine-tuned")
        model = models[0].encoder.w2v_model
    else:
        print(f"Checkpoint: pre-trained w/o fine-tuning")

    count_params(model, 'model')
    for x in inspect.getmembers(model, inspect.ismethod):
        print(x[0])
    
    model.cuda()
    model.eval()

    # これでいける
    mymodel = MyAVHubertModel()
    state = checkpoint_utils.load_checkpoint_to_cpu(ckpt_path)
    pretrained_dict = state['model']
    pretrained_dict = fix_model_state_dict(pretrained_dict)
    mymodel_dict = mymodel.state_dict()
    mymodel_dict = {'encoder.w2v_model.' + key: value for key, value in mymodel_dict.items()}
    match_dict = {k: v for k, v in pretrained_dict.items() if k in mymodel_dict}
    match_dict = {key.replace('encoder.w2v_model.', ''): value for key, value in match_dict.items()}
    mymodel.load_state_dict(match_dict, strict=True)
    mymodel.cuda()
    mymodel.eval()

    with torch.no_grad():
        # Specify output_layer if you want to extract feature of an intermediate layer
        # model has extract_finetune and extract_features
        # extract_features causes error.
        frames = frames.expand(8, -1, -1, -1, -1)
        print(f'input_frame : {frames.shape}')
        feature, padding_mask, res_output = model.extract_finetune(source={'video': frames, 'audio': None}, padding_mask=None, output_layer=None, return_res_output=True)
        my_feature, my_padding_mask, my_res_output = mymodel(frames, padding_mask=None, output_layer=None, return_res_output=True)
        print(feature.shape, res_output.shape)
        print(my_feature.shape, my_res_output.shape)
        print(torch.equal(feature, my_feature))
        print(torch.equal(res_output, my_res_output))
    
    return feature, res_output


def main():
    video_path = '/home/minami/av_hubert_data/roi.mp4'
    ckpt_path = '/home/minami/av_hubert_data/base_vox_433h.pt'
    feature, res_output = extract_visual_feature(video_path, ckpt_path)


if __name__ == '__main__':
    main()