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
from pathlib import Path

import hubert_asr

from avhubert_copied import MyAVHubertModel, Config
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


def convert_to_path(path_or_str):
    if isinstance(path_or_str, str):
        return Path(path_or_str)
    elif isinstance(path_or_str, Path):
        return path_or_str
    

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def load_pretrained_model(model_path, model, model_name):
    """
    学習したモデルの読み込み
    現在のモデルと事前学習済みのモデルで一致した部分だけを読み込むので,モデルが変わっていても以前のパラメータを読み込むことが可能
    """
    model_path = convert_to_path(model_path)
    model_dict = model.state_dict()

    if model_path.suffix == ".ckpt":
        if torch.cuda.is_available():
            pretrained_dict = torch.load(str(model_path))[str(model_name)]
        else:
            pretrained_dict = torch.load(str(model_path), map_location=torch.device('cpu'))[str(model_name)]
    elif model_path.suffix == ".pth":
        if torch.cuda.is_available():
            pretrained_dict = torch.load(str(model_path))
        else:
            pretrained_dict = torch.load(str(model_path), map_location=torch.device('cpu'))

    pretrained_dict = fix_model_state_dict(pretrained_dict)
    match_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(match_dict)
    model.load_state_dict(match_dict)
    return model


def load_avhubert_torch(ckpt_path, model_size):
    ckpt_path = '/home/minami/av_hubert_data/base_vox_iter5_torch.ckpt'
    cfg = Config(model_size)
    avhubert = MyAVHubertModel(cfg)
    avhubert = load_pretrained_model(ckpt_path, avhubert, 'avhubert')
    return avhubert


def extract_visual_feature(video_path, ckpt_path, finetuned, model_size):
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
    model.cuda()
    model.eval()

    mymodel = load_avhubert(ckpt_path, finetuned=finetuned, model_size=model_size)
    mymodel.cuda()
    mymodel.eval()
    count_params(mymodel, 'avhubert')
    count_params(mymodel.feature_extractor_video, 'feature_extractor_video')
    count_params(mymodel.encoder, 'transformer_encoder')

    mymodel_torch = load_avhubert_torch(ckpt_path, model_size=model_size)
    mymodel_torch.cuda()
    mymodel_torch.eval()

    with torch.no_grad():
        frames = frames.expand(8, -1, -1, -1, -1)
        print(f'input_frame : {frames.shape}')
        feature, padding_mask, res_output = model.extract_finetune(source={'video': frames, 'audio': None}, padding_mask=None, output_layer=None, return_res_output=True)

        return_res_output = True
        my_feature = mymodel(frames, padding_mask=None, output_layer=None, return_res_output=return_res_output)
        my_feature_torch = mymodel_torch(frames, padding_mask=None, output_layer=None, return_res_output=return_res_output)

        print(feature.shape, res_output.shape)
        print(my_feature.shape)
        print(my_feature_torch.shape)

        if return_res_output:
            print(torch.equal(res_output, my_feature))
            print(torch.equal(res_output, my_feature_torch))
        else:
            print(torch.equal(feature, my_feature))
            print(torch.equal(feature, my_feature_torch))
    
    return feature, res_output


def main():
    video_path = '/home/minami/av_hubert_data/roi.mp4'
    # ckpt_path = '/home/minami/av_hubert_data/base_vox_433h.pt'      # finetuning
    ckpt_path = '/home/minami/av_hubert_data/base_vox_iter5.pt'     # pretrained
    # ckpt_path = '/home/minami/av_hubert_data/large_vox_iter5.pt'    # pretrained
    feature, res_output = extract_visual_feature(video_path, ckpt_path, finetuned=False, model_size='base')


if __name__ == '__main__':
    main()