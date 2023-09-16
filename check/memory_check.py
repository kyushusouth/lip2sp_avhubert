import torch
from avhubert_copied import MyAVHubertModel, Config


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
    ckpt_path = '/home/minami/av_hubert_data/base_vox_iter5.pt'     # pretrained
    avhubert = load_avhubert(ckpt_path, finetuned=False, model_size='base')
    optimizer = torch.optim.Adam(
        params=avhubert.parameters(),
        lr=0.001, 
        betas=(0.9, 0.98),
        weight_decay=1.0e-2,   
    )
    avhubert.cuda()
    avhubert.train()

    for i in range(100):
        print(i)
        batch_size = 8
        t = 250
        hw = 88
        dummy_data = torch.rand(batch_size, 1, t, hw, hw).cuda()
        feature = avhubert(dummy_data, padding_mask=None, return_res_output=True)
        dummy_feature = torch.rand_like(feature)
        loss = torch.mean((feature - dummy_feature) ** 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':
    main()