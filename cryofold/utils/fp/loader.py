import torch
import random
import numpy as np
from cryofold.utils.fp.ema import ExponentialMovingAverage
import yaml
from easydict import EasyDict as edict

def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = 'cpu'
    return device

def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema


def load_checkpoint(model, train_cfg, ckpt_dict):
    ema = load_ema(model, decay=train_cfg.train.ema)

    if 'module' in list(ckpt_dict['state_dict'].keys())[0]:  # if trained with DDP
        ckpt_dict['state_dict'] = {k[7:]: v for k, v in ckpt_dict['state_dict'].items()}
    model.load_state_dict(ckpt_dict['state_dict'])
    ema.load_state_dict(ckpt_dict['ema'])

    return model, ema

def load_config(config, seed, inference=False):
    if inference:
        folder_name = 'inference'
    config_str = """
mode: vf

data:
  data: bc40
  train_path:
  cluster_path:
  test_path: './test'
  min_length: 40
  max_length: 512
  edge_type: knn
  max_radius: 16.0
  max_neighbors: 30

ckpt: 'params/bc40.pth'
conf_ckpt: 'params/confidence.pth'
seed: 1234
sample:
  batch_size: 1
  n_samples: 1
  use_ema: True
  eps: 2.0e-3
  save_trajectory: False
  coeff: 5.0
  num_steps: 10

"""
    # else:
    #     folder_name = 'training'
    # # config_dir = f'./config/{folder_name}/{config}.yaml'
    # config = edict(yaml.safe_load(config_str))
    config = edict(yaml.safe_load(config_str), Loader=yaml.FullLoader)
    # breakpoint()
    # config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    
    config.seed = seed


    return config