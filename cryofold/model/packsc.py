import os
import time
import torch

# from utils.train_utils import count_parameters
from pathlib import Path
# from dataset_loader import get_feature_pdb
import numpy as np
import math
import shutil

from cryofold.utils.fp.constants import chi_mask as chi_mask_true
from cryofold.utils.fp.constants import atom14_mask as atom_mask_true
from cryofold.utils.fp.loader import load_seed, load_device, load_ema, load_checkpoint, load_config
from cryofold.utils.fp.metrics import metrics_per_chi, atom_rmsd, count_clashes
from cryofold.utils.fp.structure_utils import create_structure_from_crds, get_feature_pdb
from cryofold.utils.fp.sidechain_utils import Idealizer
from cryofold.diff.structure.protein_chain import ProteinChain
from cryofold.model.fp.cnf import CNF
from cryofold.model.fp.confidence import Confidence
from cryofold.model.fp.equiformer_v2 import EquiformerV2

class Sampler(object):
    def __init__(self, config, use_gt_masks=True, ddp=False, in_path=None, out_path=None):
        super(Sampler, self).__init__()
        self.config = config
        self.use_gt_masks = use_gt_masks
        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        # self.train_loader, self.test_loader, _, _ = get_dataloader(self.config, ddp=ddp, sample=True)
        self.idealizer = Idealizer(use_native_bb_coords=True)
        self.in_path = in_path
        self.out_path = out_path

    def sample(self, ts, save_traj=False):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'

        # print(f'{self.ckpt}')
        ckpt_dict = torch.load(self.config.ckpt, weights_only=False)
        train_cfg = ckpt_dict['config']
        # self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(train_cfg)
        # print(self.config.sample)
        self.model = CNF(EquiformerV2(**train_cfg.model), train_cfg, coeff=self.config.sample.coeff,
                         stepsize=self.config.sample.num_steps, mode=self.config.mode).cuda()
        # self.model = torch.compile(self.model)
        # print(f'Number of parameters: {count_parameters(self.model)}')
        self.model, self.ema = load_checkpoint(self.model, train_cfg, ckpt_dict)
        self.model.eval()
        self.ema.copy_to(self.model.parameters())

        if self.config.conf_ckpt is not None:
            conf_ckpt = torch.load(self.config.conf_ckpt, weights_only=False)
            self.conf_model = Confidence(EquiformerV2(**conf_ckpt['config'].model), conf_ckpt['config']).cuda()
            if 'module.' in list(conf_ckpt["state_dict"].keys())[0]:
                state_dict = {k[7:]: v for k, v in conf_ckpt["state_dict"].items()}
            self.conf_model.load_state_dict(state_dict)

        output_dict = {}
        with torch.no_grad():
            batch = get_feature_pdb(self.in_path)
            # for batch in tqdm(self.test_loader):
            # batch = batch.to(f'cuda:{self.device[0]}')
            batch = batch.cuda()
            aa_str, aa_onehot, aa_num, coords, mask, atom_mask, batch_id, pdb_codes = batch.aa_str, batch.aa_onehot, batch.aa_num, \
                                                                                        batch.pos, batch.aa_mask, batch.atom_mask, batch.batch, batch.id
            chi, chi_alt, chi_mask = batch.chi, batch.chi_alt, batch.chi_mask
            bb_coords = coords[:, :4]
            batch_id = torch.zeros_like(bb_coords[:, 0,0]).long()
            # batch_id = 0
            # print("batch_id: ", batch_id)
            # breakpoint()
            if self.use_gt_masks:
                chi_mask = chi_mask_true.to(aa_num)
                chi_mask = chi_mask[aa_num]
                batch.chi_mask = chi_mask

                atom_mask = atom_mask_true.to(atom_mask)
                atom_mask = atom_mask[aa_num]
                batch.atom_mask = atom_mask

            chi = (chi + math.pi) * chi_mask
            chi_alt = (chi_alt + math.pi) * chi_mask

            batch_size = batch_id.max().item() + 1
            # batch_size = 1
            output_dict = {**output_dict, **{i:{} for i in pdb_codes}}

            with torch.no_grad():
                best_pred_idx, best_pred_rmsd, best_gt_idx, best_gt_rmsd = 0, 999, 0, 999
                for sample_idx in range(self.config.sample.n_samples):
                    # check if files exist
                    exists = True
                    # pdb_path = self.out_path.replace(".pdb", f"_run_{sample_idx+1}.pdb")
                    pred_sc, dropout = self.model.decode(batch, return_traj=save_traj)
                    pred_sc = (pred_sc - math.pi) * chi_mask  # shift torsions back to [-pi,pi]

                    if save_traj:
                        pred_sc_traj = pred_sc.clone()
                        pred_sc = pred_sc[-1]

                    all_atom_coords = self.idealizer(aa_num, bb_coords, pred_sc) * atom_mask.unsqueeze(-1)
                    gt_idealized = self.idealizer(aa_num, bb_coords, chi-math.pi) * atom_mask.unsqueeze(-1)
                    pred_sc = (pred_sc + math.pi) * chi_mask
                    
                    seq = ''.join(aa_str)
                    pc = ProteinChain.from_atom14(all_atom_coords.cpu(),sequence = seq)
                    pc.to_pdb(str(self.out_path))
                   
        return pc

def pack_sidechain(pdb_path, out_path, seed=42):
    config = load_config("base", seed=seed, inference=True)
    sampler = Sampler(config, in_path=pdb_path, out_path=out_path)
    sampler.sample(time.strftime('%b%d-%H:%M:%S', time.gmtime()), save_traj=False)
    

if __name__ == '__main__':
    import argparse

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, default='test/a.pdb')
    parser.add_argument('out_path', type=str, default='test/ao.pdb')
    parser.add_argument('--save_traj', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    config = load_config("base", seed=args.seed, inference=True)
    sampler = Sampler(config, in_path=args.in_path, out_path=args.out_path)
    sampler.sample(time.strftime('%b%d-%H:%M:%S', time.gmtime()), save_traj=args.save_traj)
    # print(f'Inference took a total of {time.time() - start_time:.2f} seconds')