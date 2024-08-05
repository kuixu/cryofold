
from curses import keyname
from email.policy import strict
import logging
import os
from pathlib import Path

import random
from re import T
import sys
import time

import numpy as np
import pytorch_lightning as pl
import torch

from cryofold.config import model_config
from cryofold.data.data_modules import (
    CryoFoldDataModule,
    DummyDataLoader,
)
from cryofold.model.model import CryoFold
from cryofold.np import residue_constants
from cryofold.utils.exponential_moving_average import ExponentialMovingAverage
from cryofold.utils.loss import CryoFoldLoss, lddt_ca, get_pred_match
from cryofold.utils.lr_schedulers import CryoFoldLRScheduler
from cryofold.utils.superimposition import superimpose
from cryofold.utils.tensor_utils import tensor_tree_map
from cryofold.utils.validation_metrics import (
    drmsd,
    gdt_ts,
    gdt_ha,
    accuracy
)
from cryofold.utils.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
    get_global_step_from_zero_checkpoint
)

class Engine(pl.LightningModule):
    def __init__(self, config):
        super(Engine, self).__init__()
        self.config = config
        self.model = CryoFold(config)
        self.loss = CryoFoldLoss(config.loss)
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay
        )
        
        self.cached_weights = None
        self.last_lr_step = -1

        if self.config.model.freeze:
            self.model.freeze()
        if self.config.model.input_embedder.freeze:
            self.model.input_embedder.freeze()
        if self.config.model.template.freeze:
            self.model.template_angle_embedder.freeze()
            self.model.template_pair_embedder.freeze()
            self.model.template_pair_stack.freeze()
            self.model.template_pointwise_att.freeze()

        if self.config.model.extra_msa.freeze:
            self.model.extra_msa_embedder.freeze()
            self.model.extra_msa_stack.freeze()

        if self.config.model.evoformer_stack.freeze:
            self.model.evoformer.freeze()
        # cryoem modules
        if self.config.model.cryoem.enabled:
            if self.config.model.cryoem.cryoformer_embedder.freeze:
                self.model.cryoformer_emb.freeze()
                # self.model.cryo_sgl_norm.freeze()
            if self.config.model.cryoem.cryoformer_encoder.freeze:
                self.model.cryoformer_enc.freeze()
            if self.config.model.cryoem.cryoformer_decoder.freeze:
                self.model.cryoformer_dec.freeze()
        # backbone_frame
        if self.config.model.backbone_frame.enabled and \
            self.config.model.backbone_frame.freeze:
            self.model.backbone_frame.freeze()


        
        if self.config.model.structure_module.enabled and self.config.model.structure_module.freeze:
            self.model.structure_module.freeze()

        #     # self.model.input_embedder.unfreeze()
        #     self.model.recycling_embedder.unfreeze()
        #     if self.config.model.cryoem.enabled:
        #         self.model.cryoformer_emb.unfreeze()
        #         self.model.cryoformer_enc.unfreeze()
        #         self.model.cryoformer_dec.unfreeze()
        #         self.model.aux_heads.amino_acid.unfreeze()
        #         self.model.aux_heads.secondary_structure.unfreeze()
        #         self.model.aux_heads.cryoem_segmentation.unfreeze()
        #         self.model.backbone_frame.unfreeze()
        #         if self.config.model.structure_module.enabled:
        #             self.model.structure_module.unfreeze()


        

    def forward(self, batch):
        return self.model(batch, self.global_step)
    
    def _log(self, loss_breakdown, batch, outputs, train=True):
        # phase = "train" if train else "val"
        phase = "t" if train else "v"
        
        for loss_name, indiv_loss in loss_breakdown.items():
            # self.log(
            #     f"{phase}/{loss_name}", 
            #     indiv_loss, 
            #     on_step=train, on_epoch=(not train), prog_bar=True, logger=True,
            # )

            # if(train):
            self.log(
                f"{phase}_l/{loss_name}",
                indiv_loss,
                on_step=True, on_epoch=False, prog_bar=True, logger=True,
            )
        # import pdb;pdb.set_trace()

        with torch.no_grad():
            other_metrics = self._compute_validation_metrics(
                batch, 
                outputs,
                superimposition_metrics=True
            )

            # unrelaxed_protein = protein.from_prediction(
            #     features=batch,result=outputs, 
            #     b_factors=outputs['plddt'][0],
            #     chain_index=np.zeros_like(outputs['plddt'][0]),
            #     remark='cf',parents='',parents_chain_index='',)
            # with open("a.pdb", 'w') as fp:
            #     fp.write(protein.to_pdb(unrelaxed_protein))

        for k,v in other_metrics.items():
            self.log(
                f"{phase}_m/{k}", 
                v, 
                on_step=False, on_epoch=True, logger=True
            )

    def training_step(self, batch, batch_idx):
        if(self.ema.device != batch["aatype"].device):
            self.ema.to(batch["aatype"].device)

        # Run the model
        outputs = self(batch)

        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # Compute loss
        loss, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        # Log it
        self._log(loss_breakdown, batch, outputs)

        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        if(self.cached_weights is None):
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling 
            # load_state_dict().
            clone_param = lambda t: t.detach().clone()
            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict()["params"])
       
        # Run the model
        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # Compute loss and other metrics
        batch["use_clamped_fape"] = 0.
        _, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        self._log(loss_breakdown, batch, outputs, train=False)
        
    def validation_epoch_end(self, _):
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None


    def _compute_validation_metrics(self, 
        batch, 
        outputs, 
        superimposition_metrics=False
    ):
        metrics = {}
        gt_coords     = batch["all_atom_positions"]
        all_atom_mask = batch["all_atom_mask"]
        gt_coords_masked = gt_coords * all_atom_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = all_atom_mask[..., ca_pos]


        def compute_positions_metrics(pred_coords, pos_type='fl'):
            """

                pos_type: fl(final), bf(backbone frame)

            """
            # This is super janky for superimposition. Fix later
            pred_coords_masked = pred_coords * all_atom_mask[..., None]
            pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        
            lddt_ca_score = lddt_ca(
                pred_coords,
                gt_coords,
                all_atom_mask,
                eps=self.config.globals.eps,
                per_residue=False,
            )
    
            metrics["lddt_ca_"+pos_type] = lddt_ca_score
    
            drmsd_ca_score = drmsd(
                pred_coords_masked_ca,
                gt_coords_masked_ca,
                mask=all_atom_mask_ca, # still required here to compute n
            )
    
            metrics["drmsd_ca_"+pos_type] = drmsd_ca_score
        
            if(superimposition_metrics):
                try:
                    superimposed_pred, alignment_rmsd = superimpose(
                        gt_coords_masked_ca, pred_coords_masked_ca, all_atom_mask_ca,
                    )
                    gdt_ts_score = gdt_ts(
                        superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
                    )
                    gdt_ha_score = gdt_ha(
                        superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
                    )

                    metrics["drmsd_al_"+pos_type] = alignment_rmsd
                    metrics["gdt_ts_"+pos_type] = gdt_ts_score
                    metrics["gdt_ha_"+pos_type] = gdt_ha_score
                except:
                    metrics["drmsd_al_"+pos_type] = torch.tensor([10.0])
                    metrics["gdt_ts_"+pos_type] = torch.tensor([0.0])
                    metrics["gdt_ha_"+pos_type] = torch.tensor([0.0])
        
        if 'hungarian_matches' in outputs:
            seq_mask      = batch["seq_mask"]>0
            gt_sstype     = batch["sstype"][seq_mask]
            gt_aatype     = batch["aatype"][seq_mask]
            cs_mask       = batch['cryoem_seglabel'].flatten(1)<23
            gt_cstype     = batch['cryoem_seglabel'].flatten(1)[cs_mask]  
            
            match_indices = outputs["hungarian_matches"][-1]
            pred_ss_lgt   = outputs['secondary_structure_logits'][-1]
            pred_aa_lgt   = outputs['amino_acid_logits'][-1]
            pred_cs       = outputs['cryoem_segmentation_logits'].flatten(2).transpose(1,2)[cs_mask]
            pred_aa       = pred_aa_lgt[seq_mask] 
            pred_ss       = pred_ss_lgt[seq_mask]   
            # import pdb;pdb.set_trace()
            idx = get_pred_match(match_indices)
            N_bs = batch["aatype"].shape[0]
            # import pdb;pdb.set_trace()
            gt_aam = [batch["aatype"][i][seq_mask[i]] for i in range(N_bs) ]
            gt_ssm = [batch["sstype"][i][seq_mask[i]] for i in range(N_bs) ]
            gt_aatype_o = torch.cat([t[J] for t, (_, J) in zip(gt_aam, match_indices)])
            gt_sstype_o = torch.cat([t[J] for t, (_, J) in zip(gt_ssm, match_indices)])
            metrics["acc_aam"] = accuracy(pred_aa_lgt[idx], gt_aatype_o)[0]
            metrics["acc_ssm"] = accuracy(pred_ss_lgt[idx], gt_sstype_o)[0]
            metrics["acc_aa"]  = accuracy(pred_aa, gt_aatype)[0]
            metrics["acc_ss"]  = accuracy(pred_ss, gt_sstype)[0]
            metrics["acc_cs"]  = accuracy(pred_cs, gt_cstype)[0] 
            
        pred_coords_fl = outputs["final_atom_positions"][-1]
        compute_positions_metrics(pred_coords_fl, pos_type='fl')
        if 'backbone_frame' in outputs:
            pred_coords_bf = outputs["backbone_frame"][...,4:][:,:,:,None,:].repeat([1,1,1,37,1])[-1]
            compute_positions_metrics(pred_coords_bf, pos_type='bf')
            # import pdb;pdb.set_trace()
        

        
        
    
        return metrics

    def configure_optimizers(self, 
        learning_rate: float = 1e-3,
        eps: float = 1e-5,
    ) -> torch.optim.Adam:
#        return torch.optim.Adam(
#            self.model.parameters(),
#            lr=learning_rate,
#            eps=eps
#        )
        # Ignored as long as a DeepSpeed optimizer is configured
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            eps=eps
        )
        lr_scheduler = CryoFoldLRScheduler(
            optimizer, **self.config.lr_scheduler
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "CryoFoldLRScheduler",
            }
        }

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint["ema"]
        # if(not self.model.template_config.enabled):
        #     ema["params"] = {k:v for k,v in ema["params"].items() if not "template" in k}
        # self.ema.load_state_dict(ema)
        self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()
    
    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step

    def load_model(self, ckpt_path, resume_model_weights_only):
        print("Loading model: ", ckpt_path)

        if(os.path.isdir(ckpt_path)):  
            state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
            last_global_step = get_global_step_from_zero_checkpoint(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location='cpu')
            
            if 'global_step' in state_dict:
                last_global_step = int(state_dict['global_step'])
            else:
                last_global_step = -1
            
            if "ema" in state_dict:
                # The public weights have had this done to them already
                state_dict = state_dict["ema"]["params"]

        if not resume_model_weights_only:
            self.resume_last_lr_step(last_global_step)

        

        keyname = list(state_dict.keys())[0]
        if keyname.startswith('module.'):
            state_dict = {k[len("module."):]:v for k,v in state_dict.items()}
        keyname = list(state_dict.keys())[0]
        if keyname.startswith('model.'):
            # model.
            # self.load_state_dict(state_dict)
            self.load_state_dict(state_dict, strict=False)

            state_dict = {k[len("model."):]:v for k,v in state_dict.items()}
            # load into ema
            self.on_load_checkpoint({"ema":{"params":state_dict}})
            keyname = keyname[len("model."):]
        else:
            # load into ema
            self.on_load_checkpoint({"ema":{"params":state_dict}})
            state_dict = {"model."+k:v for k,v in state_dict.items()}
            # model.
            # self.load_state_dict(state_dict)
            self.load_state_dict(state_dict, strict=False)

        aa = self.ema.params[keyname]
        bb = self.model.state_dict()[keyname]
        torch.testing.assert_close(aa, bb)
        print("Model weights Loaded.")
        logging.info("Successfully loaded last lr step...")

        # import pdb;pdb.set_trace()

