
import torch
import torch.nn as nn

from scipy.optimize import linear_sum_assignment
import pytorch_lightning as pl

from cryofold.utils.rigid_utils import Rotation
from cryofold.model.primitives import (
    Linear, 
    LayerNorm, 
    MLP, 
    MLP3D, 
    LayerNorm3d,
    TransposeConvUpsampling 
)
from cryofold.utils.loss import (
    compute_plddt,
    compute_tm,
    compute_predicted_aligned_error,
)


class AuxiliaryHeads(nn.Module):
    def __init__(self, config):
        super(AuxiliaryHeads, self).__init__()

        if config.lddt.enabled:
            self.plddt = PerResidueLDDTCaPredictor(
                **config["lddt"],
            )
        
        self.distogram = DistogramHead(
            **config["distogram"],
        )

        self.masked_msa = MaskedMSAHead(
            **config["masked_msa"],
        )

        self.experimentally_resolved = ExperimentallyResolvedHead(
            **config["experimentally_resolved"],
        )

        if config.amino_acid.enabled:
            self.amino_acid = AminoAcidHead(
                **config["amino_acid"],
            )

        if config.secondary_structure.enabled:
            self.secondary_structure = SecondaryStructureHead(
                **config["secondary_structure"],
            )
        
        if config.hungarian_matcher.enabled:
            self.hungarian_matcher = HungarianMatcherHead(
                **config["hungarian_matcher"],
            )
        
        if config.cryoem_segmentation.enabled:
            self.cryoem_segmentation = CryoEMSegmentationHead(
                **config.cryoem_segmentation,
            )
        
        if config.tm.enabled:
            self.tm = TMScoreHead(
                **config.tm,
            )

        self.config = config

    def forward(self, outputs, batch):
        aux_out = {}
        if 'backbone_frame' in outputs:
            aux_out["backbone_frame"] = outputs['backbone_frame']
            aux_out["normed_backbone_frame"] = outputs['normed_backbone_frame']

        if self.config.lddt.enabled:
            lddt_logits = self.plddt(outputs["sm"]["single"])
            aux_out["lddt_logits"] = lddt_logits

            # Required for relaxation later on
            aux_out["plddt"] = compute_plddt(lddt_logits)

        distogram_logits = self.distogram(outputs["pair"])
        aux_out["distogram_logits"] = distogram_logits

        masked_msa_logits = self.masked_msa(outputs["msa"])
        aux_out["masked_msa_logits"] = masked_msa_logits

        experimentally_resolved_logits = self.experimentally_resolved(
            outputs["single"]
        )
        aux_out[
            "experimentally_resolved_logits"
        ] = experimentally_resolved_logits

        if self.config.amino_acid.enabled:
            amino_acid_logits = self.amino_acid(
                outputs["single_dec"]
            )
            aux_out[
                "amino_acid_logits"
            ] = amino_acid_logits

        if self.config.secondary_structure.enabled:
            secondary_structure_logits = self.secondary_structure(
                outputs["single_dec"]
            )
            aux_out[
                "secondary_structure_logits"
            ] = secondary_structure_logits

        if self.config.hungarian_matcher.enabled:
            hungarian_matches = self.hungarian_matcher(
                aux_out, batch
            )
            aux_out[
                "hungarian_matches"
            ] = hungarian_matches
        
        
            
        if self.config.cryoem_segmentation.enabled:
            cryoem_segmentation_logits = self.cryoem_segmentation(
                outputs["cryoem_density"], outputs["density_feats"],
                outputs["density_repr"], outputs["cryoem_size"],
            )
            aux_out[
                "cryoem_segmentation_logits"
            ] = cryoem_segmentation_logits
            
        

        if self.config.tm.enabled:
            tm_logits = self.tm(outputs["pair"])
            aux_out["tm_logits"] = tm_logits
            aux_out["predicted_tm_score"] = compute_tm(
                tm_logits, **self.config.tm
            )
            aux_out.update(
                compute_predicted_aligned_error(
                    tm_logits,
                    **self.config.tm,
                )
            )

        return aux_out



class CryoEMSegmentationHead(pl.LightningModule):

    """Head to predict Density Semantic class.
    
    """
    
    def __init__(self, hidden_dim=64, hidden_dim_cf=384, c_out=23, dropout=0.1, 
                 patch_c1=512, patch_c2=2048, **kwargs):
        super().__init__()
        l2_channels = patch_c1
        l4_channels = patch_c2
        cf_channels = hidden_dim_cf

        self.dropout = nn.Dropout3d(dropout)

        self.cf = MLP3D(input_dim=cf_channels, embed_dim=hidden_dim)
        self.l2 = MLP3D(input_dim=l2_channels, embed_dim=hidden_dim)
        self.l4 = MLP3D(input_dim=l4_channels, embed_dim=hidden_dim)

        self.up1 = TransposeConvUpsampling(
            in_channels=hidden_dim, out_channels=hidden_dim,
            kernel_size=(3,3,3), scale_factor=(2, 2, 2))
        
        self.up2 = TransposeConvUpsampling(
            in_channels=hidden_dim, out_channels=hidden_dim,
            kernel_size=(3,3,3), scale_factor=(2, 2, 2))
        
        self.up3 = TransposeConvUpsampling(
            in_channels=hidden_dim, out_channels=hidden_dim,
            kernel_size=(3,3,3), scale_factor=(2, 2, 2))

        self.proj = nn.Conv3d(1,  hidden_dim, kernel_size=1)
        self.fuse = nn.Conv3d(hidden_dim*3, hidden_dim, kernel_size=1)
        self.pred = nn.Conv3d(hidden_dim, c_out, kernel_size=1)
        self.norm = LayerNorm3d(hidden_dim)
        
    
    def forward(self, cryoem_density, density_feats, density_repr, cryoem_size):
        """Builds AffineHead module.
        Arguments
        -------------
            cryoem_density: [*, C, W, H, L]
            density_feats: list[
                                [*, C, W, H, L],
                                [*, C, W, H, L]
            density_repr: [*, C, W, H, L]
            cryoem_size: [*, 3]
        
        Return
        -------------
            logits: [*, class, W, H, L]
        """

        d2 = density_feats[0]
        d4 = density_feats[1]
        # import pdb;pdb.set_trace()
        if d2.shape[0]>1:
            assert (cryoem_size[0]==cryoem_size[1]).all()
        # import pdb;pdb.set_trace()
        if len(cryoem_size.shape)>1:
            cryoem_size = cryoem_size[0]
        de = density_repr
        N_bs = de.shape[0]
        sz = (int(cryoem_size[0].item()), int(cryoem_size[1].item()), int(cryoem_size[2].item()))

        cf = self.cf(de).permute(0,2,1).reshape(N_bs, -1, de.shape[2], de.shape[3], de.shape[4])
        cf = self.up1(cf, (d2.shape[2], d2.shape[3], d2.shape[4]))

        d4 = self.l4(d4).permute(0,2,1).reshape(N_bs, -1, d4.shape[2], d4.shape[3], d4.shape[4])
        d4 = self.up2(d4, (d2.shape[2], d2.shape[3], d2.shape[4]))
        d2 = self.l2(d2).permute(0,2,1).reshape(N_bs, -1, d2.shape[2], d2.shape[3], d2.shape[4])

        x = self.fuse(torch.cat([cf, d4, d2], dim=1))
        x = self.up3(x.contiguous(), sz)
        x = self.dropout(x)
        x = self.norm(x + self.proj(cryoem_density))
        x = self.pred(x)
        return x


class HungarianMatcherHead(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_aa: float = 1, cost_ss: float = 2, cost_ca: float = 5, **kwargs):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_aa = cost_aa
        self.cost_ss = cost_ss
        self.cost_ca = cost_ca
        assert cost_aa != 0 or cost_ss != 0 or cost_ca != 0, "all costs cant be 0"

    
    @torch.no_grad()
    def forward_bs1(self, outputs, batch):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            batch: This is a list of batch (len(batch) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected batch (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        
        # if len(outputs["amino_acid_logits"].shape) ==2:
        #     bs = 1
        #     num_queries = outputs["amino_acid_logits"].shape[:1]
            
        # else:
        #     bs, num_queries = outputs["amino_acid_logits"].shape[:2]
        # pred_aa = outputs["amino_acid_logits"][:,0,:,:20]
        # pred_ss = outputs["secondary_structure_logits"][:,0,:,:4]
        # pred_ca = outputs["normed_backbone_frame"][:,0,:,4:].float()

        pred_aa = outputs["amino_acid_logits"].flatten(0, 1)[:,:,:20]
        pred_ss = outputs["secondary_structure_logits"].flatten(0, 1)[:,:,:4]
        pred_ca = outputs["normed_backbone_frame"].flatten(0, 1)[:,:,4:].float()
        bs, num_queries = pred_aa.shape[:2]
        # import pdb;pdb.set_trace()
        if len(batch["tgt_aa"].shape) ==2:
            tgt_aa_ = batch["tgt_aa"][:,0]
            tgt_ss_ = batch["tgt_ss"][:,0]
            tgt_ca_ = batch["tgt_ca"][:,:,0].to(pred_ca.dtype)
        elif len(batch["tgt_aa"].shape) ==3:
            tgt_aa_ = batch["tgt_aa"][0,:,0]
            tgt_ss_ = batch["tgt_ss"][0,:,0]
            tgt_ca_ = batch["tgt_ca"][0,:,:,0].float()
        else:
            import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        # We flatten to compute the cost matrices in a batch
        # out_aa = outputs["amino_acid_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # out_ss = outputs["secondary_structure_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # out_ca = outputs["normed_backbone_frame"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_aa = pred_aa.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_ss = pred_ss.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_ca = pred_ca.flatten(0, 1)  # [batch_size * num_queries, 4]
        # Also concat the target labels and boxes
        # import pdb;pdb.set_trace()

        tgt_aa = torch.cat([tgt_aa_]*bs)
        tgt_ss = torch.cat([tgt_ss_]*bs)
        tgt_ca = torch.cat([tgt_ca_]*bs)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.

        cost_aa = -out_aa[:, tgt_aa]
        cost_ss = -out_ss[:, tgt_ss]

        # Compute the L1 cost between boxes
        cost_ca = torch.cdist(out_ca, tgt_ca, p=1)

        # Compute the giou cost betwen boxes

        # Final cost matrix
        C = self.cost_ca * cost_ca + self.cost_aa * cost_aa + self.cost_ss * cost_ss
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(tgt_ca_)]*bs

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        matches = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        # import pdb;pdb.set_trace()
       
        return matches

    @torch.no_grad()
    def forward(self, outputs, batch):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [dec_size, batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [dec_size, batch_size, num_queries, 4] with the predicted box coordinates
            batch: This is a list of batch (len(batch) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected batch (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        N_dec = outputs["amino_acid_logits"].shape[0]
        matches = []
        for i in range(N_dec):
            matches.append( self.forward_single_dec(outputs, batch, i))
        return matches


    def forward_single_dec0(self, outputs, batch, i):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [dec_size, batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [dec_size, batch_size, num_queries, 4] with the predicted box coordinates
            batch: This is a list of batch (len(batch) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected batch (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        
        # if len(outputs["amino_acid_logits"].shape) ==2:
        #     bs = 1
        #     num_queries = outputs["amino_acid_logits"].shape[:1]
            
        # else:
        #     bs, num_queries = outputs["amino_acid_logits"].shape[:2]
        # pred_aa = outputs["amino_acid_logits"][:,0,:,:20]
        # pred_ss = outputs["secondary_structure_logits"][:,0,:,:4]
        # pred_ca = outputs["normed_backbone_frame"][:,0,:,4:].float()
        # N_dec, N_bs, N_res, _ = outputs["amino_acid_logits"].shape
        pred_aa = outputs["amino_acid_logits"][i,:,:,:20]
        pred_ss = outputs["secondary_structure_logits"][i,:,:,:4]
        pred_ca = outputs["normed_backbone_frame"][i,:,:,4:].float()
        if torch.isnan(pred_aa).any():
            import pdb;pdb.set_trace()

        # pred_aa = outputs["amino_acid_logits"].flatten(0, 1)[:,:,:20]
        # pred_ss = outputs["secondary_structure_logits"].flatten(0, 1)[:,:,:4]
        # pred_ca = outputs["normed_backbone_frame"].flatten(0, 1)[:,:,4:].float()
        bs, num_queries = pred_aa.shape[:2]
        # import pdb;pdb.set_trace()
        if len(batch["tgt_aa"].shape) ==2:
            tgt_aa_ = batch["tgt_aa"][:,0]
            tgt_ss_ = batch["tgt_ss"][:,0]
            tgt_ca_ = batch["tgt_ca"][:,:,0].to(pred_ca.dtype)
        elif len(batch["tgt_aa"].shape) ==3:
            tgt_aa_ = batch["tgt_aa"][0,:,0]
            tgt_ss_ = batch["tgt_ss"][0,:,0]
            tgt_ca_ = batch["tgt_ca"][0,:,:,0].float()
        else:
            import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        # We flatten to compute the cost matrices in a batch
        # out_aa = outputs["amino_acid_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # out_ss = outputs["secondary_structure_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # out_ca = outputs["normed_backbone_frame"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_aa = pred_aa.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_ss = pred_ss.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_ca = pred_ca.flatten(0, 1)  # [batch_size * num_queries, 4]
        # Also concat the target labels and boxes
        # import pdb;pdb.set_trace()

        tgt_aa = torch.cat([tgt_aa_]*bs)
        tgt_ss = torch.cat([tgt_ss_]*bs)
        tgt_ca = torch.cat([tgt_ca_]*bs)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.

        cost_aa = -out_aa[:, tgt_aa]
        cost_ss = -out_ss[:, tgt_ss]

        # Compute the L1 cost between boxes
        cost_ca = torch.cdist(out_ca, tgt_ca, p=1)

        # Compute the giou cost betwen boxes

        # Final cost matrix
        C = self.cost_ca * cost_ca + self.cost_aa * cost_aa + self.cost_ss * cost_ss
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(tgt_ca_)]*bs
        

        # print("C:", C.max())
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        matches = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        # import pdb;pdb.set_trace()
       
        return matches
    
    def forward_single_dec(self, outputs, batch, i):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [dec_size, batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [dec_size, batch_size, num_queries, 4] with the predicted box coordinates
            batch: This is a list of batch (len(batch) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected batch (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        
        pred_aa = outputs["amino_acid_logits"][i,:,:,:21]
        pred_ss = outputs["secondary_structure_logits"][i,:,:,:4]
        pred_ca = outputs["normed_backbone_frame"][i,:,:,4:].float()
        N_bs, N_res = pred_aa.shape[:2]
        if torch.isnan(pred_aa).any():
            import pdb;pdb.set_trace()

        pred_aa = pred_aa.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        pred_ss = pred_ss.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        pred_ca = pred_ca.flatten(0, 1)  # [batch_size * num_queries, 4]

        
        # seq_mask = batch['seq_mask'].bool()[...,0]
        gt_mask = batch['atom14_gt_exists'].bool()[...,1,0]
        # import pdb;pdb.set_trace()

        tgt_aa = batch['aatype'][gt_mask][...,0]
        tgt_ss = batch['sstype'][gt_mask][...,0]
        tgt_ca = batch['normed_ca_positions'][gt_mask][...,0].to(pred_ca.dtype)
        
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_aa = -pred_aa[:, tgt_aa]
        cost_ss = -pred_ss[:, tgt_ss]

        # Compute the L1 cost between boxes
        cost_ca = torch.cdist(pred_ca, tgt_ca, p=1)

        # Final cost matrix
        C = self.cost_ca * cost_ca + self.cost_aa * cost_aa + self.cost_ss * cost_ss
        C = C.view(N_bs, N_res, -1).cpu()

        if len(gt_mask.shape)==1:
            gt_mask = gt_mask[None]
        sizes = [ num.item() for num in gt_mask.sum(-1)]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        matches = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        # import pdb;pdb.set_trace()
       
        return matches

class BackboneFrameHead(pl.LightningModule):
    """Head to predict bachbone affine at the masked locations.
    """
    
    def __init__(self, c_s=384, c_out_q=4, c_out_t=3, **kwargs):
        super().__init__()
        self.c_s = c_s
        self.quat_head  = MLP(c_s, c_s, c_out_q)
        self.trans_head = MLP(c_s, c_s, c_out_t)

    def forward(self, single):
        """Builds AffineHead module.
        Arguments
        -------------
        single: [N_res, c_s]
        
        Return
        -------------
        logits: [N_res, 7]
        """
        assert single.shape[-1] == self.c_s

        out_quat  = self.quat_head(single)
        out_trans = self.trans_head(single).sigmoid() 
        
        normed_rots = Rotation(quats=out_quat, normalize_quats=True)
        out_frame  = torch.cat([normed_rots.get_quats(), out_trans], dim=-1)
        return out_frame
    

class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self, no_bins, c_in, c_hidden, **kwargs):
        super(PerResidueLDDTCaPredictor, self).__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = LayerNorm(self.c_in)

        self.linear_1 = Linear(self.c_in, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_3 = Linear(self.c_hidden, self.no_bins, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def forward(self, z):  # [*, N, N, C_z]
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits


class TMScoreHead(nn.Module):
    """
    For use in computation of TM-score, subsection 1.9.7
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of bins
        """
        super(TMScoreHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pairwise embedding
        Returns:
            [*, N_res, N_res, no_bins] prediction
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        return logits


class MaskedMSAHead(nn.Module):
    """
    For use in computation of masked MSA loss, subsection 1.9.9
    """

    def __init__(self, c_m, c_out, **kwargs):
        """
        Args:
            c_m:
                MSA channel dimension
            c_out:
                Output channel dimension
        """
        super(MaskedMSAHead, self).__init__()

        self.c_m = c_m
        self.c_out = c_out

        self.linear = Linear(self.c_m, self.c_out, init="final")

    def forward(self, m):
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
        Returns:
            [*, N_seq, N_res, C_out] reconstruction
        """
        # [*, N_seq, N_res, C_out]
        logits = self.linear(m)
        return logits


class ExperimentallyResolvedHead(nn.Module):
    """
    For use in computation of "experimentally resolved" loss, subsection
    1.9.10
    """

    def __init__(self, c_s, c_out, **kwargs):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super(ExperimentallyResolvedHead, self).__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, N, C_out] logits
        """
        # [*, N, C_out]
        logits = self.linear(s)
        return logits

class AminoAcidHead(pl.LightningModule):
    """
    For use in computation of "Amino Acid classification" loss, subsection
    1.9.10
    """

    def __init__(self, c_s, c_out, **kwargs):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of Amino Acid type
        """
        super(AminoAcidHead, self).__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, N, C_out] logits
        """
        # [*, N, C_out]
        logits = self.linear(s)
        return logits

class SecondaryStructureHead(pl.LightningModule):
    """
    For use in computation of "Secondary Structure" loss, subsection
    1.9.10
    """

    def __init__(self, c_s, c_out, **kwargs):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of Secondary Structure types
        """
        super(SecondaryStructureHead, self).__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, N, C_out] logits
        """
        # [*, N, C_out]
        logits = self.linear(s)
        return logits
