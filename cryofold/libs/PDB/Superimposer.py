# Copyright (C) 2002, Thomas Hamelryck (thamelry@binf.ku.dk)
#
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.

"""Superimpose two structures."""


import numpy
import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from .PDBExceptions import PDBException
import torch
import torch.nn.functional as F
from torch import nn

from scipy.optimize import linear_sum_assignment

class Superimposer:
    """Rotate/translate one set of atoms on top of another to minimize RMSD."""

    def __init__(self):
        """Initialize the class."""
        self.rotran = None
        self.rms = None

    def run(self, fixed_coord, moving_coord):
        sup = SVDSuperimposer()
        sup.set(fixed_coord, moving_coord)
        sup.run()
        self.rms = sup.get_rms()
        self.rotran = sup.get_rotran()

    def set_rotran(self, rotran):
        self.rotran = rotran

    def set_atoms(self, fixed, moving):
        """Prepare translation/rotation to minimize RMSD between atoms.

        Put (translate/rotate) the atoms in fixed on the atoms in
        moving, in such a way that the RMSD is minimized.

        :param fixed: list of (fixed) atoms
        :param moving: list of (moving) atoms
        :type fixed,moving: [L{Atom}, L{Atom},...]
        """
        if not len(fixed) == len(moving):
            raise PDBException("Fixed and moving atom lists differ in size")
        length = len(fixed)
        fixed_coord = numpy.zeros((length, 3))
        moving_coord = numpy.zeros((length, 3))
        for i in range(0, length):
            if isinstance(fixed[i], numpy.ndarray):
                fixed_coord[i] = fixed[i]
                moving_coord[i] = moving[i]
            else:
                fixed_coord[i] = fixed[i].get_coord()
                moving_coord[i] = moving[i].get_coord()
        self.run(fixed_coord, moving_coord)

    def fit_helix_ali(self, atoms, helix, helix_len=5):
        assert len(atoms)>=5, 'at least {} atoms to fit helix'.format(helix_len)

        solutions = {}
        solutions['id']=[]
        solutions['rms']=[]
        solutions['rotran']=[]
        best = 0
        for i in range(len(atoms)-helix_len+1):
            self.run(atoms[i:i+helix_len], helix[:helix_len])
            # print(i, self.rms)
            if len(solutions['rms'])>0 and self.rms < min(solutions['rms']):
                best = i
                
            solutions['id'].append(i)
            solutions['rms'].append(self.rms)
            solutions['rotran'].append(self.rotran)
        # print('best: ', best)
        self.best_resid = best
        self.rms = solutions['rms'][best]
        self.rotran = solutions['rotran'][best]


    def fit_helix_icp(self, atoms, helix, helix_len=5):
        from .icp import icp

        solutions = {}
        solutions['id']=[]
        solutions['rms']=[]
        solutions['rotran']=[]
        best = 0
        # import pdb;pdb.set_trace()
        for i in range(len(atoms)-helix_len+1):
            T, distances, iterations, mean_error = icp(helix[:helix_len], atoms[i:i+helix_len], tolerance=0.000001)
            self.rms = mean_error
            self.rotran = T[0:3,0:3].T, T[0:3,3]
            if len(solutions['rms'])>0 and self.rms < min(solutions['rms']):
                best = i
                
            solutions['id'].append(i)
            solutions['rms'].append(self.rms)
            solutions['rotran'].append(self.rotran)
        # print('best: ', best)
        # import pdb;pdb.set_trace()
        self.best_resid = best
        self.rms = solutions['rms'][best]
        self.rotran = solutions['rotran'][best]

    def fit_helix(self, atoms, est_len=True):
        from . import PeptideBuilder
        from scipy.spatial import distance_matrix

        dist = distance_matrix(atoms, atoms)
        if est_len:
            length_est = max(len(atoms), int(dist.max()/1.51) )
        else:
            length_est = len(atoms)
        print(len(atoms), int(dist.max()/1.51))

        helix_str = PeptideBuilder.make_helix(length_est)
        
        # self.fit_helix_icp(atoms, helix_str.get_coords('ca'), length_est)
        self.fit_helix_ali(atoms, helix_str.get_coords('ca'), length_est)

        self.apply(helix_str.get_atoms())

        # print(self.best_resid, self.rms)

        return helix_str
    
    def sgd(self):
        # model = 

        
        return theta  

    # def sgd_refine(self, struct):




    def apply(self, atom_list):
        """Rotate/translate a list of atoms."""
        if self.rotran is None:
            raise PDBException("No transformation has been calculated yet")
        rot, tran = self.rotran
        rot = rot.astype("f")
        tran = tran.astype("f")
        for atom in atom_list:
            atom.transform(rot, tran)



class TorsionAngle(nn.Module):
    """ This is the Torsion Angle refinement module that performs object detection """
    def __init__(self, num,dim):
        super().__init__()
        self.delta = torch.nn.Parameter(torch.zeros(num,dim), require_grad=True)

    def forward(self, x):
        x = x + 7.0 * torch.sigmoid(self.delta)
        return x


def solver(x, y):
    max_norm = 0.1
    model = TorsionAngle(x.shape[0], x.shape[1])
    model.train()
    # criterion.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.98), weight_decay=0.00001)
    for epoch in range(10):
        outputs = model(x)
        # generate coords add matcher
        yp = angle_to_coords(outputs)
        indices = self.matcher(yp, y)
        # idx = self._get_src_permutation_idx(indices)
        # src_coords = outputs['pred_coords'][idx] 
        # tgt_coords = torch.cat([y[i] for t, (_, i) in zip(targets, indices)], dim=0) 

        # yp = 
        loss = F.mse_loss(yp, y)
        print(loss)
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
    


class HungarianMatcherCoords(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_coords: float = 5):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class  = cost_class
        self.cost_coords = cost_coords
        
        assert cost_class != 0 or cost_coords != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # bs, num_queries = outputs["pred_logits"].shape[:2]
        bs, num_queries = outputs.shape[:2]

        # We flatten to compute the cost matrices in a batch
        # out_prob   = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # out_coords = outputs["pred_coords"].flatten(0, 1)  # [batch_size * num_queries, 3]
        # if self.regbackbone:
        #     # [batch_size * num_queries, 3*4]
        #     out_coords_ca = out_coords.view(-1,4,3)[:,1,:]
        # else:
        #     out_coords_ca = out_coords
        
        # Also concat the target labels and boxes
        # tgt_ids    = torch.cat([v["class"] for v in targets])
        # tgt_coords = torch.cat([v["coords"] for v in targets]) #
        # tgt_coords_ca = tgt_coords[:,1,:]


        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # import pdb;pdb.set_trace()

        # Compute the L1 cost between boxes
        cost_coords = torch.cdist(outputs, targets, p=1)

        # Compute the giou cost betwen boxes
        # cost_giou = -generalized_box_iou(box_cxcyczwhl_to_xyzxyz(out_bbox), box_cxcyczwhl_to_xyzxyz(tgt_bbox))
        #if not (out_bbox[:, 3:] >= out_bbox[:, :3]).all():
        #    import pdb;pdb.set_trace()
        # cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        # Final cost matrix
        # C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # C = self.cost_coords * cost_coords + self.cost_class * cost_class #+ self.cost_giou * cost_giou
        C = cost_coords
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["coords"]) for v in targets]
        # import pdb;pdb.set_trace()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
