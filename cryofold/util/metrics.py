# Copyright (c) CryoNet Team, and its affiliates. All Rights Reserved

import numpy as np
import pandas as pd
import torch, math, time, subprocess, os
from io import StringIO 

try:
    from .boxes3d import Boxes
except ImportError:
    from boxes3d import Boxes


def f1_score(y_true, y_pred, classes_num=3):
    from sklearn.metrics import f1_score
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    classes = np.unique(y_true)

    f1_classes = f1_score(y_true, y_pred, average=None)

    f1 = np.ones(classes_num)
    for ic in range(len(classes)):
        f1[classes[ic]] = f1_classes[ic]
    return f1

def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    param:
        output: (N, num_classes), pred class
        target: (N, 1), gt class
    return:
        accuracy list: top1, topk accuracy
    """
    maxk = max(topk)
    batch_size = target.size(0)
    if len(output.shape)==1:
        target = target.long()
        pred = output.long()
        topk=(1,)
        correct = pred.eq(target.expand_as(pred))
    else:
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def onehot(target, nclass):
    target_mat = torch.zeros(target.shape[0], nclass)
    for i, j in enumerate(target):
        target_mat[i][j] = 1
    return target_mat

def meanAP(outputs, targets):
    """
    Computes mean average precision
    param:
        output: (N, num_classes), pred class
        target: (N, 1), gt class
    return:
        accuracy list: top1, topk accuracy
    """
    nclass = outputs.size(1)
    targets = onehot(targets, nclass)
    ap = torch.zeros(nclass)
    rg = torch.arange(1, outputs.size(0) + 1).float()
    for k in range(nclass):
        # sort scores
        score  = outputs[:, k]
        target = targets[:, k]
        _, sortind = torch.sort(score, 0, True)
        truth = target[sortind]
        
        # compute true positive sums
        tp = truth.float().cumsum(0)

        # compute precision curve
        precision = tp.div(rg)

        # compute average precision
        ap[k] = precision[truth.byte()].sum() / max(float(truth.sum()), 1)
    return ap.mean()
    
def confusion_matrix(outputs, targets):
    """
    param:
        output: (N, num_classes), pred class
        target: (N, 1), gt class
    """
    nclass = outputs.size(1)
    confusion_matrix = torch.zeros(nclass, nclass)
    _, preds = torch.max(outputs, 1)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix


def _get_flattened(dmap):
    """
    Get the elements of the upper triagular of the matrix
    
    [0,1,1,1,1]
    [0,0,1,1,1]
    [0,0,0,1,1]
    [0,0,0,0,1]
    [0,0,0,0,0]
    
    dmap: numpy.ndarray
        [ N_res*(N_res-1)/2 ] or [N_res, N_res]
    """
    # assert isinstance(dmap, np.ndarray)
    if isinstance(dmap, torch.Tensor):
        dmap = dmap.numpy()

    assert dmap.ndim == 1 or dmap.ndim == 2
    
    if dmap.ndim == 1:
        return dmap
    elif dmap.ndim == 2:
        return dmap[np.triu_indices_from(dmap, k=1)]

def _get_separations(dmap):
    """
    Get the distance between the row indice and col indice
    """
    t_indices = np.triu_indices_from(dmap, k=1)
    separations = np.abs(t_indices[0] - t_indices[1])
    return separations

def _get_sep_thresh_b_indices(dmap, thresh, comparator):
    """
    Return a 1D boolean array indicating where the sequence 
    separation in the upper triangle meets the threshold comparison
    
    dmap: numpy.ndarray
    thresh: float
    comparator: gt, lt, ge, le
    """
    assert comparator in {'gt', 'lt', 'ge', 'le'}, "ERROR: Unknown comparator for thresholding!"
    separations = _get_separations(dmap)
    if comparator == 'gt':
        threshed = separations > thresh
    elif comparator == 'lt':
        threshed = separations < thresh
    elif comparator == 'ge':
        threshed = separations >= thresh
    elif comparator == 'le':
        threshed = separations <= thresh
    return threshed

def _get_dist_thresh_b_indices(dmap, thresh, comparator):
    """
    Return a 1D boolean array indicating where the distance in the
    upper triangle meets the threshold comparison
    
    dmap: numpy.ndarray
    thresh: float
    comparator: gt, lt, ge, le
    """
    assert comparator in {'gt', 'lt', 'ge', 'le'}, "ERROR: Unknown comparator for thresholding!"
    dmap_flat = _get_flattened(dmap)
    if comparator == 'gt':
        threshed = dmap_flat > thresh
    elif comparator == 'lt':
        threshed = dmap_flat < thresh
    elif comparator == 'ge':
        threshed = dmap_flat >= thresh
    elif comparator == 'le':
        threshed = dmap_flat <= thresh
    return threshed


def calculate_LDDT(true_map, pred_map, R=15.0, sep_thresh=-1, T_set=[0.5, 1, 2, 4]):
    """
    lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests.
    doi: 10.1093/bioinformatics/btt473.
    
    Parameters
    -----------------------
    true_map: numpy.ndarray
        Distance map of the true structure
    pred_map: numpy.ndarray
        Distance map of the predicted structure
    R: float
        The threshold of the distance between residual pairs to consider
    sep_thresh: int
        The number of residues of seperation to consider
    T_set: list
        Maximum difference of distance map
    """
    assert true_map.ndim == 2
    assert pred_map.ndim == 2
    
    # Helper for number preserved in a threshold
    def get_n_preserved(ref_flat, mod_flat, thresh):
        err = np.abs(ref_flat - mod_flat)
        n_preserved = (err < thresh).sum()
        return n_preserved
    
    # flatten upper triangles
    true_flat_map = _get_flattened(true_map)
    pred_flat_map = _get_flattened(pred_map)
    
    # Find set L
    S_thresh_indices = _get_sep_thresh_b_indices(true_map, sep_thresh, 'gt')
    R_thresh_indices = _get_dist_thresh_b_indices(true_flat_map, R, 'lt')
    
    L_indices = S_thresh_indices & R_thresh_indices
    
    true_flat_in_L = true_flat_map[L_indices]
    pred_flat_in_L = pred_flat_map[L_indices]
    
    # Number of pairs in L
    L_n = L_indices.sum()
    
    # Calculated lDDT
    preserved_fractions = []
    for _thresh in T_set:
        _n_preserved = get_n_preserved(true_flat_in_L, pred_flat_in_L, _thresh)
        _f_preserved = _n_preserved / (L_n + 0.00001)
        preserved_fractions.append(_f_preserved)
    
    lDDT = np.mean(preserved_fractions)
    return round(lDDT, 4)


def _parse_TMscore_outputs(output_txt):
    """
    output_txt: str
        Print by TMscore
    """
    
    metrics = {}
    lines = output_txt.strip().split('\n')
    for line in lines:
        if line.startswith('Number of residues in common'):
            metrics['TM-num'] = int(line.split('=')[1].strip())
        elif line.startswith('RMSD of'):
            metrics['RMSD'] = float(line.split('=')[1].strip())
        elif line.startswith('TM-score'):
            metrics['TM-score'] = float(line.split('=')[1].split('(')[0].strip())
        elif line.startswith('GDT-TS-score'):
            metrics['GDT-TS-score'] = float(line.split('=')[1].split('%')[0].strip())
    return metrics

def calc_tmscore(pdb_true, pdb_pred):
    return calc_tmscore_usalign(pdb_true, pdb_pred)
    if "pd.pdb" in pdb_pred:
        pdb_pred_al = pdb_pred.replace("pd.pdb", "al")
    elif "ca.pdb" in pdb_pred:
        pdb_pred_al = pdb_pred.replace("pd.pdb", "al")
    else:
        pdb_pred_al = pdb_pred.replace(".pdb", "_al")
    # subprocess.getoutput(f"pdb_sort {pdb_pred} > {pdb_pred}.pdb")
    # subprocess.getoutput(f"pdb_sort {pdb_true} > {pdb_true}.pdb")
    output = subprocess.getoutput(f"TMscore -ter 1 -c {pdb_pred} {pdb_true} -ter 1 -o {pdb_pred_al} ")
    # import pdb;pdb.set_trace()
    # output = subprocess.getoutput(f"TMscore -ter 1 -c {pdb_pred} {pdb_true}")
    # output = subprocess.getoutput(f"TMscore -c {pdb_pred}.pdb {pdb_true}.pdb -ter 1 ")
    # output = subprocess.getoutput(f"TMscore -c {pdb_pred}.pdb {pdb_true}.pdb")
    metrics = _parse_TMscore_outputs(output)
    # subprocess.getoutput(f"rm {pdb_pred}.pdb")
    # subprocess.getoutput(f"rm {pdb_true}.pdb")
    subprocess.getoutput(f"rm {pdb_pred_al} {pdb_pred_al}.pml {pdb_pred_al}_all* {pdb_pred_al}_atm*")
    # if 'res_num' in metrics:
    #     if metrics['res_num'] != pdb_mask_num:
    #         print("Warning: different number of residues between TMscore and batchX:", metrics['res_num'], pdb_mask_num)
    #     del metrics['res_num']
    # else:
    #     print("Warning: res_num not in metrics")
    
    if 'TM-num' not in metrics:
        metrics['TM-num'] = 0.0
    if 'RMSD' not in metrics:
        metrics['RMSD'] = 0.0
    if 'TM-score' not in metrics:
        metrics['TM-score'] = 0.0
    if 'GDT-TS-score' not in metrics:
        metrics['GDT-TS-score'] = 0.0
    return metrics

def _parse_TMscore_usalign_outputs(output_txt, format='1'):
    """
    output_txt: str
        Print by TMscore
    """
    if format == '1':
        metrics = {}
        lines = output_txt.strip().split('\n')
        for line in lines:
            #print(line)
            #import pdb;pdb.set_trace()
            if line.startswith('Aligned length= '):
                s_i = len('Aligned length= ')
                fds = line.split(', ')
                #import pdb;pdb.set_trace()

                metrics['L_common'] = int(fds[0].split('= ')[1])
                metrics['RMSD'] = float(fds[1].split('= ')[1])
                metrics['TM-num'] = 0
            elif 'normalized by length of Structure_2' in line:
                metrics['TM-score'] = float(line.split(' ')[1].strip())
            # sd
        return metrics
    
    elif format == '2':
        df = pd.read_csv(
                    StringIO(output_txt),
                    sep="\s+",
                    skiprows=lambda x: x % 2 != 1,
                    names=[
                        "target",
                        "template",
                        "tm1",
                        "tm2",
                        "rmsd",
                        "id1",
                        "id2",
                        "idali",
                        "docked_seqlength",
                        "template_seqlength",
                        "aligned_length",
                    ],
                ).reset_index(drop=True)
        metrics = {}
        #import pdb;pdb.set_trace()
        if df.shape[0] == 0:
            metrics['TM-num'] = 0
            metrics['RMSD'] = -1.0
            metrics['TM-score'] = 0.0
            metrics['L_common'] = 0
            return metrics

        # lines = output_txt.strip().split('\n')
        # for line in lines:
        if 'idali' in df.columns:
            metrics['TM-num'] = df['idali'].item()
            metrics['L_common'] = df['aligned_length'].item()
            
        if 'rmsd' in df.columns:
            metrics['RMSD'] = df['rmsd'].item()
        if 'tm2' in df.columns:
            metrics['TM-score'] = df['tm2'].item()
        # elif line.startswith('GDT-TS-score'):
            # metrics['GDT-TS-score'] = float(line.split('=')[1].split('%')[0].strip())
        # print(metrics)
        metrics['lDDT'] = 0.0
    else:
        raise "Wrong format!!! format should be one of 1/2."
    return metrics

def calc_tmscore_usalign(pdb_true, pdb_pred):
    if "pd.pdb" in pdb_pred:
        pdb_pred_al = pdb_pred.replace("pd.pdb", "al")
    elif "ca.pdb" in pdb_pred:
        pdb_pred_al = pdb_pred.replace("pd.pdb", "al")
    else:
        pdb_pred_al = pdb_pred.replace(".pdb", "_al")

    pdb_pred_log = pdb_pred.replace(".pdb", ".uslog")
    #import pdb;pdb.set_trace()
    
    if not os.path.exists(pdb_pred_log):
        # subprocess.getoutput(f"pdb_sort {pdb_pred} > {pdb_pred}.pdb")
        # subprocess.getoutput(f"pdb_sort {pdb_true} > {pdb_true}.pdb")
        # output = subprocess.getoutput(f"TMscore -ter 1 -c {pdb_pred} {pdb_true} -ter 1 -o {pdb_pred_al} ")
        # output = subprocess.getoutput(f"USalign {pdb_pred} {pdb_true} -mm 1 -ter 1 -mol prot -outfmt 2 -o {pdb_pred_al} > {pdb_pred_log}")
        output = subprocess.getoutput(f"USalign {pdb_pred} {pdb_true} -mm 1 -ter 1 -mol prot -o {pdb_pred_al} > {pdb_pred_log}")
    #else:
    #    print("uslog is yes.")
    output2 = ''
    #for line in output.split('\n'):
    #    if not line.startswith('Warning') and not line.startswith('Sequence is too short') and not line.startswith('ERROR!'):
    #        output2 += line + '\n'
    with open(pdb_pred_log, 'r') as f:
        for line in f:
            if not line.startswith('Warning') and not line.startswith('Sequence is too short') and not line.startswith('ERROR!'):
                output2 += line + '\n'  
    #import pdb;pdb.set_trace()
    # output = subprocess.getoutput(f"TMscore -ter 1 -c {pdb_pred} {pdb_true}")
    # output = subprocess.getoutput(f"TMscore -c {pdb_pred}.pdb {pdb_true}.pdb -ter 1 ")
    # output = subprocess.getoutput(f"TMscore -c {pdb_pred}.pdb {pdb_true}.pdb")
    metrics = _parse_TMscore_usalign_outputs(output2, format='1')
    # subprocess.getoutput(f"rm {pdb_pred}.pdb")
    # subprocess.getoutput(f"rm {pdb_true}.pdb")
    subprocess.getoutput(f"rm {pdb_pred_al}*.pml")
    # if 'res_num' in metrics:
    #     if metrics['res_num'] != pdb_mask_num:
    #         print("Warning: different number of residues between TMscore and batchX:", metrics['res_num'], pdb_mask_num)
    #     del metrics['res_num']
    # else:
    #     print("Warning: res_num not in metrics")
    # print(metrics)
    if 'TM-num' not in metrics:
        metrics['TM-num'] = 0.0
    if 'RMSD' not in metrics:
        metrics['RMSD'] = 0.0
    if 'TM-score' not in metrics:
        metrics['TM-score'] = 0.0
    if 'lDDT' not in metrics:
        metrics['lDDT'] = 0.0
    if 'GDT-TS-score' not in metrics:
        metrics['GDT-TS-score'] = 0.0
    return metrics
    


class Metrics:
    def __init__(self, predicts, targets, nclass=None):
        """
        param:
            output: (N, num_classes), pred class
            target: (N, 1), gt class
        """
        self.predicts = predicts
        self.targets  = targets
        if nclass is None:
            self.nclass = predicts.size(1)
        else:
            self.nclass = nclass
    
    def topk_acc(self, topk=(1,5)):
        return accuracy(self.predicts, self.targets, topk)

    def mAP(self):
        return meanAP(self.predicts, self.targets)

class HyperParameter:
    def __init__(self, args=None):
        # self.set_args(args)
        self.metric=2
        # self.num_class            = 22    # fixed 
        # self.gap_penalty          = 0.5   
        # self.voxel_size           = 0.666 # fixed 

        if self.metric == 0 :
            self.avg_dis         = 0.11
            self.std_dis         = 0.1
            self.visit_thres     = 0.6
            self.metric_string   = "iou"
            self.found_dis       = 0.11

        elif self.metric == 1 :
            self.metric_string   = "boxcenter"
            self.found_dis       = 3
            # self.beta_a          = [
            #     0.0, 4.0, 4.0, 3.0, 3.0, 3.0,
            #     3.0,3.0
            # ]
            self.beta_a          = [
                0.0, 4.07, 3.95, 2.76, 2.72, 2.53,
                2.78,2.85
            ]
            self.avg_dis_a       = [ 
                0,               # 0-order  
                5.004869,        # 1-order
                6.863611,        # 2-order
                7.7992973,       # 3-order
                9.277723,        # 4-order
                11.290703,       # 5-order
                12.749742,       # 6-order
                13.827489,       # 7-order
            ]
            self.std_dis_a       =[
                1,               # 0-order  
                0.96127814 ,     # 1-order+ 0.041
                1.1356679,       # 2-order
                2.2931666,       # 3-order
                3.0079613,       # 4-order
                3.2550573,       # 5-order
                3.8092184,       # 6-order
                4.437546,        # 7-order
            ]
            # self.avg_dis         = 4.867
            # self.std_dis         = 1.093 + 0.1 +1
            # self.visit_thres     = 2
        elif self.metric == 2 :
            self.metric_string   = "ca"
            self.found_dis       = 3
            self.beta_a          = [
                0.0, 6.0, 2.5, 2.0, 2.0,2.3,
                2.5,2.5
            ]
            self.avg_dis_a       = [ 
                0.0,             # 0-order  
                3.7994094,       # 1-order
                5.9303923,       # 2-order
                7.2320356,       # 3-order
                8.869832,        # 4-order
                10.757701,       # 5-order
                12.259113,       # 6-order
                13.4832945,      # 7-order
            ]
            self.std_dis_a       =[
                1,               # 0-order  
                0.018842695 + 1, # 1-order
                0.6252558 + 0.4, # 2-order
                2.018165,        # 3-order
                2.7228906,       # 4-order
                3.1068196,       # 5-order
                3.6568744,       # 6-order
                4.231135,        # 7-order
            ]
            # angle: min:78.4488, max:151.3294
            self.avg_ang         = 104.2654
            self.std_ang         = 16.6767
        else:
            raise "Wrong metric!!! metric should be one of 0/1/2."
 

class AAMetrics():
    def __init__(self, gt_boxes, pd_boxes, class_idx=0, path_idx=0, args=None):
        self.device = pd_boxes.device
        if args is None:
            args = HyperParameter()
        # if isinstance(gt_boxes, Boxes):
        self.gt_boxes = gt_boxes
        # else:
        #     self.gt_boxes = Boxes(gt_boxes, class_idx=class_idx, path_idx=path_idx)

        # if isinstance(gt_boxes, Boxes):
        self.pd_boxes = pd_boxes
        # else:
        #     self.pd_boxes = Boxes(pd_boxes, class_idx=class_idx, path_idx=path_idx)
        metric = args.metric_string
        self.args         = args
        self.dis          = {}
        self.dis_pd       = {}
        self.EMPTY        = -99999
        # self.dis_pd_mask  = {}
        # self.dis['iou']            = self.distance_matrix(metric="iou")
        self.dis[metric]           = self.distance_matrix()
        # self.dis['boxcenter']    = self.distance_matrix(metric="boxcenter")
        self.dis_pd[metric]        = self.distance_matrix(pd=True)
        # self.dis_pd['boxcenter'] = self.distance_matrix(metric="boxcenter", pd=True)
        # thr = args.avg_dis_a[1] + self.args.std_dis_a[1]*args.beta_a[1]
        # self.dis_pd_mask   =  self.distance_mask(metric=metric, thr=thr, pd=True)
        # self.dis_pd_mask['boxcenter'] =  self.distance_mask(metric="boxcenter", thr=3):
        # angles=self.gt_boxes.get_angles()
        # print(angles)
        # print(self.gt_boxes.get_angle(0,1,2))


    def set_gt_boxes(self, gt_boxes, class_idx=0, path_idx=0):
        if isinstance(gt_boxes, Boxes):
            self.gt_boxes = gt_boxes
        else:
            self.gt_boxes = Boxes(gt_boxes, class_idx=class_idx, path_idx=path_idx)

    def set_pd_boxes(self, pd_boxes, class_idx=0, path_idx=0):
        if isinstance(pd_boxes, Boxes):
            self.pd_boxes = pd_boxes
        else:
            self.pd_boxes = Boxes(pd_boxes, class_idx=class_idx, path_idx=path_idx)

    @property
    def length(self):
        return len(self.gt_boxes)

    def stats_order(self, order):
        dis = self.dis[self.args.metric_string]
        dis_a = torch.zeros(len(dis)-order)
        for i in range(len(dis)-order):
            dis_a[i] = dis[i,i+order]
        beta_a = [
            (dis_a.mean() - dis_a.min())/dis_a.std(),
            torch.abs(dis_a.mean() - dis_a.max())/dis_a.std()
        ]
        print("order", order)
        print("min: {}, max: {}, mean: {}, std: {}".format(dis_a.min(),dis_a.max(),dis_a.mean(), dis_a.std()))
        print("beta: {}, {}".format(beta_a[0], beta_a[1]))
        return max(beta_a).item(), dis_a.mean().item(), dis_a.std().item()

    def stats_orders(self, max_order=7):
        beta_a    = []
        avg_dis_a = []
        std_dis_a = []
        for i in range(1,max_order):
            beta, avg_dis, std_dis = self.stats_order(i)
            beta_a.append(beta)
            avg_dis_a.append(avg_dis)
            std_dis_a.append(std_dis)
        

    def distance_matrix(self, metric="ca", pd=False):
        """
        return:
            IoU of self.gt_boxes, self.pd_boxes
        """
        
        if pd:
            pd_ca = self.pd_boxes.get_ca()
            boxes_dis = torch.cdist(pd_ca, pd_ca)
        else:
            gt_ca = self.gt_boxes.get_ca()
            pd_ca = self.pd_boxes.get_ca()
            boxes_dis = torch.cdist(gt_ca, pd_ca)
        return boxes_dis
        

    def distance_mask(self, metric="ca", thr=3, pd=False):
        if pd:
            dis = self.dis_pd[metric]
        else:
            dis = self.dis[metric]
        if metric == "iou":
            mask = (dis >= thr)
        else:
            mask = (dis <= thr)
        return mask

    def found(self, metric="ca", thr=3):
        """
        Percentage of protein/RNA residues in the deposited model or the unique part of the 
        deposited model that are within 3 Å of a residue (residues represented by their CA 
        or P atoms) in the automatically-generated model. 
        (by phenix, map_to_model nature method 2018 paper) 
        Returns:
            found: Percentage of predicted boxes found in gt boxes
        """

        dis = self.dis[metric]
        pd_path = self.pd_boxes.get_path()
        selected_boxes_ind = pd_path>=0
        if len(selected_boxes_ind) == 0:
            return 0
        dis = dis[:,selected_boxes_ind]
        if metric == "iou":
            max_val, max_ind = dis.max(dim=1)
            return torch.sum(max_val > thr).float() / self.length
        else:
            min_val, min_ind = dis.min(dim=1)
            return torch.sum(min_val < thr).float() / self.length

    def seq_match(self, metric="ca", thr=3, return_matches=False):
        """
        Percentage of found residues that have the same residue name.
        (proposed by phenix, map_to_model nature method 2018 paper) 
        Returns:
            seq_match: Percentage of predicted boxes class matched with gt boxes
        """
        assert self.gt_boxes.classes.sum() > 0
        match = 0
        mask = self.distance_mask(metric, thr)

        pd_path = self.pd_boxes.get_path()

        # selected_boxes_ind = pd_path[pd_path>=0].long()
        selected_boxes_ind = pd_path>=0
        if len(selected_boxes_ind) == 0:
            return 0
        ind_mask = torch.zeros(len(self.pd_boxes)).to(self.device) - 1
        ind_mask[selected_boxes_ind] = 1

        pd_classes = (torch.zeros(len(self.pd_boxes)).to(self.device) - 1).int()
        pd_classes[selected_boxes_ind] = self.pd_boxes.get_classes()[selected_boxes_ind]
        matches_pd = []
        if return_matches:
            pdb_path = self.pd_boxes.get_path()
        for i in range(mask.size(0)):
            if self.gt_boxes.classes[i] in pd_classes[mask[i, :]]:
                match += 1
            indices = ((mask[i, :] == 1).nonzero())
            if return_matches:
                for j in range(len(indices)):
                    id_pd = indices[j]
                    if self.gt_boxes.classes[i] == pd_classes[id_pd]:
                            matches_pd.append(id_pd.item())                    
                            # matches_pd.append(pdb_path[id_pd].item())                    
        if return_matches:
            return torch.tensor(match / self.length).to(self.device), matches_pd
        else:
            return torch.tensor(match / self.length).to(self.device)

    def sse_f1_score(self, metric="ca", thr=3):
        """
        Percentage of found residues that have the same residue name.
        (proposed by phenix, map_to_model nature method 2018 paper) 
        Returns:
            seq_match: Percentage of predicted boxes class matched with gt boxes
        """
        assert self.gt_boxes.classes.sum() > 0
        match = 0
        mask = self.distance_mask(metric, thr)

        pd_path = self.pd_boxes.get_path()

        # selected_boxes_ind = pd_path[pd_path>=0].long()
        selected_boxes_ind = pd_path>=0
        if len(selected_boxes_ind) == 0:
            return 0
        ind_mask = torch.zeros(len(self.pd_boxes)).to(self.device) - 1
        ind_mask[selected_boxes_ind] = 1

        pd_sses = (torch.zeros(len(self.pd_boxes)).to(self.device) - 1).int()
        pd_sses[selected_boxes_ind] = self.pd_boxes.get_sses()[selected_boxes_ind]
        # import pdb;pdb.set_trace()
        y_true = []
        y_pred = []
        for i in range(mask.size(0)):
            y_true.append(self.gt_boxes.sses[i])
            if self.gt_boxes.sses[i] in pd_sses[mask[i, :]]:
                y_pred.append(pd_sses[mask[i, :]][0])
                match += 1
            else:
               if mask[i, :].sum()>0:
                   y_pred.append(pd_sses[mask[i, :]][0])
               else:
                   y_pred.append(4)
               # import pdb;pdb.set_trace()
        y_true = np.array(y_true)-1
        y_pred = np.array(y_pred)-1
        # print(y_true)
        # print(y_pred)
        # import pdb;pdb.set_trace()

        f1 = f1_score(y_true, y_pred)  
        return f1

    

    def score_matched_chain(self, pd_resnum, pd_inds_by_gt, gt_resnum, gt_ind):

        def is_neighbor(i, j, m, n):
            if (i - j ==1) and (m - n ==1):
                return True
            elif (i - j !=1) and (m - n !=1):
                return True
            else:
                return False

        def get_resnum(ind, i):
            if i is None or i < 0 or i >= len(ind):
                return self.EMPTY
            else:
                return ind[i]

        pd_ind_l = pd_inds_by_gt[gt_ind-1] if gt_ind-1 >= 0 else self.EMPTY
        pd_ind_c = pd_inds_by_gt[gt_ind]
        pd_ind_r = pd_inds_by_gt[gt_ind+1] if gt_ind+1 <len(pd_inds_by_gt) else self.EMPTY

        if pd_ind_c == self.EMPTY:
            return 0

        pd_resnum_l = get_resnum(pd_resnum, pd_ind_l)
        pd_resnum_c = get_resnum(pd_resnum, pd_ind_c)
        pd_resnum_r = get_resnum(pd_resnum, pd_ind_r) 

        gt_resnum_l = get_resnum(gt_resnum, gt_ind-1)
        gt_resnum_c = get_resnum(gt_resnum, gt_ind)
        gt_resnum_r = get_resnum(gt_resnum, gt_ind+1)

        gt_class = self.gt_boxes.get_classes()[gt_ind]
        pd_class = self.pd_boxes.get_classes()[pd_ind_c]
        if (gt_class == pd_class).item():
            l = is_neighbor(pd_resnum_c, pd_resnum_l, gt_resnum_c, gt_resnum_l)
            r = is_neighbor(pd_resnum_r, pd_resnum_c, gt_resnum_r, gt_resnum_c)
            if l and r:
                return 1
            elif l or r:
                return 0.5
            else:
                return 0
        else:
            return 0


    def chain_match(self, metric="ca", thr=3, use_pdb=False):
        """Percentage of seq matched residues that have the same neighbours in the chain 
        according to the sequence. 
        (by us)
        
        Args:
            metric (str, optional): metric type to estimating chain match. Defaults to "ca".
            thr (int, optional): threshold for estimating a match. Defaults to 3(A).
        
        Returns:
            [float]: Percentage of traced chain matched with gt chain
        """
        assert self.gt_boxes.path.sum() > 0
        gt_path = self.gt_boxes.path 
        dis = self.dis[metric]

        pd_path = self.pd_boxes.get_path().long() 
        dis_min_by_gt, pd_inds_by_gt = dis.min(dim=1)
        pd_inds_by_gt[dis_min_by_gt>thr] = self.EMPTY
        scores = []
        for gt_ind in range(self.length):
            score = self.score_matched_chain(pd_path,
                pd_inds_by_gt, gt_path, gt_ind)
            scores.append(score)
        return torch.tensor(sum(scores) / self.length).to(self.device)

    # def evaluate_chain(self, path, correct_class=None):
    #     if correct_class is not None:
    #         pd_classes = self.pd_boxes.get_classes()
    #         correct_class = torch.from_numpy( correct_class).int()
    #         pd_classes[correct_class>=0] = correct_class[correct_class>=0]
    #         # pd_classes[correct_class>=0] = torch.from_numpy( correct_class[correct_class>=0]).int()
    #         self.pd_boxes.set_classes(pd_classes)
    #     self.pd_boxes.set_path(path)
    #     self.log(metric=self.args.metric_string, thr=self.args.found_dis, found=True, \
    #         seq_match=True, chain_match=True, path=path)

    def evaluate_chain(self, path, correct_class=None, score_path=None):
        if correct_class is not None:
            pd_classes = self.pd_boxes.get_classes()
            correct_class = torch.from_numpy(correct_class).int()
            pd_classes[correct_class >= 0] = correct_class[correct_class >= 0]
            # pd_classes[correct_class>=0] = torch.from_numpy( correct_class[correct_class>=0]).int()
            self.pd_boxes.set_classes(pd_classes)
        self.pd_boxes.set_path(path)
        self.log(metric=self.args.metric_string, thr=self.args.found_dis, found=True,
                 seq_match=True, chain_match=True, path=score_path)

    def angle_mis_match(self, i, j, k):
        ca = False
        if self.args.metric_string == 'ca':
            ca = True
        angle = self.pd_boxes.get_angle(i, j, k, ca=ca)
        if angle > (self.args.avg_ang + self.args.std_ang * 3) or \
           angle < (self.args.avg_ang - self.args.std_ang * 1.8) :
            return True
        else:
            return False
        # import pdb; pdb.set_trace()


    def is_mis_match(self, dis, order=1):
        if torch.allclose(dis, torch.tensor([0.0]).to(self.device)):
            return True
        if dis > (self.args.avg_dis_a[order] + self.args.std_dis_a[order]*self.args.beta_a[order]) or \
           dis < math.fabs(self.args.avg_dis_a[order] - self.args.std_dis_a[order]*self.args.beta_a[order]) :
            return True
        else:
            return False
    
    def is_neighbor_mis_match(self, path, start, box_id, eval_order=4, beta=2):
        matches = []
        gap_num = 0
        criterion = min(eval_order, int(self.args.kmer/2))
        for order in range(-eval_order, 0):
            ind_neighbor = start + order

            if ind_neighbor >=0 and ind_neighbor <= len(path) - 1:
                ind_box_in_path = path[ind_neighbor]
    
                if ind_box_in_path >=0 :
                    tmp_d = self.dis_pd[self.args.metric_string][ind_box_in_path][box_id]
                    match = self.is_mis_match(tmp_d, int(math.fabs(order)))
                    if match:
                        return True
                    matches.append(match)
                else:
                    gap_num += 1
        if gap_num > criterion:
            return True
        return np.array(matches).any()

    def is_neighbor_mis_match_all(self, path, start, box_id, eval_order=4, left=False, angle_match=False):
        if angle_match:
            if start - 1 >=0 and start + 1 <= len(path) - 1:
                ind_left  = path[start - 1]
                ind_right = path[start + 1]
                if ind_left >=0 and ind_right >=0:
                    if self.angle_mis_match(ind_left, box_id, ind_right):
                        # import pdb; pdb.set_trace()
                        return True
        matches = []
        # gap_num = 0
        criterion = eval_order
        for order in range(-eval_order, eval_order + 1):
            if order == 0 or (left and order > 0):
                continue
            ind_neighbor = start + order
            if ind_neighbor >=0 and ind_neighbor <= len(path) - 1:
                ind_box_in_path = path[ind_neighbor]
                if ind_box_in_path >=0 :
                    tmp_d = self.dis_pd[self.args.metric_string][ind_box_in_path][box_id]
                    match = self.is_mis_match(tmp_d, int(math.fabs(order)))
                    # print("p: {}, pn:{} b:{} o:{}, dis:{:.3f} {}".format(start, ind_box_in_path, box_id, order, tmp_d, match))
                    if match:
                        return True
                    matches.append(False)
                # else:
                #     gap_num += 1
        # if gap_num > criterion:
        #     return True
        if len(matches)== 0 :
            return False
        return np.array(matches).any()

    def is_fragment_mis_match(self, fragment, path_node,left_or_right):
        gap_num = 1
        if (path_node == -1):
            return False,0
        if left_or_right==0:#path node at left
            right_one = -1
            for ii in fragment:
                if ii != -1:
                    right_one = ii
                    break
                gap_num += 1
            dis_ij = self.dis_pd[self.args.metric_string][path_node][right_one]
        else:
            left_one=-1
            for ii in fragment[::-1]:
                if ii != -1:
                    left_one = ii
                    break
                gap_num += 1
            dis_ij = self.dis_pd[self.args.metric_string][path_node][left_one]
            # print("hoho",path_node,left_one,dis_ij)

        if (dis_ij > (gap_num + 2) * self.args.avg_dis_a[1]):
            return True,dis_ij
        return False,dis_ij

    def get_error_dis(self, dis, order=1, metric="l1", zero_penalty=9999):
        if torch.allclose(dis, torch.tensor([0.0]).to(self.device)):
            return zero_penalty
        error_dis = (dis - self.args.avg_dis_a[order]) / self.args.std_dis_a[order]

        if metric=="l2":
            error_dis = torch.pow(error_dis, 2)
        else:
            error_dis = torch.abs(error_dis)
        return error_dis    

    def neighbor_score(self, path, start, box_id, eval_order=1, use_valid_order=False):
        total_error_dis = 0.0
        num_valid_order = 0
        for order in range(-eval_order, eval_order + 1):
            if order == 0:
                continue
            ind_neighbor = start + order
            if ind_neighbor >=0 and ind_neighbor <= len(path) - 1:
                order_neighbor = path[ind_neighbor]
                if order_neighbor >=0 :
                    tmp_d   = self.dis_pd[self.args.metric_string][order_neighbor][box_id]
                    tmp_e_d = self.get_error_dis(tmp_d, int(math.fabs(order)), metric="l1")
                    total_error_dis += tmp_e_d
                    num_valid_order += 1
        if use_valid_order and num_valid_order > 0:
            return total_error_dis / num_valid_order
        else:
            return total_error_dis

    def calculate_path_score(self, path, eval_order=4):
        metric      = self.args.metric_string
        seq_id      = self.gt_boxes.get_classes()
        pd_boxes    = self.pd_boxes
        gap_penalty = self.dis_pd[metric].max() *eval_order*2
        scores = []
        # s = ""
        for ii in range(len(path)):
            now  = path[ii]
            if now < 0 :
                scores.append(gap_penalty.item())
                # s += "{}:{:.3f},".format(ii, gap_penalty.item())
                continue
            for order in range(-eval_order, eval_order+1):
                if order == 0:
                    continue
                ind_neighbor = ii + order
                if ind_neighbor >=0 and ind_neighbor <= len(path) - 1:
                    order_neighbor     = path[ii + order]
                    if order_neighbor >= 0:
                        dis_neighbor       = self.dis_pd[metric][now][order_neighbor]
                        dis_neighbor_error = self.get_error_dis(dis_neighbor, int(math.fabs(order)))
                        scores.append(dis_neighbor_error)
        return sum(scores) / len(path) / (eval_order*2)
    def calculate_path_score0(self, path, eval_order=4):
        if isinstance(path, torch.Tensor):
            path    = path.int()
        # else:

        metric      = self.args.metric_string
        # seq_id      = self.gt_boxes.get_classes()
        # pd_boxes    = self.pd_boxes
        gap_penalty = self.dis[metric].max() *eval_order*2
        scores = []
        # s = ""
        for ii in range(len(path)):
            now  = path[ii]
            if now < 0 :
                scores.append(gap_penalty.item())
                # s += "{}:{:.3f},".format(ii, gap_penalty.item())
                continue
            for order in range(-eval_order, eval_order+1):
                if order == 0:
                    continue
                ind_neighbor = ii + order
                if ind_neighbor >=0 and ind_neighbor <= len(path) - 1:
                    order_neighbor     = path[ii + order]
                    if order_neighbor >= 0:
                        dis_neighbor       = self.dis_pd[metric][now][order_neighbor]
                        dis_neighbor_error = self.get_error_dis(dis_neighbor, int(math.fabs(order)))
                        scores.append(dis_neighbor_error)
                        # s += "{}-{}:{:.3f},".format(ii, order, dis_neighbor_error.item())
        # print(s)
        return sum(scores) / len(path) / (eval_order*2)
    

    def calculate_path_score_mask(self, path, now_id, eval_order=2):
        metric = self.args.metric_string
        pd_boxes = self.pd_boxes
        # gap_penalty = self.dis[metric].max() * eval_order * 2
        pd_scores = pd_boxes.get_scores()
        scores = []
        # s = ""
        ii=now_id
        now = path[ii]
        if now < 0:
            # scores.append(gap_penalty.item())
            # s += "{}:{:.3f},".format(ii, gap_penalty.item())
            return 100000,-1,-1
        for order in range(-eval_order, eval_order + 1):
            if order == 0:
                continue
            ind_neighbor = ii + order
            if ind_neighbor >= 0 and ind_neighbor <= len(path) - 1:
                order_neighbor = path[ii + order]
                if order_neighbor >= 0:
                    dis_neighbor = self.dis_pd[metric][now][order_neighbor]
                    dis_neighbor_error = self.get_error_dis(dis_neighbor, int(math.fabs(order)))
                    scores.append(dis_neighbor_error)
        gap_num = 0
        non_coverage = 0
        if (path[0] < 0):
            gap_num += 1
            non_coverage += 1
        for ii in range(1, len(path)):
            if (path[ii] < 0):
                if (path[ii - 1] > 0):
                    gap_num += 1
                non_coverage+=1
        return sum(scores), gap_num, non_coverage

    def calc_mask0(self, dis_thres=5.8, box_score_thres=0.25):
        box_scores = self.pd_boxes.get_scores()
        csv_list=[]
        score_list=[]
        neighbor_num=4

        whole_path_list=[[i for i in range(len(self.pd_boxes))]]
        for path in whole_path_list:
            score=np.zeros(len(path))
            for i in range(len(path)):
                if i<neighbor_num:
                    start = 0
                    now = i
                else:
                    start = i - neighbor_num
                    now = neighbor_num
                end=i+neighbor_num+1
                piece = path[start:end]
                dis_score, gap_num, non_coverage = self.calculate_path_score_mask(piece, now)
                avg_dis_score = float(dis_score / (len(piece) - non_coverage))
                if path[i] >= 0:
                    now_list = [avg_dis_score, float(box_scores[path[i]])]
                    if now_list[0] > dis_thres or now_list[1] < box_score_thres:
                        now_list.append(0)
                    else:
                        now_list.append(1)
                    csv_list.append(now_list)
                    score[i]=now_list[-1]
            score_list.append(score)
        # with open(targetfile, 'w')as f:
        #     f_csv = csv.writer(f)
        #     f_csv.writerow(["dis", "fill", "box", "seq", "dis_no_fill"])
        #     f_csv.writerows(csv_list)
        return score_list,whole_path_list

    # def calc_mask_neighbor_list(self):

    def estimate_mask(self, dis_thres=20, box_score_thres=0.25):
        box_scores = self.pd_boxes.get_scores()
        csv_list=[]
        score_list=[]
        scores = []
        neighbor_num=4

        whole_path_list=self.pd_boxes.get_whole_path_list()#[[i for i in range(len(self.pd_boxes))]]
        for path in whole_path_list:
            score=np.zeros(len(path))
            for i in range(len(path)):
                if i<neighbor_num:
                    start = 0
                    now = i
                else:
                    start = i - neighbor_num
                    now = neighbor_num
                end=i+neighbor_num+1
                piece = path[start:end]
                dis_score, gap_num, non_coverage = self.calculate_path_score_mask(piece, now)
                avg_dis_score = float(dis_score / (len(piece) - non_coverage))
                if path[i] >= 0:
                    now_list = [avg_dis_score, float(box_scores[path[i]])]
                    if now_list[0] > dis_thres or now_list[1] < box_score_thres:
                        now_list.append(0)
                    else:
                        now_list.append(1)
                    csv_list.append(now_list)
                    score[i]=now_list[-1]
                    scores.append(score[i])
            score_list.append(score)
        return scores, score_list

    def get_neighbors_mask(self, neighbor_num=5):
        # must: input whole path list follow the output pdb aa order(only recording -1)
        score_list, whole_path_list = self._cal_mask_score_pd()
        mask_output_list = []

        for chain_id in range(len(score_list)):
            path_len = len(score_list[chain_id])
            path = whole_path_list[chain_id]
            scores = score_list[chain_id]
            start = -1
            i = 0
            while (i < path_len):
                if (scores[i] != 0):
                    break
                i += 1
            if (i != 0):
                if i == path_len:
                    end_m = i
                else:
                    end_m = i
                    count = 0
                    while (end_m < path_len and count < neighbor_num):
                        if scores[end_m] != 0:
                            count += 1
                        end_m += 1
                mask_output_list.append([scores[:end_m].tolist(), path[:end_m]])

            while (i < path_len):
                if (scores[i] == 0 and start == -1):
                    start = i
                if (start != -1 and scores[i] != 0):
                    start_m = start - 1
                    count = 0
                    while (start_m >= 0 and count < neighbor_num):
                        if scores[start_m] != 0:
                            count += 1
                        start_m -= 1
                    start_m += 1

                    end_m = i
                    count = 0
                    while (end_m < path_len and count < neighbor_num):
                        if scores[end_m] != 0:
                            count += 1
                        end_m += 1
                    # print(start_m,end_m,start,i,path_len)
                    # print(mask_output_list)
                    mask_output_list.append(
                        [scores[start_m:end_m].tolist(), path[start_m:end_m]])
                    start = -1
                i += 1
            if (start != -1):
                start_m = start - 1
                count = 0
                while (start_m >= 0 and count < neighbor_num):
                    if scores[start_m] != 0:
                        count += 1
                    start_m -= 1
                start_m += 1
                mask_output_list.append([scores[start_m:].tolist(), path[start_m:]])
                # mask_output_list.append([";".join(scores[start_m:].tolist()), [res_index_dict[x] for x in path[start_m:]]])

        return mask_output_list


    def calculate_rmsd0(self,metric,thr):
        dis = self.dis[metric]
        min_val, min_ind = dis.min(dim=1)
        
        ind = min_val < thr
        aa=torch.arange(dis.shape[0])
        if len(ind) > 0:
            return torch.mean(dis[aa[ind],min_ind[ind]] **2 ) **0.5
        else:
            return torch.tensor([9999.0]).to(self.device)
    
    def calculate_rmsd(self, metric, thr):
        NULL_RMSD = torch.tensor(20.0).to(self.device)
        def get_atom_num(pose):
            num = 0
            for i in pose:
                if (i[0] == 0 and i[1] == 0 and i[2] == 0):
                    continue
                num += 1
            return num

        dis = self.dis[metric]
        min_val, min_ind = dis.min(dim=1)
        pd_poses = self.pd_boxes.get_poses(True)
        gt_poses = self.gt_boxes.get_poses(True)
        # import pdb;pdb.set_trace()
        ind = min_val < thr

        aa = torch.arange(dis.shape[0])
        pd_aa = min_ind[ind]
        gt_aa = aa[ind]
        if len(ind) == 0:
            return NULL_RMSD, NULL_RMSD, NULL_RMSD
        ca_rmsd = torch.mean(dis[gt_aa, pd_aa] ** 2) ** 0.5
        
        # backbone atom 
        pd_atom_num = [get_atom_num(i) for i in pd_poses[pd_aa]]
        gt_atom_num = [get_atom_num(i) for i in gt_poses[gt_aa]]

        if len(gt_atom_num) == 0 or len(pd_atom_num) == 0:
            return ca_rmsd, NULL_RMSD, NULL_RMSD
        if (min(gt_atom_num) < 4 or min(pd_atom_num) < 4):
            return ca_rmsd, NULL_RMSD, NULL_RMSD

        ind = np.where(np.array(gt_atom_num)==np.array(pd_atom_num))[0]

        backbone_pd   = pd_poses[pd_aa[ind], :4].reshape((-1, 3))
        backbone_gt   = gt_poses[gt_aa[ind], :4].reshape((-1, 3))
        backbone_dis2 = (backbone_pd - backbone_gt).pow(2).sum(1)
        if len(backbone_dis2)==0:
            return ca_rmsd, NULL_RMSD, NULL_RMSD
        backbone_rmsd = torch.mean(backbone_dis2) ** 0.5

        # all atom 
        pd_classes = self.pd_boxes.get_classes()
        gt_classes = self.gt_boxes.get_classes()
        all_atom_pd = []
        all_atom_gt = []
        unbingo_count = 0
        for i in range(len(pd_aa)):
            l1 = pd_atom_num[i]
            l2 = gt_atom_num[i]
            if l1 == l2 and pd_classes[pd_aa[i]] == gt_classes[gt_aa[i]]:
                all_atom_pd.extend(pd_poses[pd_aa[i], :l1].numpy())
                all_atom_gt.extend(gt_poses[gt_aa[i], :l1].numpy())
            else:
                unbingo_count += l2
        if len(all_atom_pd) == 0 or len(all_atom_gt) == 0:
            return ca_rmsd, backbone_rmsd, NULL_RMSD
        all_atom_pd = torch.from_numpy(np.array(all_atom_pd))
        all_atom_gt = torch.from_numpy(np.array(all_atom_gt))
        # import pdb;pdb.set_trace()
        all_atom_dis2 = (all_atom_pd - all_atom_gt).pow(2).sum(1)
        all_atom_rmsd = torch.mean(all_atom_dis2) ** 0.5
        return ca_rmsd, backbone_rmsd, all_atom_rmsd

    def calculate_prob(self, ):
        """Probility of aa prediction. 
        (by us)
        
        Returns:
            [float]: mean probility of all the aa 
        """
        assert self.pd_boxes.scores.sum() >= 0
        return self.pd_boxes.scores.mean().to(self.device)

    def calculate_LDDT(self):
        try:
            gt_ca = self.gt_boxes.get_ca()
            pd_ca = self.pd_boxes.get_ca()
            gtdis = torch.cdist(gt_ca, gt_ca)
            pddis = torch.cdist(pd_ca, pd_ca)
            lddt = calculate_LDDT(gtdis, pddis)
        except:
            lddt = 0
        return lddt

    def compute(self, path=None, use_pdb=False):
        metric = self.args.metric_string
        thr    = self.args.found_dis
        
        scores = {}
        # scores['metric']    = metric
        # scores['threshold'] = thr
        scores['gt']    = torch.tensor(len(self.gt_boxes)).float().to(self.device)
        scores['selected']  = torch.tensor(len(self.pd_boxes)).float().to(self.device)
        scores['found']     = self.found(metric, thr)
        
        s, m = self.seq_match(metric, thr, return_matches=True)
        sorted(m)
        self.matches = np.unique(m)
        scores['seq_match'] = s
        # scores['matches'] = self.matches
        # scores['rank_match'] = self.rank_match(metric)
        scores['chain_match'] = self.chain_match(metric, thr, use_pdb)
        if path is None:
            path = self.pd_boxes.get_path()
            # print(path)
        # scores['path_score'] = self.calculate_path_score(path)
        # scores['rmsd'] = self.calculate_rmsd(metric,thr)
        ca_rmsd, backbone_rmsd, all_atom_rmsd  = self.calculate_rmsd(metric,thr)
        scores['rmsd_ca'] = ca_rmsd
        scores['rmsd_bb'] = backbone_rmsd
        scores['rmsd_al'] = all_atom_rmsd
        return scores
       

    def log(self, metric="ca", thr=3, found=True, seq_match=True, \
        rank_match=False, chain_match=True, rmsd=True, sse=True, prob=True, path=None, \
        tmscore=False, lddt=False, use_pdb=False, title="[AAMetrics]", tmscore_met={}):
        info = "{} gt: {} selected: {}".format(title,
            len(self.gt_boxes), len(self.pd_boxes))
        if found:
            f = self.found(metric, thr)
            info += " found: {:.3f}".format(f)
        if seq_match:
            # import pdb; pdb.set_trace()
            # if return_matches:
            s, m = self.seq_match(metric, thr, return_matches=True)
            sorted(m)
            self.matches = np.unique(m)
                # m_line = ','.join(map(str, m))
            # else:
            #     info += " seq_match: {:.3f} matches: {}".format(s, m_line)
            # s = self.seq_match(metric, thr, return_matches=return_matches)
            info += " seq_match: {:.3f}".format(s)
        if rank_match:
            r = self.rank_match(metric)
            info += " rank_match: {:.3f}".format(r)
        if chain_match:
            c = self.chain_match(metric, thr, use_pdb)
            info += " chain_match: {:.3f}".format(c)
        # if path is not None:
        #     c = self.calculate_path_score(path)
        #     info += " path_score: {:.3f}".format(c)
        if rmsd:
            # c = self.calculate_rmsd(metric,thr)
            ca_rmsd, backbone_rmsd, all_atom_rmsd  = self.calculate_rmsd(metric,thr)
            if isinstance(c, torch.Tensor):
                c = c.item()
            info += " RMSD_ca: {:.3f}".format(ca_rmsd)
            info += " RMSD_bb: {:.3f}".format(backbone_rmsd)
            info += " RMSD_al: {:.3f}".format(all_atom_rmsd)
        if sse:
            f1 = self.sse_f1_score()
            info += " Helix_f1: {:.3f}".format(f1[0])
            info += " Strands_f1: {:.3f}".format(f1[1])
            info += " Coil_f1: {:.3f}".format(f1[2])
        if prob:
            try:
                p = self.calculate_prob()
            except:
                p = 0.0
            info += " prob: {:.3f}".format(p)
        if tmscore:
            info += " RMSD_tm: {:.3f}".format(tmscore_met['RMSD'])
            info += " TM-num: {:d}".format(int(tmscore_met['L_common']))
            info += " TM-score: {:.3f}".format(tmscore_met['TM-score'])
            info += " GDT-TS: {:.3f}".format(tmscore_met['GDT-TS-score'])
            info += " LDDT: {:.3f}".format(tmscore_met['lDDT'])
        #if lddt: 
        #    lddt = self.calculate_LDDT()
        #    info += " LDDT: {:.3f}".format(lddt)

        print(info)
        
         
