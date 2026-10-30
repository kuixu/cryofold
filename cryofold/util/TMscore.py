#!/usr/bin/env python
#-*- coding:utf-8 -*-
###############################################
#
#
#   Calculate TM scores
#   Author: Pan Li (lipan@shuimubio.com)
#                    @ Shuimu BioScience
#        https://www.shuimubio.com/
#
#
################################################

import os, sys, time, re, random, pickle, gzip, io, configparser, shutil, pathlib, tempfile, hashlib, argparse, json, inspect
from typing import Dict, Union, Optional
import numpy as np

from cryonet.libs.structure.hy36 import encode_atom_id, encode_chain_id
from cryonet.libs.structure import protein
from cryonet.model import constants as residue_constants
restypes_with_x = residue_constants.restypes_with_x

cur_path = str(pathlib.Path(__file__).parent.resolve())


##############################
## Code for lDDT
##############################

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
    assert isinstance(dmap, np.ndarray)
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


def get_chain2seq(prot):
    chain2seq = {}
    for ch_idx in set(prot.chain_index):
        
        residue_index = prot.residue_index[prot.chain_index == ch_idx]
        aatype = prot.aatype[prot.chain_index == ch_idx]
        
        restypes = [ restypes_with_x[d] for d in aatype ]
        num_res = max(residue_index) + 1
        res_list = ['X'] * (num_res + 1)
        for res_idx, res_type in zip(residue_index, restypes):
            res_list[res_idx] = res_type
        chain2seq[ch_idx] = "".join(res_list)
    return chain2seq

# def align2seq_nogap(seq1, seq2):

#     def find_overlap_substr(target_seq, sub_seq):
        
#         n1 = len(target_seq)
#         n2 = len(sub_seq)
#         find = False
        
#         for i in range(n1):
#             for j in range(n2):
#                 ii = i
#                 jj = j
#                 match = 0
#                 while target_seq[ii] == sub_seq[jj] or target_seq[ii] == 'X' or sub_seq[jj] == 'X':
#                     if target_seq[ii] == sub_seq[jj] and (target_seq[ii] != 'X' or sub_seq[jj] != 'X'):
#                         match += 1
#                     ii += 1
#                     jj += 1
#                     if ii == n1 or jj == n2:
#                         find = True
#                         break
#                 if find:
#                     break
#             if find:
#                 break
        
#         return i, j, match
    
#     i1, j1, match1 = find_overlap_substr(seq1, seq2)
#     i2, j2, match2 = find_overlap_substr(seq2, seq1)
    
#     i, j = i1, j1
#     if match2 > match1:
#         i, j = i2, j2
    
#     return i, j, max(match1, match2)

# def is_two_seq_equal(seq1, seq2):
#     num_res_1 = len( seq1.replace("X", "") )
#     num_res_2 = len( seq2.replace("X", "") )
#     min_num_seq = min(num_res_1, num_res_2)
#     i, j, match = align2seq_nogap(seq1, seq2)
#     if match >= min_num_seq // 10 and match > 10:
#         return True
#     else:
#         return False

def get_overvap_res_index(seq1, seq2):
    #i, j, match = align2seq_nogap(seq1, seq2)
    #assert match > 5, f"Got {match}"
    
    from alphafold.data.tools import kalign

    kalign = kalign.Kalign(binary_path=shutil.which("kalign"))
    result = kalign.align([seq1, seq2])
    split = result.index('>', 1)
    result1 = result[:split].strip().split('\n')
    result2 = result[split:].strip().split('\n')

    alnseq1 = "".join(result1[1:])
    alnseq2 = "".join(result2[1:])
    assert len(alnseq1) == len(alnseq2), f"Error: {alnseq1}; {alnseq2}"

    #print(alnseq1)
    #print(alnseq2)

    res_index_1 = []
    res_index_2 = []
    i1 = i2 = 0
    for r1, r2 in zip(alnseq1, alnseq2):
        if r1 == r2 and r1 != 'X' and r2 != 'X':
            res_index_1.append(i1)
            res_index_2.append(i2)
        if r1 != '-':
            i1 += 1
        if r2 != '-':
            i2 += 1
    res_index_1 = set(res_index_1)
    res_index_2 = set(res_index_2)
    return res_index_1, res_index_2

def get_matched_pos(gt_seq, pd_seq, gt_ch_idx, pd_ch_idx, gt_prot, pd_prot):
    gt_res_index, pd_res_index = get_overvap_res_index(gt_seq, pd_seq)

    true_chain = (gt_prot.chain_index == gt_ch_idx)
    true_res = np.array([d in gt_res_index for d in gt_prot.residue_index])

    gt_atom_position = gt_prot.atom_positions[ true_chain & true_res ]
    gt_atom_mask = gt_prot.atom_mask[ true_chain & true_res ]

    true_chain = (pd_prot.chain_index == pd_ch_idx)
    true_res = np.array([d in pd_res_index for d in pd_prot.residue_index])

    pd_atom_position = pd_prot.atom_positions[ true_chain & true_res ]
    pd_atom_mask = pd_prot.atom_mask[ true_chain & true_res ]

    gt_ca_pos = gt_atom_position[:, 1][ gt_atom_mask[:, 1].astype(np.bool_) ]
    pd_ca_pos = pd_atom_position[:, 1][ pd_atom_mask[:, 1].astype(np.bool_) ]
    
    return gt_ca_pos, pd_ca_pos
    
def get_chain_distance(gt_chain2seq, pd_chain2seq, gt_ch_idx, pd_ch_idx, gt_prot, pd_prot):
    gt_seq = gt_chain2seq[gt_ch_idx]
    pd_seq = pd_chain2seq[pd_ch_idx]
    #if is_two_seq_equal(gt_seq, pd_seq):
    
    num_res_gt = len( gt_seq.replace("X", "") )
    num_res_pd = len( pd_seq.replace("X", "") )
    min_num_seq = min(num_res_gt, num_res_pd)

    gt_ca_pos, pd_ca_pos = get_matched_pos(gt_seq, pd_seq, gt_ch_idx, pd_ch_idx, gt_prot, pd_prot)
    
    if gt_ca_pos.shape[0] >= min_num_seq // 2:
        return np.sqrt( np.sum( ( gt_ca_pos.mean(0) - pd_ca_pos.mean(0) ) ** 2 ) )
    else:
        return 1e9

def get_chain_pairs(gt_chain2seq, pd_chain2seq, gt_prot, pd_prot):

    gt_chain_list = sorted(gt_chain2seq.keys())
    pd_chain_list = sorted(pd_chain2seq.keys())

    #### Pair of GT chains and PD chains

    chain_pair_list_all = []

    for gt_ch_idx in gt_chain_list:
        for pd_ch_idx in pd_chain_list:
            dist = get_chain_distance(gt_chain2seq, pd_chain2seq, gt_ch_idx, pd_ch_idx, gt_prot, pd_prot)
            if dist < 10000:
                chain_pair_list_all.append([ gt_ch_idx, pd_ch_idx, dist ])
            #else:
            #    print(dist)

    chain_pair_list_all.sort(key=lambda x: x[2])

    #### Match all paired chains

    gt_chains = gt_chain_list.copy()
    pd_chains = pd_chain_list.copy()

    chain_pair_list = []
    for idx in range(len(chain_pair_list_all)):
        gt_ch_idx, pd_ch_idx, d = chain_pair_list_all[idx]
        if gt_ch_idx not in gt_chains or pd_ch_idx not in pd_chains:
            continue
        chain_pair_list.append([gt_ch_idx, pd_ch_idx, d])
        gt_chains.remove(gt_ch_idx)
        pd_chains.remove(pd_ch_idx)

    if len(gt_chains) != 0:
        # gt_chain_names = [ protein.PDB_CHAIN_IDS[idx] for idx in gt_chains ]
        gt_chain_names = [ encode_chain_id(idx) for idx in gt_chains ]
        print(f"Warning: Chains not match in GT: {gt_chain_names}", file=sys.stderr)

    if len(pd_chains) != 0:
        # pd_chain_names = [ protein.PDB_CHAIN_IDS[idx] for idx in pd_chains ]
        pd_chain_names = [ encode_chain_id(idx) for idx in pd_chains ]
        print(f"Warning: Chains not match in PD: {pd_chain_names}", file=sys.stderr)

    return chain_pair_list

def read_pdb_string(pdb_file):
    pdb_string = open(pdb_file).read()
    if pdb_string.count('\nENDMDL') >= 2:
        #start = pdb_string.index('ENDMDL\n')
        end = pdb_string.index('\nENDMDL')
        pdb_string = pdb_string[:end+7]
    return pdb_string

def calc_TM_score(gt_pdb_file, pd_pdb_file, verbose=False):

    #### Read PDB files

    gt_prot = protein.from_pdb_string(read_pdb_string(gt_pdb_file))
    pd_prot = protein.from_pdb_string(read_pdb_string(pd_pdb_file))

    #### Mapping from chain ID to sequence

    gt_chain2seq = get_chain2seq(gt_prot)
    pd_chain2seq = get_chain2seq(pd_prot)

    #print(gt_chain2seq)
    #print(pd_chain2seq)

    #um_ch_gt = len(gt_chain2seq)
    #num_ch_pd = len(pd_chain2seq)

    chain_pair_list = get_chain_pairs(gt_chain2seq, pd_chain2seq, gt_prot, pd_prot)
    if len(chain_pair_list) == 0:
        print(f"Warning: No chain paired", file=sys.stderr)
        return {
            'RMSD': 0.0,
            'L_common': 0.0,
            'L_target': 0.0,
            'TM-score': 0.0,
            'GDT-TS-score': 0.0,
            'lDDT': 0.0
        }

    #### Concatenate all positions

    gt_ca_pos_list = []
    pd_ca_pos_list = []

    for gt_ch_idx, pd_ch_idx, d in chain_pair_list:
        
        # gt_ch_name = protein.PDB_CHAIN_IDS[gt_ch_idx]
        gt_ch_name = encode_chain_id(gt_ch_idx)
        # pd_ch_name = protein.PDB_CHAIN_IDS[pd_ch_idx]
        pd_ch_name = encode_chain_id(pd_ch_idx)
        if verbose:
            print(f"GT={gt_ch_name}, PD={pd_ch_name}, dist={round(d, 3)}")
        gt_seq = gt_chain2seq[gt_ch_idx]
        pd_seq = pd_chain2seq[pd_ch_idx]
        gt_ca_pos, pd_ca_pos = get_matched_pos(gt_seq, pd_seq, gt_ch_idx, pd_ch_idx, gt_prot, pd_prot)
        gt_ca_pos_list.append(gt_ca_pos)
        pd_ca_pos_list.append(pd_ca_pos)

    gt_ca_pos = np.concatenate(gt_ca_pos_list)
    pd_ca_pos = np.concatenate(pd_ca_pos_list)

    L_common = gt_ca_pos.shape[0]
    if verbose:
        print(f"L_common: {L_common}")

    #### Caluclate TM scores

    L_target = int(sum(gt_prot.atom_mask[:, 1]))
    if verbose:
        print(f"L_target: {L_target}")
    di = np.sqrt(np.sum((gt_ca_pos - pd_ca_pos)**2, 1))
    d0_L_target = 1.24 * np.power( L_target - 15, 1/3 ) - 1.8
    TM_score = (1/L_target)*np.sum(1 / (1 + (di / d0_L_target) ** 2))
    # print(d0_L_target)

    #### Caluclate GDT-TS scores

    GDT_TS = np.mean([ np.mean(di < 1.0), np.mean(di < 2.0), np.mean(di < 4.0), np.mean(di < 8.0) ])
    GDT_TS = L_common / L_target * GDT_TS

    #### Caluclate GDT-TS scores
    gt_dist = np.sqrt(np.sum( (gt_ca_pos[:, None] - gt_ca_pos[None, :])**2, 2 ))
    pd_dist = np.sqrt(np.sum( (pd_ca_pos[:, None] - pd_ca_pos[None, :])**2, 2 ))
    lDDT = calculate_LDDT(gt_dist, pd_dist, R=15.0, sep_thresh=-1, T_set=[0.5, 1, 2, 4])

    metrics = {
        'RMSD': np.mean(di),
        'L_common': L_common,
        'L_target': L_target,
        'TM-score': TM_score,
        'GDT-TS-score': GDT_TS,
        'lDDT': lDDT
    }

    return metrics


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage: TMscore.py model.pdb native.pdb")
        exit(1)

    model_pdb = sys.argv[1]
    native_pdb = sys.argv[2]

    metrics = calc_TM_score(native_pdb, model_pdb, True)
    RMSD = round(metrics['RMSD'], 3)
    TMscore = round(metrics['TM-score'], 3)
    GDT_TS = round(metrics['GDT-TS-score'], 3)
    lDDT = round(metrics['lDDT'], 3)
    print(f"RMSD={RMSD}")
    print(f"TMscore={TMscore}")
    print(f"GDT-TS-score={GDT_TS}")
    print(f"lDDT={lDDT}")

