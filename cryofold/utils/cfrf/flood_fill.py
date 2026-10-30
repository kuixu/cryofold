"""
MIT License

Copyright (c) 2022 Kiarash Jamali

This file is modified from [https://github.com/3dem/model-angelo/blob/main/model_angelo/gnn/flood_fill.py].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
"""
import os

import numpy as np
from collections import namedtuple
from typing import Dict
import torch
from scipy.spatial import cKDTree
from cryofold.utils.cfrf.save_pdb_utils import chain_atom14_to_cif
from cryofold.utils.cfrf.protein import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from cryofold.utils.cfrf.residue_constants import restype_atom14_mask,select_torsion_angles
from cryofold.utils.cfrf.affine_utils import get_affine_translation
FloodFillChain = namedtuple("FloodFillChain", ["start_N", "end_C", "residues"])

def normalize_local_confidence_score(
    local_confidence_score: np.ndarray,
    best_value: float = 0.15,
    worst_value: float = 0.85,
) -> np.ndarray:
    normalized_score = (worst_value - local_confidence_score) / (
        worst_value - best_value
    )
    normalized_score = np.clip(normalized_score, 0, 1)
    return normalized_score

def remove_overlapping_ca(
    ca_positions: np.ndarray,bfactors,existence_mask=None,radius_threshold: float = 0.3,
) -> np.ndarray:
    kdtree = cKDTree(ca_positions)
    bfactors_copy = np.copy(bfactors)
    sorted_indices = np.argsort(bfactors_copy)[::-1]
    if existence_mask is None:
        existence_mask = np.ones(len(ca_positions), dtype=bool)

    for i in sorted_indices:
        if existence_mask[i]:
            too_close = np.array(
                kdtree.query_ball_point(ca_positions[i], r=radius_threshold,)
            )
            too_close = too_close[too_close != i]
            existence_mask[too_close] = False
    return existence_mask


def final_results_to_cif(
    final_results,
    cif_path,
    sequences=None,
    aatype=None,
    verbose=False,
    print_fn=print,
    aggressive_pruning=False,
    mask_threshold = 0.3,
    end_flag = False,
    protein=None
):
    """
    Currently assumes the ordering it comes with, I will change this later
    """
    bfactors = normalize_local_confidence_score(final_results["local_confidence"]) * 100
    backbone_affine = torch.from_numpy(final_results["pred_affines"])
    existence_mask = (
        torch.from_numpy(final_results["existence_mask"]).sigmoid() > mask_threshold
    ).numpy()
    existence_mask = remove_overlapping_ca(ca_positions=get_affine_translation(backbone_affine),bfactors=bfactors,existence_mask=existence_mask,radius_threshold=0.65 if end_flag else 0.3)
    
    # if aatype is None:
    pd_aatype = np.argmax(final_results["aa_logits"], axis=-1)[existence_mask]
    # print("gt aatype: ", aatype)
    # print("pd aatype: ", pd_aatype)
    backbone_affine = backbone_affine[existence_mask]
    mask2unmask = np.arange(len(existence_mask))[existence_mask]

    torsion_angles = select_torsion_angles(
        torch.from_numpy(final_results["pred_torsions"][existence_mask]), aatype=aatype
    )
    edge_logits = final_results["edge_logits"][existence_mask]
    edge_index = final_results['edge_index'][existence_mask]
    all_frames = torsion_angles_to_frames(aatype, backbone_affine, torsion_angles)
    all_atoms = frames_and_literature_positions_to_atom14_pos(aatype, all_frames)
    atom_mask = restype_atom14_mask[aatype]
    bfactors = (
        normalize_local_confidence_score(
            final_results["local_confidence"][existence_mask]
        )
        * 100
    )

    all_atoms_np = all_atoms.numpy()
    if protein is None:
        raise ValueError("protein is None")
        # chains = ortools_build_path(all_atoms_np[:,[0,2]],edge_logits)
        # chains = flood_fill(all_atoms_np, bfactors,edge_logits,edge_index,mask2unmask)
    else:
        chain_ids = protein.chain_index
        chains = [np.where(chain_ids==i)[0] for i in range(np.unique(chain_ids).shape[0])]

    print("chains: ", len(chains))
    # if end_flag:
    #     chains = ortools_build_path(all_atoms_np[:,[0,2]],edge_logits)
    # else:
    #     chains = flood_fill(all_atoms_np, bfactors)
    chains_concat = np.concatenate(chains)

    # Prune chains based on length
    if sequences is not None or end_flag:
        pruned_chains = [c for c in chains if len(c) > 2]
    else:
        pruned_chains = chains
    chain_atom14_to_cif(
        [aatype[c] for c in pruned_chains],
        [all_atoms[c] for c in pruned_chains],
        [atom_mask[c] for c in pruned_chains],
        cif_path,
        bfactors=[bfactors[c] for c in pruned_chains],
    )

    new_final_results = dict(
        [(k, v[chains_concat]) for (k, v) in final_results.items()]
    )
    new_final_results["chain_aa_logits"] = [
        final_results["aa_logits"][existence_mask][c] for c in chains
    ]
    new_final_results["pruned_chain_aa_logits"] = [
        final_results["aa_logits"][existence_mask][c] for c in pruned_chains
    ]

    return new_final_results
def BayesCoreect(possible_indices,idx,dists,edge_logits,edge_index,mask2unmask,eps=1e-3):
    # possible_logits = []
    # for poi in possible_indices:
    #     if np.any(edge_index[idx]==mask2unmask[poi]):
    #         possible_logits.append(edge_logits[idx][edge_index[idx]==mask2unmask[poi]])
    #     else:
    #         possible_logits.append(-1)
    possible_logits = [edge_logits[idx][edge_index[idx]==mask2unmask[poi]] for poi in possible_indices]
    possible_logits = np.array(possible_logits).flatten()
    # possible_logits[possible_logits == -1] = np.max(possible_logits)
    dist_logits = 1-np.exp(-(1.4/(dists+eps))**6)
    return np.argsort(1-dist_logits*possible_logits)
    
def flood_fill(atom14_positions, b_factors,edge_logits,edge_index,mask2unmask,n_c_distance_threshold=2.1):
    n_positions = atom14_positions[:, 0]
    c_positions = atom14_positions[:, 2]
    kdtree = cKDTree(c_positions)
    b_factors_copy = np.copy(b_factors)

    chains = []
    chain_ends = {}
    while np.any(b_factors_copy != -1):
        idx = np.argmax(b_factors_copy)
        possible_indices = np.array(
            kdtree.query_ball_point(
                n_positions[idx],
                r=n_c_distance_threshold,
                return_sorted=True
            )
        )
        possible_indices = possible_indices[possible_indices != idx]
        got_chain = False
        if len(possible_indices) > 0:
            pos_dits = np.sqrt(np.sum(np.square(n_positions[idx][None] - c_positions[possible_indices]), axis=-1))
            possible_indices = possible_indices[BayesCoreect(possible_indices, idx, pos_dits, edge_logits[...,1], edge_index, mask2unmask)]
            for possible_prev_residue in possible_indices:
                if possible_prev_residue == idx:
                    continue
                if possible_prev_residue in chain_ends:
                    chains[chain_ends[possible_prev_residue]].append(idx)
                    chain_ends[idx] = chain_ends[possible_prev_residue]
                    del chain_ends[possible_prev_residue]
                    got_chain = True
                    break
                elif b_factors_copy[possible_prev_residue] >= 0.0:
                    chains.append([possible_prev_residue, idx])
                    chain_ends[idx] = len(chains) - 1
                    b_factors_copy[possible_prev_residue] = -1
                    got_chain = True
                    break

        if not got_chain:
            chains.append([idx])
            chain_ends[idx] = len(chains) - 1

        b_factors_copy[idx] = -1

    og_chain_starts = np.array([c[0] for c in chains])
    og_chain_ends = np.array([c[-1] for c in chains])

    chain_starts = og_chain_starts.copy()
    chain_ends = og_chain_ends.copy()

    n_chain_starts = n_positions[chain_starts]
    c_chain_ends = c_positions[chain_ends]
    N = len(chain_starts)
    spent_starts, spent_ends = set(), set()

    kdtree = cKDTree(n_chain_starts)

    no_improvement = 0
    chain_end_match = 0

    while no_improvement < 2 * N:
        found_match = False
        if chain_end_match in spent_ends:
            no_improvement += 1
            chain_end_match = (chain_end_match + 1) % N
            continue

        start_matches = kdtree.query_ball_point(
            c_chain_ends[chain_end_match], r=n_c_distance_threshold, return_sorted=True
        )
        if len(start_matches)>0:
            start_matches = np.array(start_matches)
            pos_dits = np.sqrt(np.sum(np.square(c_chain_ends[chain_end_match][None] - n_chain_starts[start_matches]), axis=-1))
            start_matches = start_matches[BayesCoreect(og_chain_starts[start_matches], og_chain_ends[chain_end_match], pos_dits, edge_logits[...,0], edge_index, mask2unmask)]
        for chain_start_match in start_matches:
            if (
                chain_start_match not in spent_starts
                and chain_end_match != chain_start_match
            ):
                chain_start_match_reidx = np.nonzero(
                    chain_starts == og_chain_starts[chain_start_match]
                )[0][0]
                chain_end_match_reidx = np.nonzero(
                    chain_ends == og_chain_ends[chain_end_match]
                )[0][0]
                if chain_start_match_reidx == chain_end_match_reidx:
                    continue

                new_chain = (
                    chains[chain_end_match_reidx] + chains[chain_start_match_reidx]
                )

                chain_arange = np.arange(len(chains))
                tmp_chains = np.array(chains, dtype=object)[
                    chain_arange[
                        (chain_arange != chain_start_match_reidx)
                        & (chain_arange != chain_end_match_reidx)
                    ]
                ].tolist()
                tmp_chains.append(new_chain)
                chains = tmp_chains

                chain_starts = np.array([c[0] for c in chains])
                chain_ends = np.array([c[-1] for c in chains])

                spent_starts.add(chain_start_match)
                spent_ends.add(chain_end_match)
                no_improvement = 0
                found_match = True
                chain_end_match = (chain_end_match + 1) % N
                break

        if not found_match:
            no_improvement += 1
            chain_end_match = (chain_end_match + 1) % N

    return chains