#!/usr/bin/env python

import pathlib, argparse, json, os, sys, tempfile, re, shutil, glob, pickle, gzip, random
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import List, Union, Optional
import multiprocessing as mp
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from cryofold.diff.structure.protein_chain import ProteinChain
from cryofold.diff.structure.protein_complex import ProteinComplex, protein_chain_file_list_to_protein_complex
from cryofold.model.sm import CF
from cryofold.model.api import CFProtein, GenerationConfig

from cryofold.data.msa_template import run_phmmer_seqdb, build_templates_from_phmmer_cifdb
from cryofold.np import protein, residue_constants
from cryofold.utils import General, _info, _warn, _error
from cryofold.model.backbone_imp import build_backbone_imp
from cryofold.model.refinement import do_refinement, do_refinement_best
# from cryofold.utils.superimposition import superimpose
from cryofold.utils.cn.chain_comparison   import chain_comparison
from cryofold.libs.structure.density  import DensityInfo, mol_atom_density_np, mol_atom_density_th


NCPU = 32

load_fasta  = General.load_fasta
# cpx = '_internal/cpx/bin/cpythonx'
# if not os.path.exists(cpx):
#     cpx = 'tools/cpx/bin/cpythonx'
#     assert os.path.exists(cpx)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def annotate_multimer_seqs(name2seq, seq_info_dict, identity_cutoff=0.8):
    seq2names = {}
    for name, seq in name2seq.items():
        seq2names[seq] = seq2names.get(seq, []) + [ name ]
    
    for seq in seq2names:
        names = seq2names[seq]
        if seq in seq_info_dict:
            continue
        num_rep = len(names)

####################################################
### Check input sequences and filter invalid sequences
####################################################


def check_seq_dict(name2seq, min_num_res=15, max_invalid_aa_rate=0.9, max_ATCGU_rate=0.95):
    """
    Remove non-protein sequence and sequence with too manu UNK residues

    Parameters
    -----------------
    name2seq: dict of name -> sequence
    min_num_res: int
        Minimun length of sequence
    max_invalid_aa_rate: float
    max_ATCGU_rate: float

    Return
    ----------------
    None
    """
    
    for name in list(name2seq.keys()):
        seq = name2seq[name]
        if len(seq) < min_num_res:
            print(_warn(f"Warning: seq {seq} too short, skip it"))
            del name2seq[name]
            continue
        num_wrong_aa = sum([ aa not in residue_constants.restypes for aa in seq ])
        if num_wrong_aa / len(seq) > max_invalid_aa_rate:
            print(_warn(f"Warning: {num_wrong_aa} residues in {seq} not canonical amino acid type skip it"))
            del name2seq[name]
            continue
        na_code_num = seq.count('A') + seq.count('T') + seq.count('C') + seq.count('G') + seq.count('U')
        if na_code_num / len(seq) > max_ATCGU_rate:
            print(_warn(f"Warning: It seems too many nucleotide residues -- {seq}, skip it"))
            del name2seq[name]
            continue

####################################################
### Init config_dict
####################################################

def build_config_dict(seq2names, seq_info_dict):
    """
    Build the configure dictionary from seq2names

    Parameters
    ------------------
    seq2names: dict
        A mapping from name -> sequence
    seq_info_dict: dict
        A mapping from name -> seq_type
        seq_type can be 'Antibody', 'Multimer' (d is a integer), 'Monomer'
    
    Return
    ------------------
    config_dict: dict
        A dictionary to record the group information
    """
    config_dict = {
        "cryonet": "",      # modified to backbone
        "map_file": "",
        "group2templ_info": {},   # to be improved
        "group2templ": {}   # to be improved
    }
    
    seq_idx = 1
    for group_idx, (seq, names) in enumerate(seq2names.items()):

        # print("group_idx:", group_idx)
        # print("seq in seq_info_dict:", seq in seq_info_dict)
        # print("seq_info_dict[seq]:", seq_info_dict[seq])

        rep_num  = len(names)
        seq_type = 'Monomer'
        do_func  = 'split_templ_sword'
        if seq in seq_info_dict:
            seq_type = seq_info_dict[seq]['type']    # Antibody, Multimer, Monomer
            rep_num  = seq_info_dict[seq]['rep_num'] # # of repeates
            do_func  = seq_info_dict[seq]['do_func'] # # of repeates
            new_seq  = seq_info_dict[seq]['seq'] # Intergrated sequence
            seq      = new_seq
        config_dict['group2templ_info'][group_idx] = {
            'seq':     seq,
            'rep_num': rep_num,
            'type':    seq_type,
            'do_func': do_func,
            'pred_templ':[
                f'msa_idx_0.cif',
                f'msa_idx_1.cif',
                f'msa_idx_2.cif',
                f'msa_idx_3.cif',
                f'msa_idx_4.cif'
                ]
        }

    return config_dict




def load_json_config(config_json_file):
    """
    Load config json file, convert template id from str to int to avoid key error
    """
    config_dict = json.load(open(config_json_file))
    
    for name in ['group2templ', 'group2templ_info', 'cryonet_sup', 'cryonet_sup_dock', 'cryonet_sup_dock_da']:
        if name not in config_dict:
            continue
        for key in list(config_dict[name].keys()):
            int_key = int(key)
            config_dict[name][int_key] = config_dict[name][key]
            del config_dict[name][key]
            if 'tid' in config_dict[name][int_key]:
                config_dict[name][int_key]['tid'] = [ int(d) for d in config_dict[name][int_key]['tid'] ]
    
    return config_dict



class NpEncoder(json.JSONEncoder):
    """
    Avoid json.dump fail
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

def map_file_from_config(config: dict, resolution: float) -> str:
    """
    Decide which .map/.mrc file to be used by resolution value.

    Parameters
    --------------
    config: dict
    resolution: float

    Return
    --------------
    map_file: str
    """
    if resolution>4.0 and 'bb_map_file' in config:
        if os.path.exists(config['bb_map_file']):
            return config['bb_map_file']
        else:
            print(_error(f"Warning: bb map file {config['bb_map_file']} not exists, use {config['map_file']} as replacement."))
            return config['map_file']
    else:
        return config['map_file']


def prot2density(prot:ProteinChain, voxel_size:float=1.0, 
    atom_weight:float=14.0, res:float=3.0, datatype='numpy')->DensityInfo:
    """
    Convert ProteinChain object to Density object
    """
    assert datatype in ('numpy', 'torch')
    
    atom_coords = prot.atom37_positions[ prot.atom37_mask ]
    if datatype == 'numpy':
        density, offset = mol_atom_density_np(atom_coords, atom_weight=atom_weight, res=res, voxel_size=voxel_size)
    else:
        density, offset = mol_atom_density_th(atom_coords, atom_weight=atom_weight, res=res, voxel_size=voxel_size)
    den_sim = DensityInfo(density=density, offset=offset, apix=voxel_size, datatype='torch')
    return den_sim

def erase_map_with_pdb(pdbpath, mappath, outpath, resolution=3.0):
    """
    Erase map with PDB model
    """
    # print(_info(f"erasing map with: {pdbpath}"))
    den_exp = DensityInfo(mappath)

    if type(pdbpath) == list:
        # tmp_pdb = tempfile.NamedTemporaryFile(delete=True, suffix=".pdb").name
        tmp_pdb = mappath.replace('.mrc', f'_aem_cm{len(pdbpath)}.pdb')
        combine_pdbs(pdbpath, tmp_pdb)
        # import pdb;pdb.set_trace()

        pdbpath = tmp_pdb
    # prot = af2_features.read_pdb(pdbpath)
    prot = ProteinChain.from_pdb_full(pdbpath)
    denobj_mol = prot2density(prot, voxel_size=den_exp.apix, atom_weight=14.0, res=resolution)
    mol_den = denobj_mol.density

    mask = mol_den!=0
    thr = (mol_den*mask).sum()/mask.sum()
    mask_one = mol_den<thr
    mask_zer = mol_den>=thr
    mol_den[mask_one]  = 1.0
    mol_den[mask_zer] = 0.0

    denobj_mol_mask = DensityInfo(density=mol_den, offset=denobj_mol.offset, apix=denobj_mol.apix)

    masked_density, masked_offset = den_exp.mask(denobj_mol_mask, keep_origin=True)
    if masked_density.numel() ==0:
        print(_warn("warning!!! zero erasing. "))
    else:
        masked_den = DensityInfo(density=masked_density, offset=masked_offset, apix=den_exp.apix)
        masked_den.save(outpath)

def combine_pdbs(pdb_files, out_file):
    # pdbpaths = ' '.join(pdb_files)
    # cmd = f'{cpx} --nogui --cmd "open {pdbpaths}; combine ; save {out_file} #{len(pdb_files)+1};exit;" >> /dev/null 2>&1'
    # os.system(cmd)

    pcompl = protein_chain_file_list_to_protein_complex(pdb_files)
    pcompl.to_pdb(out_file)


def final_pdbs(filter, out_file, should_be):
    pdb_files = glob.glob(filter.replace("#","*"))
    # for i in range(len(pdb_files)):
    #     print(i+1, pdb_files[i])
    if int(should_be) == len(pdb_files):
        combine_pdbs(pdb_files, out_file)
        # print("out file: ", out_file)
    else:
        combine_pdbs(pdb_files, out_file.replace("_cf.pdb", "_cf0.pdb"))
        # print(f'Should be {should_be} but got {len(pdb_files)}')

def cb_pred(pdb1, pdb2, out_file):
    if pdb1.lower().endswith(('.pdb')):
        pc1 = ProteinChain.from_pdb_full(pdb1)
    else:
        pc1 = ProteinChain.from_cif(pdb1)
    
    if pdb2.lower().endswith(('.pdb')):
        pc2 = ProteinChain.from_pdb_full(pdb2)
    else:
        pc2 = ProteinChain.from_cif(pdb2)

    mobile_inds=list(range(pc2.atom37_positions.shape[0]))
    target_inds=list(range(pc1.atom37_positions.shape[0]))
    pc2_ali = pc2.align(pc1, mobile_inds, target_inds)
    pc2_ali.to_pdb(out_file)
    # breakpoint()
    return pc2_ali
    # pc1.atom37_mask
    # pc1.atom37_positions
    # breakpoint()
    # # superimposed_pred, alignment_rmsd = superimpose(
    #                     gt_coords_masked_ca, pred_coords_masked_ca, all_atom_mask_ca,
    #                 )


def is_overlap_with_pre(path, b_path, overlap_thr=0.5):
    met_inter2 = chain_comparison(path, b_path, return_results=True)
    is_overlap = met_inter2['found'] >= overlap_thr
    return is_overlap


def is_two_pdb_overlap(pdbfile1, pdbfile2, overlap_thr=0.5, mode='both') -> bool:
    """
    Measure overlap between two PDB files.

    Parameters
    --------------
    pdbfile1: str
    pdbfile2: str
    overlap_thr: Cutoff Found value to define overlap
    mode: left, right, both.

    Return
    --------------
    overlaped: bool
    """
    assert mode in ('left', 'right', 'both')
    met_inter1 = chain_comparison(pdbfile1, pdbfile2, return_results=True)
    met_inter2 = chain_comparison(pdbfile2, pdbfile1, return_results=True)

    if mode == 'left':
        overlaped = (met_inter1['found'] >= overlap_thr)
    elif mode == 'right':
        overlaped = (met_inter2['found'] >= overlap_thr)
    else:
        overlaped = (met_inter1['found'] >= overlap_thr) or (met_inter2['found'] >= overlap_thr)
    # print("found: ", met_inter1['found'])
    return overlaped

def is_overlap(dok_file, domain_paths, overlap_thr=0.5, disable_mp=False):
    """
    Compare dok_file with each PDB file in domain_paths.
    """
    if domain_paths is None or len(domain_paths)==0:
        return False, []

    result_list = []
    overlapped_paths = []
    
    if disable_mp:
        for dpath in domain_paths:
            overlaped = is_two_pdb_overlap(dok_file, dpath, overlap_thr)
            result_list.append(overlaped)
            if overlaped:
                overlapped_paths.append(dpath)
    else:
        # print(f"{NCPU} process to check overlap.")
        with mp.Pool( min(NCPU, os.cpu_count()) ) as pool:
            h_list = []
            for dpath in domain_paths:
                h = pool.apply_async(is_two_pdb_overlap, (dok_file, dpath, overlap_thr))
                h_list.append([h, dpath])
            for h, dpath in h_list:
                overlaped = h.get()
                result_list.append(overlaped)
                if overlaped:
                    overlapped_paths.append(dpath)

    return any(result_list), overlapped_paths

def nms_pdb(df, accept_path_list, accept_gid_list, overlap_thr=0.5, cc_thr=0.2, disable_mp=False, keep_rep_num=False, group2templ_info=None):
    domain_paths = df['path'].tolist()
    dokfile_gid = df['gid'].tolist()
    if len(accept_path_list)==0:
        # cid = df['cid'].tolist()[0]
        accept_list = [0]
        s = 1
        # wait_list = domain_paths[1:]

    else:
        accept_list = []
        s = 0
        # wait_list = domain_paths
    reject_list = []
    for i in range(s, len(domain_paths)):
        dok_file = domain_paths[i]
        dok_gid = dokfile_gid[i]
        if len(accept_list) == 0:
            accept_pdbs = accept_path_list
            accept_gids = accept_gid_list
        else:
            accept_pdbs = df.loc[accept_list, 'path'].tolist() + accept_path_list
            accept_gids = df.loc[accept_list, 'gid'].tolist() + accept_gid_list
        if keep_rep_num and len(accept_gids)>1:
            # check this `dok_file` reach the rep_num
            # import pdb;pdb.set_trace() 
            gids_, counts_ = np.unique(accept_gids, return_counts=True)
            curr_gid_rep_num_dict = {}
            for gid, count in zip(gids_, counts_):
                curr_gid_rep_num_dict[gid] = count
            curr_rep_num = curr_gid_rep_num_dict.get(dok_gid, 0)
            rep_num = group2templ_info[dok_gid]['rep_num']
            # import pdb;pdb.set_trace() 

            if curr_rep_num>=rep_num:
                print(f"gid ({rep_num}) used {curr_rep_num} times.")
                reject_list.append(i)
                continue
        overlapped, overlapped_paths = is_overlap(dok_file, accept_pdbs, overlap_thr, disable_mp)
        if overlapped:
            reject_list.append(i)
        else:
            accept_list.append(i)
    # import pdb; pdb.set_trace()
    return accept_list, reject_list



def refine_best(mappath, pdbpath, outpath, bb_pdb_path, resolution=3.0, gid=0, tid=0, cid=0, gtid=0, n=1, do_refine=False, den_thr=0.0):

    # results = do_refinement_best(pdbpath, mappath, outpath, resolution=resolution, do_refine=do_refine, den_thr=den_thr)
    results = do_refinement(pdbpath, mappath, outpath, resolution=resolution, do_refine=do_refine, den_thr=den_thr, n=n)
    # print(results)
    if len(results)>0:
        for i in range(len(results)):
            results[i]['gid'] = gid
            results[i]['tid'] = tid
            results[i]['cid'] = cid
            results[i]['gtid'] = gtid
            results[i]['rep_num'] = n
            results[i]['stype'] = 1
            # del results[i]['found']
            # del results[i]['chain_match']
        return results
    else:
        return None

def refine_all_best(mapfile_whole, group2templ, dock_met_path, accept_gtid_list, resolution=4.0, den_thr=None, it=0, do_refine=False):

    # mapfile_whole = map_file_from_config(config, resolution)
    bb_pdb_path = None
    # template2path = config['group2templ']
    # templ_list = []
    # tid_list = []
    # gid_list = []
    # gra_base = [0]
    # for k,v in  template2path.items():
    #     len_v = len(v)
    #     templ_list.extend(v)
    #     gid_list.extend([k]*len_v)
    #     tid_list.extend(list(range(len_v)))
    #     gra_base.append(gra_base[-1]+v[0]['rep_num'])
    tid_list = group2templ['tid']
    gid_list = group2templ['gid']
    gtid_list = group2templ['gtid']
    templ_list_bak = group2templ['templs']
    templ_list = []
    # templ_copy = group2templ['copy_num']
    # curr_chain_num = len(df)
    curr_chain_num = 0

    for id in range(len(templ_list_bak)):
        if id not in accept_gtid_list:
            templ_list.append(templ_list_bak[id])
    
    if len(accept_gtid_list) == 0:
        templ_list = templ_list_bak

    else:
        accept_gtid_list_uq, accept_gtid_count = np.unique(accept_gtid_list,return_counts=True)
        for gtid, count in zip(accept_gtid_list_uq, accept_gtid_count):
            if gtid >= len(templ_list_bak):
                print(gtid, accept_gtid_list_uq, accept_gtid_count)
                # import pdb;pdb.set_trace()
            if count < templ_list_bak[gtid]['rep_num']:
                templ_list.append(templ_list_bak[gtid])
    if len(templ_list)==0:
        import pdb;pdb.set_trace()
    # met_path = mapfile_whole.replace('.mrc', f'_dock_all.csv')
    met_path = dock_met_path
    if os.path.exists(met_path):
        print(f"Found {met_path} file, loading...")

        dock_df = pd.read_csv(met_path)
    else:
        print(f"Not found {met_path} file, perform docking on {len(templ_list)} templates...")
        dock_results = []
        with mp.Pool( min(NCPU, os.cpu_count()) ) as pool:
            h_list = []
            # path_list = [ p for p in df['path'] if p is not None ]
            for k in tqdm(range(0, len(templ_list))):
                tpa = templ_list[k]['path']
                tpo = tpa.replace('.pdb', f'_aem_it{it}.pdb')
                tid = tid_list[k]
                gid = gid_list[k]
                gtid = gtid_list[k]
                tcn = templ_list[k]['rep_num']
                cid = k + curr_chain_num + 1
                # if gtid in accept_gtid_list:
                #     print(gtid)
                #     continue
                print(tpa)

                h = pool.apply_async(refine_best, (mapfile_whole, tpa, tpo, bb_pdb_path, resolution, gid, tid, cid, gtid, tcn, do_refine, den_thr))
                h_list.append(h)
                
            for h in tqdm(h_list):
                result = h.get()
                if result is not None:
                    if isinstance(result, list):
                        dock_results.extend(result)
                    else:
                        dock_results.append(result)
        dock_df = pd.DataFrame(dock_results)
        # if len(dock_df) ==0:
        #     import pdb;pdb.set_trace()
        sorted_indices = (dock_df["corr"]).sort_values(ascending=False).index
        dock_df = dock_df.loc[sorted_indices]
        
        dock_df = dock_df.reset_index(drop=True)
        dock_df.to_csv(met_path)
    return dock_df, met_path

def cp_all_chains_best( config, mappath, resolution=4.0, den_thr=0.0, cc_thr=0.2, overlap_thr=0.5, keep_rep_num=False, do_refine=False):

   
    # config = load_json_config(jsonfile)
    key = 'group2templ' 
    
    # ma = map_file_from_config(config, resolution)
    
    
    group2templ_info = config['group2templ_info']
    chain_num = 0
    for k,v in  group2templ_info.items():
        chain_num += v['rep_num']

    template2path = config['group2templ']

    templ_list = []
    tid_list = []
    gid_list = []
    gra_base = [0]

    
    final_path = mappath.replace(".mrc", "_cf.pdb")
    sup_dir = os.path.dirname(final_path) +  f'/sup'
    wtempl_dir = os.path.dirname(final_path) +  f'/.templ'
    os.system(f"mkdir -p {sup_dir}")
    pc_list = []
    for gidx,v0 in  template2path.items():
        # v = v0[:10] 
        # len_v = len(v)
        # templ_list.extend(v)
        # gid_list.extend([k]*len_v)
        # tid_list.extend(list(range(len_v)))
        # gra_base.append(gra_base[-1]+v[0]['rep_num'])
        
        templ_dir = f'{wtempl_dir}/{gidx}'
        prd_path = group2templ_info[gidx]['pred_templ'][0]
        prd_path = f'{templ_dir}/{prd_path}'.replace(".cif",".pdb")
        v = [{'path':prd_path,'rep_num':v0[0]['rep_num']}]
        len_v = len(v)
        templ_list.extend(v)
        gid_list.extend([k]*len_v)
        tid_list.extend(list(range(len_v)))
        gra_base.append(gra_base[-1]+v0[0]['rep_num'])
    group2templ= {
        'gid':gid_list,
        'tid':tid_list,
        'gtid':list(range(len(tid_list))),
        'templs': templ_list,
    }
    accept_gtid_list = []
    accept_gid_list  = []
    accept_path_list = []
    global_accept_path_list = []
    it = 0
    it_empty = 0
    erasedmapfile2 = mappath

    # while(len(accept_gtid_list) < len(templ_list)):
    while(len(accept_gtid_list) < chain_num):

        print(f"> {it} {len(accept_gtid_list)}/{chain_num}")

        dock_met_path = mappath.replace('.mrc', f'_aem_it{it}.csv')
        if not os.path.exists(dock_met_path):
            dock_df, dock_met_path = refine_all_best(
                erasedmapfile2, group2templ, dock_met_path, 
                accept_gtid_list, den_thr=den_thr, it=it, do_refine=do_refine)
        else:
            dock_df = pd.read_csv(dock_met_path)
        print(dock_df)
        print(f"> {it} do filter corr({cc_thr})")

        dock_df_ = dock_df.loc[dock_df['corr']>=cc_thr]
        if len(dock_df_)>0:
            dock_df = dock_df_
        else:
            topk = min(len(dock_df),2)
            topk_list = list(range(topk))
            dock_df = dock_df.loc[topk_list]
        print(dock_df)

        print(f"> {it} do NMS")
        # if it ==1:
        # import pdb; pdb.set_trace()


        accept_list, reject_list = nms_pdb(
            dock_df, accept_path_list, accept_gid_list, overlap_thr=overlap_thr, 
            cc_thr=cc_thr, keep_rep_num=keep_rep_num, group2templ_info=group2templ_info)
        accept_df = dock_df.loc[accept_list]
        # accept_tid_list = accept_df['tid'].tolist() + accept_tid_list
        accept_gtid_list = accept_df['gtid'].tolist() + accept_gtid_list
        accept_gid_list  = accept_df['gid'].tolist()  + accept_gid_list

        accept_path_list = accept_df['path'].tolist()
        global_accept_path_list = accept_path_list + global_accept_path_list
        print(f"> {it} accept_list at it - {it}")
        print(dock_df.loc[accept_list])

        if len(accept_path_list) ==0:
            pdb_paths_for_erase = global_accept_path_list
            resolution_for_erase = resolution + 1 + it_empty
            it_empty += 1

        else:
            pdb_paths_for_erase = accept_path_list
            resolution_for_erase = resolution 

        # erasedmapfile = 
        erasedmapfile1 = erasedmapfile2
        erasedmapfile2 = mappath.replace('.mrc', f'_aem_erased_it{it}.mrc')
        if not os.path.exists(erasedmapfile2):
            print(f"> {it} erasing map... with {resolution_for_erase}({resolution})")
            print(f"> {it} erasing map... with {pdb_paths_for_erase})")
            erase_map_with_pdb(pdb_paths_for_erase, erasedmapfile1, erasedmapfile2, resolution=resolution_for_erase)
        it += 1

    # if int(chain_num) == len(global_accept_path_list):
    #     # chain_new = ProteinChain.my_concat(pc_list)
    #     # chain_new.to_pdb(final_path)

    #     pcompl = protein_chain_file_list_to_protein_complex(global_accept_path_list)
    #     pcompl.to_pdb(final_path)

    #     print("global_accept_path_list: ")
    #     print(global_accept_path_list)
    #     print("Predicted file: ", final_path)
    # else:
    #     print(f"Warning: {len(global_accept_path_list)} chains are not used for final complex.")
    #     breakpoint()
    # pcompl = protein_chain_list_to_protein_complex(pc_list)
    pcompl = protein_chain_file_list_to_protein_complex(global_accept_path_list)
    pcompl.to_pdb(final_path)
    
    if int(chain_num) != len(global_accept_path_list):
        # chain_new = ProteinChain.my_concat(pc_list)
        # chain_new.to_pdb(final_path)
        final_path2 = mappath.replace(".mrc", f"_cf_{len(global_accept_path_list)}-{chain_num}.pdb")
        pcompl.to_pdb(final_path2)
    



def run_inference_de(input_fasta, input_pklgz, input_mrc, work_dir, resolution=4.0, 
                     den_thr=None, exclude_pdbs=[], max_pdb_date=None, max_pdb_templ=1000, 
                    ncores=10, min_pdb_templ_cov=0.2, min_identity=0.2, cus_templ_path=""):
    """
    build template
    :param input_fasta:
    :param work_dir:
    :param exclude_pdbs:
    :param max_pdb_date:
    :param max_pdb_templ:
    :param ncores:
    :param min_pdb_templ_cov:
    :param min_identity:
    :param map_file:
    :param cifdb_file:
    :return:
    """
    # assert pdb_mode in ('none', 'phmmer', 'exactly')
    # assert afdb_mode in ('none', 'phmmer', 'exactly')
    # assert af2_mode in ('af2', 'omegafold')
    # if os.path.exists(output_json):
    #     os.system(f"rm {output_json}")
    assert os.path.exists(input_fasta)
    assert os.path.exists(input_pklgz)
    assert os.path.exists(input_mrc)

    # work_dir = "/data/jobs/f5/jobs_241203_templ/0448/.templ/0/msa_idx_0_aem"
    # global_accept_path_list = glob.glob(f"{work_dir}*.pdb")
    # print(f"{len(global_accept_path_list)}")
    # final_path = input_fasta.replace(".fasta", "_cf.pdb")
    # pcompl = protein_chain_file_list_to_protein_complex(global_accept_path_list)
    # pcompl.to_pdb(final_path)
    # breakpoint()


    if den_thr is None or den_thr < 0:
        den_map = DensityInfo(input_mrc)
        den_thr = den_map.meanns(3).item()

    features = pickle.load(gzip.open(input_pklgz, 'rb'))
    
    # assert os.path.exists(cifdb_file)
    # if afdb_mode == 'phmmer' and (afdb_phmmer_paths is None or len(afdb_phmmer_paths) == 0):
    #     afdb_phmmer_paths = [ environment.afdb_fasta_path ]
    # breakpoint()
    max_pdb_date_str = max_pdb_date
    if max_pdb_date is None:
        max_pdb_date = datetime.today()
    else:
        max_pdb_date = datetime.strptime(max_pdb_date, "%Y-%m-%d")

    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    name2seq, name2annot = load_fasta(input_fasta, load_annotation=True)
    seqs_str = ""
    for k,v in name2seq.items():
        seqs_str += v.upper()
    aatype = residue_constants.sequence_to_onehot(
      sequence=seqs_str,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True).argmax(axis=1)
    # print("features['aatype']", features['aatype'].shape)
    if  len(features['aatype'].shape) == 2:
        aatype_fea = features['aatype'].argmax(axis=1)
    else:
        aatype_fea = features['aatype']
    # aatype_check = (aatype==aatype_fea).all()
    # assert aatype_check, "sequence in fasta {input_fasta} and *.pkl.gz {pklgz_path} file do not match"
    aatype_check = len(aatype)==len(aatype_fea)
    # if not aatype_check:
    #     print(_error(f"Warning: sequence in fasta {input_fasta} and *.pkl.gz {input_pklgz} file do not match"))
    #     breakpoint()
    assert aatype_check, f"sequence in fasta {input_fasta} {len(aatype)} and *.pkl.gz {input_pklgz} {len(aatype_fea)} file do not match"

    #print("name2seq: ", len(name2seq))
    check_seq_dict(name2seq)

    print("Running model inference...")

    # -----
    model =  CF.from_pretrained("cf_v1.0", device=torch.device("cuda"))
    model.eval()
    for k,v in name2seq.items():
        seqs_str = v.upper()
        protein_prompt = CFProtein(sequence=seqs_str)
        structure_prediction_config = GenerationConfig(
            track="structure", # We want CF3 to generate tokens for the structure track
            num_steps=len(seqs_str) // 8,
            temperature=0.7,
        )
        

        try:
            structure_prediction,_ = model.generate(protein_prompt, structure_prediction_config)
            del structure_prediction
        except :
            structure_prediction  = model.generate(protein_prompt, structure_prediction_config)
            del structure_prediction
    # -----

    

    seq_info_dict = {}
    # print("name2seq: ", len(name2seq))
    # annotate_antibody_seqs(name2seq, name2annot, seq_info_dict)#; print("name2seq: ", len(name2seq))
    # annotate_fixed_domain_seqs(name2seq, seq_info_dict)#; print("name2seq: ", len(name2seq))
    annotate_multimer_seqs(name2seq, seq_info_dict)#; print("name2seq: ", len(name2seq))

    seq2names = {}
    for name, seq in name2seq.items():
        seq2names[seq] = seq2names.get(seq, []) + [ name ]
    config_dict = build_config_dict(seq2names, seq_info_dict)
    
    fasta_path = pathlib.Path(input_fasta)
    # config_dict['cryonet'] = str(fasta_path.parent / fasta_path.stem ) + "_pd_tr_ca.pdb" if cryonet_bb_pdb is None else cryonet_bb_pdb
    config_dict['map_file'] = str(fasta_path.parent / fasta_path.stem ) + ".mrc" if input_mrc is None else input_mrc

    all_templ_pdb_files = [] # Record all produced pdb files
    all_templ_pae_files = [] # Record all produced pAE files

    ##########################################
    ### Get PDB files and pAE files
    ##########################################
    # print(_info(f"Total {len(config_dict['group2templ_info'])} groups"), flush=True)

    # cifdb_file_wo_suffix = os.path.splitext(cifdb_file)[0]
    cifdb_seq_file = f"{cus_templ_path}.fasta"
    if not os.path.exists(cifdb_seq_file):
        import pdb;pdb.set_trace()
    
    cifdb_name2seq, cifdb_name2annot = load_fasta(cifdb_seq_file, load_annotation=True)

    EXIST_ON_FINISH = False # For do_not_run_af2
    group2handles = {} # AF2 handles
    group2afdir = {} # AF2 directory
    used_ids = []

    group2templ_info = config_dict['group2templ_info']
    chain_num = 0
    for k,v in  group2templ_info.items():
        chain_num += v['rep_num']


    # os.system(f"grep '>' {cifdb_seq_file}")

    for group_idx, group_info_dict in config_dict['group2templ_info'].items():


        config_dict['group2templ'][group_idx] = []
        seq      = group_info_dict['seq']
        rep_num  = group_info_dict['rep_num']
        seq_type = group_info_dict['type']
        do_func  = group_info_dict['do_func']
        seq_len = len(seq)

        # print(_info(f"==========================> Group {group_idx} : {rep_num} <==============="), flush=True)

        num_cus_templ = 0
        num_pdb_templ = 0
        num_afdb_templ = 0
        num_af2_templ = 0

        ####################################
        #### CIF to fasta
        ####################################

        
        ####################################
        #### Search in PDB
        ####################################
        
        phmmer_outfile = os.path.join(work_dir, f"group_{group_idx}_pdb_phmmer.txt")
        # if not os.path.exists(phmmer_outfile) or os.path.getsize(phmmer_outfile) == 0:
        run_phmmer_seqdb(seq, phmmer_outfile, dbtype='cifdb',afdb_path=cifdb_seq_file, exclude_pdbs=[], 
                max_resolution=8.0, max_pdb_date=max_pdb_date_str, n_cpu=ncores)
        templ_files_dir = work_dir +"/"
        out_templ_files_dir = work_dir + f"/{group_idx}/"
        if not os.path.exists(out_templ_files_dir):
            os.mkdir(out_templ_files_dir)
        # debug 
        # breakpoint()


        # os.system(f"rm -rf {out_templ_files_dir};mkdir -p {out_templ_files_dir}")
        
        
        
        templ_prots, templ_files, templ_ids = build_templates_from_phmmer_cifdb(phmmer_outfile, seq, templ_files_dir,
            out_templ_files_dir, min_coverage=min_pdb_templ_cov, min_identity=min_identity, dbtype='cifdb', 
            top_n=max_pdb_templ, auto_fix=False, verbose=0)
        if templ_prots is not None:
            accept_ids = []
            print(f"gid: {group_idx} {len(templ_ids)}")
            for prot, prot_file, id_ in zip(templ_prots, templ_files, templ_ids):

                if len(config_dict['group2templ'][group_idx])< rep_num and id_ not in used_ids:
                    pdb_id = id_.split('_')[0]
                    num_res_resolved = np.sum(prot.atom_mask.sum(1) > 0)
                    config_dict['group2templ'][group_idx].append({
                                    'path': prot_file,
                                    'pAE': None,
                                    'num_res_resolved': num_res_resolved, 
                                    'domain_list': None,
                                    'templ_type': 'PDB',
                                    'templ_name': id_,
                                    'templ_resol': -1,
                                    'seq': seq,
                                    'type': seq_type,
                                    'rep_num': rep_num,
                                    'do_func': do_func
                                })
                    used_ids.append( id_)
                    accept_ids.append( id_)
                    # print("accept: ", id_)
                # else:
                #     print(f"no templ for {group_idx} in {id_}")

                num_pdb_templ += 1
                all_templ_pdb_files.append(prot_file)
            # print("accept_ids:", accept_ids)
            # print("used_ids:", used_ids)
        # else:
        #     print(_warn(f"No PDB templates found for group {group_idx}"), flush=True)
        
        
        #if group_idx == 0:
        #    print( config_dict['group2templ'] )
        # import pdb;pdb.set_trace()
        # print(_info(f"PDB templates: {num_pdb_templ}"), flush=True)
        ####################################
        #### Search in AFDB
        ####################################
        # if try_collect_internet() is False:
        #     afdb_mode = 'none'
        # print("afdb_mode: ", afdb_mode)
    if chain_num > len(used_ids):
        gt_ids = cifdb_name2seq.keys()
        # print(f"#seq: {chain_num}, #used: {len(used_ids)}, #templ: {len(gt_ids)}")

        if len(gt_ids) == len(used_ids):
            print(f"all the chains in the templates are comfirmed, but not all the chains in the input fasta are comfirmed.")
        else:
            print(f"not all the chains in the templates are comfirmed.")
            breakpoint()
            sys.exit()
    # sort templates by num_res_resolved
    for grpidx in config_dict['group2templ']:
        try:
            sorted_list = sorted(config_dict['group2templ'][grpidx], key=lambda x: (x['num_res_resolved'], -x['templ_resol']), reverse=True)
        except:
            import pdb; pdb.set_trace()
        config_dict['group2templ'][grpidx] = sorted_list
    # import pdb; pdb.set_trace()
    # json.dump(config_dict, open(output_json.replace(".json","_tmp.json"), 'w'), indent=4, cls=NpEncoder)

    # infer protein
    # with Pool( min(ncores, os.cpu_count()) ) as pool:
    #     h_list = []
    for gid, group2templ in config_dict['group2templ'].items():
        if len(group2templ)>0 and 'path' in group2templ[0]:
            in_path = group2templ[0]['path']
            out_dir = os.path.dirname(in_path)
            out_path0 = f'{out_dir}/msa_idx_0.pdb'
            out_path1 = f'{out_dir}/msa_idx_1.pdb'
            build_backbone_imp(in_path, out_path0, out_path1)

    json.dump(config_dict, open(input_mrc.replace(".mrc","_tmp.json"), 'w'), indent=4, cls=NpEncoder)
   
    cp_all_chains_best(config_dict, input_mrc, resolution=resolution, den_thr=den_thr, cc_thr=0.2, 
                       overlap_thr=0.5, keep_rep_num=False, do_refine=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build the template from PDB and AFDB')
    parser.add_argument('-i', '--input_fasta', type=str,            required=True, help="Protein sequence fasta file.")
    parser.add_argument('-w', '--work_dir',    type=str,            required=True, help='Working directory.')
    parser.add_argument('--exclude_pdbs',      type=str,            required=False, help='PDB list to excluded, multiple PDB seperated by comma')
    parser.add_argument('--ncores',            type=int,            default=8,      help='Max number of cores to processing templates')
    parser.add_argument('--min_pdb_templ_cov', type=float,          default=0.01,   help='Minimun coverage of PDB template of sequence')
    parser.add_argument('--min_identity',      type=float,          default=0.2,   help='Minimun coverage of PDB template of sequence')
    
    parser.add_argument('--max_pdb_date',      type=str,            required=False, help='Maximun PDB template date (e.g. 2021-12-05)')
    parser.add_argument('--max_pdb_templ',     type=int,            default=1000,     help='Maximun PDB templates (default: 100)')
    parser.add_argument('--pdb_mode',          type=str,            choices=['none', 'phmmer', 'exactly'], default='phmmer',   help="none: don't use PDB,  phmmer: search PDB  with mismatch, exactly: search PDB  without mismatch")
    parser.add_argument('--map_file',          type=str,            default=None,   help="MRC file")
    parser.add_argument('--cus_templ_path',        type=str,            default=None,   help="CIF file")

    args = parser.parse_args()

    input_fasta     = args.input_fasta
    work_dir        = args.work_dir
    ncores          = args.ncores
    min_pdb_templ_cov = args.min_pdb_templ_cov
    min_identity    = args.min_identity
    max_pdb_templ   = args.max_pdb_templ
    map_file        = args.map_file
    cus_templ_path      = args.cus_templ_path
   

    # run_inference_de(input_fasta, work_dir, map_file=map_file, ncores=ncores, min_pdb_templ_cov=min_pdb_templ_cov, 
    #                  min_identity=min_identity, max_pdb_templ=max_pdb_templ, cus_templ_path=cus_templ_path)
    work_dir = "/data/jobs/f5/jobs_241203_templ/0448/.templ/0/msa_idx_0_aem"
    global_accept_path_list = glob.glob(f"{work_dir}*.pdb")
    print(f"{len(global_accept_path_list)}")
    final_path = input_fasta.replace(".fasta", "_cf.pdb")
    pcompl = protein_chain_file_list_to_protein_complex(global_accept_path_list)
    pcompl.to_pdb(final_path)
    
    # if int(chain_num) != len(global_accept_path_list):
    #     # chain_new = ProteinChain.my_concat(pc_list)
    #     # chain_new.to_pdb(final_path)
    #     final_path2 = mappath.replace(".mrc", f"_cf_{len(global_accept_path_list)}-{chain_num}.pdb")
    #     pcompl.to_pdb(final_path2)
   