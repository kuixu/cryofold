#!/usr/bin/env python

import shutil, os, sys, time, random, pickle, io, gzip, re, json, platform, tempfile, importlib, pathlib, argparse, multiprocessing, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import multiprocessing as mp
from typing import List, Union, Optional
from Bio.SVDSuperimposer import SVDSuperimposer
import torch
torch.set_num_threads(1)

from cryofold.libs.structure.hy36 import decode_atom_id, decode_chain_id
from cryofold.libs.structure.docking import do_docking, do_docking_best, do_cm_calculation


status, output = subprocess.getstatusoutput('adpem')
assert status == 1, f"Expect adpem in PATH, but got {output}"

# status, output = subprocess.getstatusoutput('ChimeraX --help')
# assert status == 64, f"Expect ChimeraX in PATH, but got {output}"

# status, output = subprocess.getstatusoutput('phenix -v')
# assert status == 1, f"Expect phenix.map_model_cc in PATH, but got {output}"

 
### Set pandas dataframe full-screen display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', None)

### CryoNet code
# from cryonet.libs.structure          import StructureBuilder
from cryonet.util.chain_comparison   import chain_comparison
from cryonet.libs.structure.density  import DensityInfo, mol_atom_density_np, mol_atom_density_th

from alphafold.common         import protein           as AF2_Prot_Module
from alphafold.common.protein import Protein           as AF2_Protein
from alphafold.common         import residue_constants as AF2_Res_Constants
# AF2_Prot_Module.extend_PDB_Chains(extend_to=500) # More chains

restypes_with_x = AF2_Res_Constants.restypes_with_x
restype_3to1    = AF2_Res_Constants.restype_3to1
restype_1to3    = AF2_Res_Constants.restype_1to3
# PDB_CHAIN_IDS   = AF2_Prot_Module.PDB_CHAIN_IDS

import af2_features

read_pdb_atom_line        = af2_features.read_pdb_atom_line        # Parse a pdb file ATOM line
read_pdb_seq              = af2_features.read_pdb_seq              # Read and parse SEQRES lines from pdb file
write_pdb_seq             = af2_features.write_pdb_seq             # Write sequence to SEQRES lines
fill_protein_to_nongapped = af2_features.fill_protein_to_nongapped # Fill gapped AF2_Protein to unggaped

#### IPyRSSA code 
try:
    import Colors, General
except ModuleNotFoundError:
    from IPyRSSA import Colors, General

_info  = lambda x: Colors.f(x, 'cyan')
_warn  = lambda x: Colors.f(x, 'yellow')
_error = lambda x: Colors.f(x, 'red')

### Global Variables (initialized constants by main)

NCPU = 64
DOCK_METHOD = 'adpem' # adpem or phenix
CC_METHOD = 'cryonet' # cryonet or phenix
CC_TYPE = 'pearson' # pearson, cosine, spearman
CC_PEAK_THR = 1.1
###############################
###
###
###  >>> Defined Functions <<<
###
###
###############################

import hashlib

def file_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

# file_path = 'example.txt'
# md5_value = calculate_md5(file_path)
# print(f"The MD5 value of the file is: {md5_value}")
#################################
### For json loading and parsing
#################################

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

def format_config_dict(config_dict):
    """
    Convert config dict key to normal data type (int, float, str, list)
    """
    if isinstance(config_dict, dict):
        new_config_dict = {}
        for key in config_dict:
            if isinstance(key, np.integer):
                new_config_dict[int(key)] = format_config_dict(config_dict[key])
            elif isinstance(key, np.floating):
                new_config_dict[float(key)] = format_config_dict(config_dict[key])
            elif isinstance(key, np.ndarray):
                new_config_dict[key.tolist()] = format_config_dict(config_dict[key])
            else:
                new_config_dict[key] = format_config_dict(config_dict[key])
        return new_config_dict
    else:
        return config_dict

def dump_config_dict(config_dict, config_json_file):
    """
    Save config dictionary to json file safely
    """
    config_dict = format_config_dict(config_dict)
    json.dump(config_dict, open(config_json_file, 'w'), indent=4, cls=NpEncoder)

#################################
###  Density Map related functions
#################################

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

def den_model_cc(den_exp, model_path, resolution=3.5, cc_type='pearson', cc_method='cryonet', return_cc_peak=False):
    """
    Calculate the CC value between the true map and simulated map

    Parameters
    -------------
    den_exp: str, DensityInfo
        Path to the true density map
    model_path: str
        Path to the simulated model
    resolution: float
        Resolution of the map
    cc_type: str
        pearson, cosine, spearman
    map_file: str
        Path to the map file

    Return
    -------------
    cc: float
    """
    if isinstance(den_exp,str):
        den_exp = DensityInfo(den_exp)

    if cc_method == 'phenix':
        cc_dict = den_model_cc_phenix(den_exp.path, model_path, resolution=resolution)
        # print(cc_dict)
        cc = cc_dict['CC_mask']
        if return_cc_peak:
            return cc_dict['CC_mask'], cc_dict['CC_peaks']
        else:
            return cc_dict['CC_mask']
    else:

        assert cc_type in ('pearson', 'cosine', 'spearman')
        prot = af2_features.read_pdb(model_path, molecular_type='protein')
        mol_den = afprot2density(prot, voxel_size=den_exp.apix, atom_weight=14.0, res=resolution)
        cc = den_cc(den_exp, mol_den, cc_type=CC_TYPE)
        if return_cc_peak:
            cc_peak = den_cc(den_exp, mol_den, cc_type=CC_TYPE, overlap_type='left')
            return cc, cc_peak
        else:
            return cc
    
    return cc

def den_model_cc_phenix(den_path, model_path, resolution=3.5):
    # os.system(f"phenix.real_space_correlation {den_path} {model_path} resolution={resolution} > /dev/null")
    tmp_log = tempfile.NamedTemporaryFile(delete=True, suffix="")

    cmd = f"phenix.map_model_cc {den_path} {model_path} resolution={resolution} file_name_prefix={tmp_log.name} ignore_symmetry_conflicts=True scattering_table=it1992 2>&1 "
    status, output = subprocess.getstatusoutput(cmd)
    # print(output)
    cc_dict = parse_phenix_cc_file(output)
    os.system(f"rm {tmp_log.name}")
    return cc_dict

def parse_cc_file(ccfile, per_chain_cc=False):

    with open(ccfile, 'r') as f:
        output = f.readlines()
        return parse_phenix_cc_file(output, per_chain_cc=per_chain_cc)
        # 
def parse_phenix_cc_file(cc_output, per_chain_cc=False):
    CC_mask, CC_volume, CC_peaks, CC_box = -1, -1,-1, -1
    CC_mc, CC_sc = -1, -1
    chain_cc={}
    chain_cc_region = False

    output = cc_output.split('\n')
    for idx in range(len(output)):
        if output[idx].startswith('Map-model CC (overall)'):
            CC_mask, CC_volume, CC_peaks, CC_box = [ float(line.split(':')[1].strip()) for line in output[idx+2:idx+6] ]
        elif output[idx].startswith('Main chain:'):
            CC_mc = float(output[idx+2].strip().split()[0])
            chain_cc_region=False
        elif per_chain_cc and chain_cc_region:
            fields = output[idx].strip().split()
            chain_cc[fields[0]] = fields[1]
        elif per_chain_cc and output[idx].startswith('chain ID  CC'):
            chain_cc_region = True
        elif output[idx].startswith('Side chain:'):
            try:
                CC_sc = float(output[idx+2].strip().split()[0])
            except IndexError:
                CC_sc = 0.0
    
    return {
        'CC_mask':     CC_mask, 
        'CC_volume':   CC_volume, 
        'CC_peaks':    CC_peaks, 
        'CC_box':      CC_box,
        'CC_mc':       CC_mc,
        'CC_sc':       CC_sc,
        'CC_per_chain':chain_cc
    }

def den_cc(den_exp: DensityInfo, den_sim: DensityInfo, cc_type: str = 'pearson', overlap_type:str = "right") -> float:
    """
    Calculate the CC value between the true map and simulated map

    Parameters
    -------------
    den_exp: True density map
    den_sim: Simulated density map
    cc_type: pearson, cosine, spearman

    Return
    -------------
    cc: float
    """
    assert cc_type in ('pearson', 'cosine', 'spearman')

    if overlap_type == 'right':
        t_ov, m_ov = den_exp.overlap_right(den_sim)
    else:
        t_ov, m_ov = den_exp.overlap_left(den_sim)

    if cc_type == 'pearson':
        X = np.vstack([t_ov, m_ov])
        cc = np.corrcoef(X)[0][1].item()
    elif cc_type == 'cosine':
        cc = torch.nn.functional.cosine_similarity(t_ov.reshape(1,-1), m_ov.reshape(1,-1)).item()
    else:
        import scipy, scipy.stats
        cc = scipy.stats.spearmanr(t_ov[0].numpy(), m_ov[0].numpy())[0]
    if np.isnan(cc):
        cc = -1.0
        # import pdb;pdb.set_trace()
    return cc

def afprot2density(prot:AF2_Protein, voxel_size:float=1.0, 
    atom_weight:float=14.0, res:float=3.0, datatype='numpy')->DensityInfo:
    """
    Convert AF2_Protein object to Density object
    """
    assert datatype in ('numpy', 'torch')
    
    atom_coords = prot.atom_positions[ prot.atom_mask.astype(np.bool_) ]
    if datatype == 'numpy':
        density, offset = mol_atom_density_np(atom_coords, atom_weight=atom_weight, res=res, voxel_size=voxel_size)
    else:
        density, offset = mol_atom_density_th(atom_coords, atom_weight=atom_weight, res=res, voxel_size=voxel_size)
    den_sim = DensityInfo(density=density, offset=offset, apix=voxel_size, datatype='torch')
    return den_sim

def get_cc_for_domain_list(den_exp:DensityInfo, prot:AF2_Protein, domain_list:list, 
                           resolution:float=3.0, return_dom_prots:bool=False, 
                           verbose:bool=False):
    """
    Calculate CC values for each domain
    
    Parameters
    ---------------
    den_exp: True density map
    prot: Full chain PDB model (should after domain adaptation stage...)
    domain_list: List of domain boundaries
    return_dom_prots: Return domain protein list
    verbose: bool
    
    Return
    ---------------
    if return_dom_prots is False:
        cc_list: CC values for each domain
    else:
        cc_list, do_prot_list
    """
    cc_list = []
    ccpeak_list = []
    do_prot_list = []
    for domain in domain_list:
        mask = np.zeros(prot.aatype.shape[0])
        for s, e in domain:
            mask[s:e+1] = 1.0
        do_prot = prot.filter_by_mask(mask.astype(np.bool_))
        do_prot_list.append(do_prot)
        tmp_pdb = tempfile.NamedTemporaryFile(delete=True, suffix=".pdb")
        af2_features.save_prot_as_pdb(do_prot, tmp_pdb.name)
        # cc = den_model_cc(den_exp, tmp_pdb.name, resolution=resolution, cc_type=CC_TYPE, cc_method=CC_METHOD)
        cc, cc_peak = den_model_cc(den_exp, tmp_pdb.name, resolution=resolution, cc_type=CC_TYPE, cc_method=CC_METHOD, return_cc_peak=True)
        # mol_den = afprot2density(do_prot, voxel_size=den_exp.apix, atom_weight=14.0, res=resolution)
        # cc = den_cc(den_exp, mol_den, cc_type=CC_TYPE)
        os.system(f"rm {tmp_pdb.name}")
        cc_list.append(cc)
        ccpeak_list.append(cc_peak)
        if verbose:
            do_str = ",".join([ f"{s}-{e}" for s,e in domain ])
            print(f"Domain CC: {do_str} -- {cc}")
    
    if return_dom_prots:
        return cc_list, ccpeak_list, do_prot_list
    else:
        return cc_list

#################################
### Auxiliary functions
#################################

def find_seq(bb_frags: List[AF2_Protein], templ_prot: AF2_Protein, max_match_distance: float = np.inf) -> list:
    """
    Parameters
    ---------------
    bb_frags: Fragments (non-gapped) from pd_tr_ca backbone
    templ_prot: AF2_Protein object (non-gapped)
    max_match_distance: Acceptable maximum distance (RMSD) between backbone and template.

    Return
    ---------------
    matches: [ [bb_idx, templ_idx_start, templ_idx_end] ]
    """
    matches = []
    templ_seq = af2_features.aatype2seq(templ_prot.aatype)
    for bb_idx, bb_prot in enumerate(bb_frags):
        bb_seq = af2_features.aatype2seq(bb_prot.aatype)
        s_idx = templ_seq.find(bb_seq)
        if s_idx != -1:
            e_idx = s_idx + len(bb_seq)
            bb_mask = bb_prot.atom_mask[:, 1].astype(np.bool_)
            templ_mask = templ_prot.atom_mask[s_idx:e_idx][:, 1].astype(np.bool_)
            mask = (bb_mask & templ_mask)
            bb_positions = bb_prot.atom_positions[mask, 1]
            templ_positions = templ_prot.atom_positions[s_idx: e_idx][mask, 1]
            dist_pos = np.linalg.norm(templ_positions - bb_positions, axis=1)
            if (mask.sum() > 5) and (dist_pos.mean() <= max_match_distance):
                matches.append((bb_idx, s_idx, e_idx))
    return matches

def is_two_pdb_overlap(pdbfile1, pdbfile2, found_value=0.5, mode='both') -> bool:
    """
    Measure overlap between two PDB files.

    Parameters
    --------------
    pdbfile1: str
    pdbfile2: str
    found_value: Cutoff Found value to define overlap
    mode: left, right, both.

    Return
    --------------
    overlaped: bool
    """
    assert mode in ('left', 'right', 'both')
    met_inter1 = chain_comparison(pdbfile1, pdbfile2, return_results=True)
    met_inter2 = chain_comparison(pdbfile2, pdbfile1, return_results=True)

    if mode == 'left':
        overlaped = (met_inter1['found'] >= found_value)
    elif mode == 'right':
        overlaped = (met_inter2['found'] >= found_value)
    else:
        overlaped = (met_inter1['found'] >= found_value) or (met_inter2['found'] >= found_value)
    # print("found: ", met_inter1['found'])
    return overlaped

def is_overlap(dok_file, domain_paths, found_value=0.5, disable_mp=False):
    """
    Compare dok_file with each PDB file in domain_paths.
    """
    if domain_paths is None or len(domain_paths)==0:
        return False, []

    result_list = []
    overlapped_paths = []
    
    if disable_mp:
        for dpath in domain_paths:
            overlaped = is_two_pdb_overlap(dok_file, dpath, found_value)
            result_list.append(overlaped)
            if overlaped:
                overlapped_paths.append(dpath)
    else:
        # print(f"{NCPU} process to check overlap.")
        with mp.Pool( min(NCPU, os.cpu_count()) ) as pool:
            h_list = []
            for dpath in domain_paths:
                h = pool.apply_async(is_two_pdb_overlap, (dok_file, dpath, found_value))
                h_list.append([h, dpath])
            for h, dpath in h_list:
                overlaped = h.get()
                result_list.append(overlaped)
                if overlaped:
                    overlapped_paths.append(dpath)

    return any(result_list), overlapped_paths

def nms_pdb_v0(df, accept_path_list, found_value=0.5, cc_thr=0.2, disable_mp=False):
    domain_paths = df['path'].tolist()
    if len(accept_path_list)==0:
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
        if len(accept_list) == 0:
            accept_pdbs = accept_path_list
        else:
            accept_pdbs = df.loc[accept_list, 'path'].tolist() + accept_path_list

        overlapped, overlapped_paths = is_overlap(dok_file, accept_pdbs, found_value, disable_mp)
        if overlapped:
            reject_list.append(i)
        else:
            accept_list.append(i)
    # import pdb; pdb.set_trace()
    return accept_list, reject_list


def get_rep_num(accept_pdbs, ):
    return {}

def nms_pdb(df, accept_path_list, accept_gid_list, found_value=0.5, cc_thr=0.2, disable_mp=False, keep_rep_num=False, group2templ_info=None):
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
        overlapped, overlapped_paths = is_overlap(dok_file, accept_pdbs, found_value, disable_mp)
        if overlapped:
            reject_list.append(i)
        else:
            accept_list.append(i)
    # import pdb; pdb.set_trace()
    return accept_list, reject_list

def read_prd_pdb(prd_pdb: str, min_frag_size=5, full_padding=True):
    """ Pan 20220630
    Read pd_tr_ca backbone PDB file

    Return
    -----------------
    seq_dict: dict
    frags: List[AF2_Protein]
    """
    seq_dict = read_pdb_seq(prd_pdb)
    assert len(seq_dict) > 0, f"Expect more than 1 sequence in {prd_pdb}, but got 0"
    frags = af2_features.read_gapped_pdb(prd_pdb, min_prot_size=min_frag_size, full_padding=full_padding)
    return seq_dict, frags

def is_overlap_with_pre(path, b_path, found_value):
    met_inter2 = chain_comparison(path, b_path, return_results=True)
    is_overlap = met_inter2['found'] >= found_value
    return is_overlap

def select_best_temp(metpath, ploy_num, found_value=0.01, solved_chains=None):
    """
    Select best template from metrics

    Return
    ---------------
    best: dict
        -- cc: List of CC
        -- tid: List of template ID
        -- path: List of path
        -- seq_match: List of seq_match
    """
    print(_info(f"select_best_temp: {metpath}, ploy_num={ploy_num}"))
    df = pd.read_csv(metpath)
    # print(df)


    docked_pdbs = []
    if solved_chains is not None:
        # import pdb; pdb.set_trace()
        # for i in range(len(solved_chains)):
        for i in solved_chains.index:
            chain = solved_chains['path'][i]
            cc    = solved_chains['cc'][i]
            docked_pdbs.append((chain, 1.0))

    best = {}
    best['cc'] = []
    best['cc_peak'] = []
    best['gid'] = []
    best['tid'] = []
    best['path'] = []
    best['seq_match'] = []
    best['stype'] = []

    i_solve = 0
    for k in df.index:
        cc = df['cc'][k]
        cc_peak = df['cc_peak'][k]
        if solved_chains is not None:
            gid = df['gid'][k]
        else:
            gid = 0
        tid = df['tid'][k]
        seq_match = df['seq_match'][k]
        stype = df['stype'][k]
        path = df['path'][k]
        is_overlap = False
        # parallel
        with mp.Pool( min(NCPU, os.cpu_count()) ) as pool:
            h_list = []
            for idx in range(len(docked_pdbs)):
                b_path, b_cc = docked_pdbs[idx]
                h = pool.apply_async(is_overlap_with_pre, (path, b_path, found_value))
                h_list.append(h)
            
            for h in h_list:
                is_overlap_tmp = h.get()
                is_overlap = (is_overlap or is_overlap_tmp)
                if is_overlap_tmp:
                    if solved_chains is None:
                        docked_pdbs[idx] = (path, cc)
                    # print(_warn(f"Overlap found: {b_path} <-> {path} found={met_inter['found']:.3f}")) 
        # for idx in range(len(docked_pdbs)):
        #     b_path, b_cc = docked_pdbs[idx]
        #     # met_inter1 = chain_comparison(b_path, path,   return_results=True)
        #     met_inter2 = chain_comparison(path,   b_path, return_results=True)
        #     is_overlap = (is_overlap or met_inter2['found'] >= found_value) # met_inter1['found'] >= found_value or 
        #     if met_inter2['found'] >= found_value: # met_inter1['found'] >= found_value or 
        #         if solved_chains is None:
        #             docked_pdbs[idx] = (path, cc)
        #         print(_warn(f"Overlap found: {b_path} <-> {path} found={met_inter2['found']:.3f}"))
        if is_overlap:
            continue
        else:
            best['cc'].append(cc)
            best['cc_peak'].append(cc_peak)
            best['gid'].append(gid)
            best['tid'].append(tid)
            best['path'].append(path)
            best['seq_match'].append(seq_match)
            best['stype'].append(stype)
            if solved_chains is None:
                docked_pdbs.append((path, cc))
            i_solve += 1
        if i_solve == ploy_num:
            break
    if len(best['cc'])<ploy_num:
        none_num = ploy_num - len(best['cc'])
        best['cc'].extend([0.0] * none_num)
        best['cc_peak'].extend([0.0] * none_num)
        best['gid'].extend([0] * none_num)
        best['tid'].extend([0] * none_num)
        best['path'].extend([None] * none_num)
        best['seq_match'].extend([0.0] * none_num)
        best['stype'].extend([1] * none_num)
    # if best['cc'][0] ==0.0:
    #     import pdb; pdb.set_trace()
    return best

def recheck_selected_temps(config, found_value=0.01):
    """
    Check overlap for all templates
    """

    print(_info(f"recheck_selected_temps"))
    # df_sup, has_unsolved_chain = sortcc(config, key='cryonet_sup', sort=True, keepnone=True)
    df_sup, has_unsolved_chain = sortcc_group(config, key='cryonet_sup', sort=True, keepnone=True)
    # print(df_sup.to_dict('records'))
    # import pdb;pdb.set_trace()

    docked_pdbs = []
    new_sup = {}
    poly_num_dict = {k: len(v['cc']) for k, v in config['cryonet_sup'].items()}
    for k in range(0, len(df_sup)):
        gid = df_sup['gid'][k]
        if gid not in new_sup:
            new_sup[gid] = {'cc': [], 'cc_peak': [], 'tid': [], 'path': [], 'seq_match': [], 'stype': []}
        cc = df_sup['cc'][k]
        cc_peak = df_sup['cc_peak'][k]
        tid = int(df_sup['tid'][k])
        seq_match = df_sup['seq_match'][k]
        stype = df_sup['stype'][k]
        path = df_sup['path'][k]
        if path is None:
            continue
        is_overlap = False
        for b_path in docked_pdbs:
            met_inter = chain_comparison(path, b_path, return_results=True)
            is_overlap = is_overlap or met_inter['found']>=found_value
            if met_inter['found'] >= found_value:
                print(_warn(f"Overlap found: {b_path} <-> {path} found={met_inter['found']:.3f}"))
        if is_overlap:
            continue
        else:
            new_sup[gid]['cc'].append(cc)
            new_sup[gid]['cc_peak'].append(cc_peak)
            new_sup[gid]['tid'].append(tid)
            new_sup[gid]['path'].append(path)
            new_sup[gid]['seq_match'].append(seq_match)
            new_sup[gid]['stype'].append(stype)
            docked_pdbs.append(path)
    for gid in new_sup:
        if len(new_sup[gid]['cc']) < poly_num_dict[gid]:
            none_num = poly_num_dict[gid] - len(new_sup[gid]['cc'])
            new_sup[gid]['cc'].extend([0.0] * none_num)
            new_sup[gid]['cc_peak'].extend([0.0] * none_num)
            new_sup[gid]['tid'].extend([0] * none_num)
            new_sup[gid]['path'].extend([None] * none_num)
            new_sup[gid]['seq_match'].extend([0.0] * none_num)
            new_sup[gid]['stype'].extend([1] * none_num)

    config['cryonet_sup'] = new_sup
    return config

def scale_map(den_path, out_path, scale=1):
    den = DensityInfo(den_path, datatype='numpy')
    den.scale(scale)
    den.save(out_path)

def clip_map_with_pdb(pdbpath: str, mappath: str, outpath: str, simulate_res: float = 20.0,
                        thr: Union[str, float, int] = 0.0, clip: bool = True):
    """
    Use pdb model to simulate a map, and use the simulated map to mask the raw map.

    Parameters
    -----------------
    pdbpath: Simulated pdb file
    mappath: Map file to mask
    outpath: Output masked map file
    simulate_res: Simulated resolution
    thr: float or str. Specify threshold to mask, or use 'mean' or 'median'
    clip: bool, automatic clip masked map file

    Return
    -----------------
    masked_den: Masked density object
    """
    assert isinstance(thr, (str, int, float)), f"Expect thr is instance of str, float or int, but got {type(thr)}"

    den_exp = DensityInfo(mappath)
    prot = af2_features.read_pdb(pdbpath)
    den_sim = afprot2density(prot, voxel_size=den_exp.apix, atom_weight=14.0, res=simulate_res)
    mol_den = den_sim.density
    if isinstance(thr, (float, int)):
        thr = thr
    elif isinstance(thr, str):
        assert thr in ('median', 'mean')
        if thr == 'median':
            thr = mol_den[mol_den>0].median()
        elif thr == 'mean':
            thr = mol_den[mol_den>0].mean()
    mol_den[mol_den<=thr]  = 0.0
    mol_den[mol_den>thr] = 1.0
    mask_value = mol_den.sum().item()
    if mask_value < 100:
        print(f"Mask size too small: {mask_value:.2f}")
    denobj_mol_mask = DensityInfo(density=mol_den, offset=den_sim.offset, apix=den_sim.apix)
    masked_density, masked_offset = den_exp.mask(denobj_mol_mask, keep_origin=False)
    masked_den = DensityInfo(density=masked_density, offset=masked_offset, apix=den_exp.apix)
    if clip:
        masked_den.clip()
    masked_den.save(outpath)
    return masked_den

def clip_map_with_box(pdbpath, mappath, outpath, radius=10):
    """
    Clip map with PDB model, only preserve map regions with model
    """
    print(_info(f"clipping density with pdb box: {pdbpath}"))

    den_exp = DensityInfo(mappath)
    print(_info(f"original cube: {den_exp.density.shape}"))

    den_exp.scale(1)

    prot = af2_features.read_pdb(pdbpath)
    ca_xyz = torch.from_numpy( prot.atom_positions[prot.atom_mask[:, 1].astype(np.bool_), 1] ).float()
    min_val, _ = ca_xyz.min(0)
    max_val, _ = ca_xyz.max(0)

    b_min = min_val - radius
    b_max = max_val + radius

    masked_density, masked_offset = den_exp.mask_box(b_min, b_max, keep_origin=False)
    if masked_density.numel() == 0:
        print(_warn("warning!!! zero clipping. "))
        shutil.copy2(mappath, outpath)
        return den_exp
    else:
        masked_den = DensityInfo(density=masked_density, offset=masked_offset, apix=den_exp.apix)
        masked_den.clip()
        masked_den.save(outpath)
        print(_info(f"clipped  cube: {masked_den.density.shape}"))
        print(_info(f"masked box map: {outpath}"))
        return masked_den

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
    prot = af2_features.read_pdb(pdbpath)
    denobj_mol = afprot2density(prot, voxel_size=den_exp.apix, atom_weight=14.0, res=resolution)
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

def phenix_dock_in_map(pdbfile, mapfile, resolution, nproc, outpdbfile, quick=False, min_cc=0.2):
    """
    Use phenix.dock_in_map to dock a single PDB chain to target_map
    
    Parameters
    -----------
    pdbfile: str
    mapfile: str
    resolution: float
    nproc: int
    outpdbfile: str
    quick: bool
    
    Return
    -----------
    status: int
        0 -- Failed
        1 -- Success
    """
    from multiprocessing import Process
    import tempfile, subprocess, os

    assert os.path.exists(pdbfile), f'{pdbfile} does not exists'
    assert os.path.exists(mapfile), f'{mapfile} does not exists'
    assert pdbfile.endswith('.pdb')

    pdbfile = os.path.abspath(pdbfile)
    mapfile = os.path.abspath(mapfile)
    outpdbfile = os.path.abspath(outpdbfile)

    if os.path.exists(outpdbfile):
        os.remove(outpdbfile)

    with tempfile.TemporaryDirectory(prefix='dockinmap_') as work_dir:
        if quick:
            refine_cycle = 1
        else:
            refine_cycle = 3
        cmd  = f"cd {work_dir}; phenix.dock_in_map {mapfile} search_model={pdbfile} resolution={resolution} "
        cmd += f"refine_cycle={refine_cycle} quick={quick} use_symmetry=False nproc={nproc} pdb_out={outpdbfile} min_cc={min_cc}"
        status, output = subprocess.getstatusoutput(cmd)

    if os.path.exists(outpdbfile):
        return 1
    else:
        print(_error(f"Phenix dock_in_map failed: {cmd}"))
        print(f"Phenix output: {output}")
        return 0

def dock_pdb_into_map(pdbpath, mappath, outpath, resolution=3.0):
    """
    Use adpem to dock PDB atom model to 3D density map
    """
    den_exp = DensityInfo(mappath)

    # tmp_map = tempfile.NamedTemporaryFile(delete=True, suffix=".mrc")
    tmp_map = f"/tmp/{file_md5(mappath)}.mrc"
    tmp_pdb = f"/tmp/{file_md5(pdbpath)}.pdb"
    # tmp_pdb = tempfile.NamedTemporaryFile(delete=True, suffix=".pdb")
    tmp_out = tempfile.NamedTemporaryFile(delete=True)
    
    if not pathlib.Path(tmp_map).exists():
        # shutil.copy2(mappath, tmp_map.name)
        shutil.copy2(mappath, tmp_map)
    
    if not pathlib.Path(tmp_pdb).exists():
        # shutil.copy2(pdbpath, tmp_pdb.name)
        shutil.copy2(pdbpath, tmp_pdb)
    
    den_mean = max(den_exp.mean, 0)
    # print(_info(f"adpem {mappath} {pdbpath}"))

    # cmd = 'adpem {} {} 16 {} {} -n 1 -o {} '.format(tmp_map.name, tmp_pdb.name, den_mean, resolution, tmp_out.name)
    cmd = 'adpem {} {} 16 {} {} -n 1 -o {} '.format(tmp_map, tmp_pdb, den_mean, resolution, tmp_out.name)
    # print(_info(cmd))
    output_results_str = subprocess.getoutput(cmd)
    dok_file = tmp_out.name + "0001.pdb"
    if pathlib.Path(dok_file).exists():
        shutil.copy2(dok_file, outpath)
    else:
        print(_error(f"Warning: Docking error -- {cmd}, please check"))
        outpath = pdbpath
    seq_dict = af2_features.read_pdb_seq(pdbpath)
    prot = af2_features.read_pdb(outpath)
    af2_features.save_prot_as_pdb(prot, outpath, seq_dict)
    # os.system(f"rm {tmp_map.name} {tmp_pdb.name} {tmp_out.name}")
    os.system(f"rm {tmp_map} {tmp_pdb} {tmp_out.name}")
    return outpath

def fit_pdb_into_map(pdbpath, mappath, outpath, resolution=3.0):
    """
    Use adpem to dock PDB atom model to 3D density map
    """
    fit_seq = AF2_Prot_Module.seq_from_pdb_seqres(pdbpath)
    tmp_path0= outpath.replace('.pdb', '_tmp0.pdb')
    tmp_path1= outpath.replace('.pdb', '_tmp1.pdb')
    cmd = f'ChimeraX --nogui --cmd "open {pdbpath}; open {mappath};fitmap #1 inMap #2; save {tmp_path0} models #1 relModel #2;exit;" >> /dev/null 2>&1'
    cmd = cmd+f'; bin/real_refine_sup.sh {mappath} {tmp_path0} {tmp_path1} {resolution} >> /dev/null 2>&1;'
    os.system(cmd)

    fit_prot = AF2_Prot_Module.from_pdb_string(open(tmp_path1).read())
    af2_features.save_prot_as_pdb(fit_prot, outpath, full_seq=fit_seq)
    os.system(f"rm {tmp_path0} {tmp_path1}")

    return outpath

def combine_pdbs(pdb_files, out_file):
    pdbpaths = ' '.join(pdb_files)
    cmd = f'ChimeraX --nogui --cmd "open {pdbpaths}; combine ; save {out_file} #{len(pdb_files)+1};exit;" >> /dev/null 2>&1'
    os.system(cmd)


def combine_pdb(pdb_files, pdb_file_seqs, pdb_file_types, out_file):
    """
    Combine all pdb files into single PDB file

    pdb_files: A list of pdb file names
    pdb_file_seqs: A list of pdb sequences
    pdb_file_types: A list of protein types (Monomer, Multimer or Antibody)
    out_file: Output combined file
    """
    ch_idx = 0

    aatype = []
    atom_positions = []
    atom_mask = []
    residue_index = []
    chain_index = []
    b_factors = []

    new_seq_dict = {}
    for idx, pdbfile in enumerate(pdb_files):
        seq = pdb_file_seqs[idx]
        ch_type = pdb_file_types[idx]
        prot = af2_features.read_pdb(pdbfile, full_padding=True)
        # print(len(seq), prot.aatype.shape[0], pdbfile)

        if ch_type in ('Antibody', 'Multimer'):
            ch1_start = 0
            while True:
                ch1_end = seq.find('X'*15, ch1_start)
                ch1_end = ch1_end if ch1_end != -1 else len(seq)
                length = ch1_end - ch1_start
                aatype.append(          prot.aatype[ch1_start:ch1_end] )
                atom_positions.append(  prot.atom_positions[ch1_start:ch1_end] )      
                atom_mask.append(       prot.atom_mask[ch1_start:ch1_end] )
                residue_index.append(   np.arange(length) )
                chain_index.append(     np.ones(length)*ch_idx )
                b_factors.append(       prot.b_factors[ch1_start:ch1_end] )
                new_seq_dict[ch_idx] = seq[ch1_start:ch1_end]
                ch_idx += 1
                # print(0, length, prot.atom_positions[ch1_start:ch1_end].shape[0], pdbfile)

                if ch1_end == len(seq):
                    break
                ch1_start = ch1_end + 15
        else:
            aatype.append( prot.aatype )
            atom_positions.append( prot.atom_positions )
            atom_mask.append( prot.atom_mask )
            residue_index.append( np.arange(len(seq)) )
            chain_index.append( np.ones(len(seq))*ch_idx )
            b_factors.append( prot.b_factors )
            new_seq_dict[ch_idx] = seq
            ch_idx += 1
            # print(1, len(seq), prot.atom_positions.shape[0],pdbfile)
    
    aatype         = np.concatenate(aatype)
    residue_index  = np.concatenate(residue_index)
    atom_positions = np.concatenate(atom_positions)
    atom_mask      = np.concatenate(atom_mask)
    b_factors      = np.concatenate(b_factors)
    chain_index    = np.concatenate(chain_index)
    if len(chain_index)!=atom_positions.shape[0]:
        import pdb; pdb.set_trace()
    af2_features.save_as_pdb(aatype, residue_index, atom_positions, atom_mask, out_file, 
            b_factors=b_factors, asym_id=chain_index, full_seq=new_seq_dict, occupancies=None, 
            ss_coding=[1, 2, 3], write_ss=False, gap_ter_threshold=10)

def sortcc_group(config, key='cryonet_sup', sort=True, keepnone=False):
    """
    Build a dataframe from config[key] dictory

    Return
    ------------
    DataFrame with following attributes:
        -- gid: group index
        -- gra: seq rep index in group
        -- cid: seq index (1-N)
        -- tid: template index
        -- cc:  CC value
        -- cc_peak:  CC value
        -- seq_match: seq match
        -- stype: stype
        -- path: PDB file path
    """
    has_unsolved_chain = False
    cryonet_sup_grp = []
    cid = 0
    # for grpidx, grppaths in config[key].items():
    grpidies = np.sort(list(config[key].keys()))
    for grpidx in grpidies:
        grppaths = config[key][grpidx]
        # rep_num = config['group2templ_info'][grpidx]['rep_num']
        # import pdb; pdb.set_trace()
        rep_num = len(grppaths['cc'])
        # print(grpidx, rep_num)
        for i in range(rep_num):
            cid += 1
            tid = int(grppaths['tid'][i])
            cc = round(grppaths['cc'][i], 2)
            cc_peak = round(grppaths['cc_peak'][i], 2)
            pa = grppaths['path'][i]
            sm = grppaths['seq_match'][i]
            try:
                stype = grppaths['stype'][i]
            except:
                import pdb; pdb.set_trace()
            if pa is None:
                has_unsolved_chain = True
                if keepnone:
                    cc, cc_peak = 0.0, 0.0
                    sm = 0.0
                    pa = None
                    tid, seq_len, stype = 0,0,0
                    cryonet_sup_grp.append(
                        {
                            'gid':grpidx,'gra':i,'cid':cid,'tid':tid,'cc':cc,'cc_peak':cc_peak,'seq_match':sm,'path':pa,'seq_len':seq_len,'rep_num':rep_num,'stype':stype
                        })
            else:
                with open(pa) as f:
                    prot = AF2_Prot_Module.from_pdb_string(f.read())
                    seq_len = len(prot.residue_index)
                    cryonet_sup_grp.append(
                        {
                            'gid':grpidx,'gra':i,'cid':cid,'tid':tid,'cc':cc,'cc_peak':cc_peak,'seq_match':sm,'path':pa,'seq_len':seq_len,'rep_num':rep_num,'stype':stype
                        })

    df = pd.DataFrame(cryonet_sup_grp)
    if sort:
        df = df.sort_values(by=['cc','seq_len'], ascending=False)
        df = df.reset_index(drop=True)
    return df, has_unsolved_chain

def clip_dock_fitinmap(ma, df, k, cc_thr=0.6, resolution=3.0, clip_res=20.0, clip_thr=0.0, no_clip=False):
    gid = df['gid'][k]
    gra = df['gra'][k]
    cid = df['cid'][k]
    tid = df['tid'][k]
    cc  = df['cc'][k]
    cc_peak  = df['cc_peak'][k]
    pa  = df['path'][k]
    sm  = df['seq_match'][k]
    st  = df['stype'][k]
    if cc >= cc_thr:
        return gid, gra, cc, cc_peak, pa, tid

    dok_file = re.sub('.pdb$', "_d.pdb",  pa)
    # print("clip_dock_fitinmap........")
    fit_pdb_into_map(pa, ma, dok_file)
    
    # prot = af2_features.read_pdb(dok_file)
    # mol_den = afprot2density(prot, voxel_size=den_clipped.apix, atom_weight=14.0, res=resolution)
    # cc_dock = den_cc(den_clipped, mol_den, cc_type=CC_TYPE)
    # cc_dock = den_model_cc(den_clipped, dok_file, resolution=resolution, cc_type=CC_TYPE, cc_method=CC_METHOD)
    cc_dock, cc_dock_peak = den_model_cc(ma, dok_file, resolution=resolution, cc_type=CC_TYPE, cc_method=CC_METHOD, return_cc_peak=True)

    
    #### Pan added 20220716
    chain_paths = [ p for p in df['path'] if p is not None ]
    chain_paths.remove(pa)
    overlap, overlapped_paths = is_overlap(dok_file, chain_paths, found_value=0.5, disable_mp=True)
    
    report_cc_chain = f"DOCK1: gid:{gid} cid:{cid} cc:{cc:.3f} cc_dock:{cc_dock:.3f} cc_peak:{cc_dock_peak:.3f} path:{dok_file} overlap:{overlap}"
    print(_info(report_cc_chain))
    if cc_dock > cc and not overlap:
        cc = cc_dock
        cc_peak = cc_dock_peak
        pa = dok_file
    return gid, gra, cc, cc_peak, pa, tid

def dock_temp_raw(mappath, ma_raw, pdbpath, bb_pdb_path, resolution=3.0, gid=0, tid=0, cid=0, idock=0, stage='v0'):

    results = do_docking(pdbpath, mappath, bb_pdb_path=bb_pdb_path, resolution=resolution, stage=stage)
    # print(results)
    if len(results)>0:
        for i in range(len(results)):
            results[i]['gid'] = gid
            results[i]['tid'] = tid
            results[i]['cid'] = cid
            results[i]['stype'] = 1
            del results[i]['found']
            del results[i]['chain_match']
        return results
    else:
        return None
    

def dock_template(mappath, pdbpath, bb_pdb_path, resolution=3.0, gid=0, tid=0, cid=0):

    results = do_docking(pdbpath, mappath, bb_pdb_path=bb_pdb_path, resolution=resolution)
    # print(results)
    if len(results)>0:
        for i in range(len(results)):
            results[i]['gid'] = gid
            results[i]['tid'] = tid
            results[i]['cid'] = cid
            results[i]['stype'] = 1
            del results[i]['found']
            del results[i]['chain_match']
        return results
    else:
        return None


def dock_template_best(mappath, pdbpath, outpath, thr, bb_pdb_path, resolution=3.0, gid=0, tid=0, cid=0, gtid=0, n=1, do_refine=True):

    results = do_docking_best(pdbpath, mappath, outpath, thr, resolution=resolution, n=n, do_refine=do_refine)
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

def dock_all_templates_best(mapfile_whole, group2templ, dock_met_path, accept_gtid_list, thr=None, it=0, do_refine=True):

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

                h = pool.apply_async(dock_template_best, (mapfile_whole, tpa, tpo,thr, bb_pdb_path, resolution, gid, tid, cid, gtid, tcn, do_refine))
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


def dock_all_chains_best(jsonfile, thr=None, dock1_cc_thr=0.2, dock2_cc_thr=0.2, resolution=3.0, found_value=0.5, map_scale: Union[str, bool] = False,
                    clip_res: float = 20.0, clip_thr: Union[float, int, str] = 0.0, build_with_group: bool = False, erased_map=None, 
                    config=None, idock=0, do_refine=True, keep_rep_num=False):

    assert isinstance(map_scale, (str, bool))
    if isinstance(map_scale, str):
        assert map_scale == 'auto', f"expect map_scale as auto when datatype of str"

    
    # config = load_json_config(jsonfile)
    key = 'group2templ' 
    if config is None:
        config = load_json_config(jsonfile)
    
    ma = map_file_from_config(config, resolution)
    
    
    # import pdb;pdb.set_trace()
    # need_erased_dock = df['cc'].min() < dock2_cc_thr or has_unsolved_chain
    # curr_chain_num = len(df)
    
    group2templ_info = config['group2templ_info']
    chain_num = 0
    for k,v in  group2templ_info.items():
        chain_num += v['rep_num']

    template2path = config['group2templ']
    templ_list = []
    tid_list = []
    gid_list = []
    gra_base = [0]
    for k,v0 in  template2path.items():
        v = v0[:10] 
        len_v = len(v)
        templ_list.extend(v)
        gid_list.extend([k]*len_v)
        tid_list.extend(list(range(len_v)))
        gra_base.append(gra_base[-1]+v[0]['rep_num'])
    group2templ= {
        'gid':gid_list,
        'tid':tid_list,
        'gtid':list(range(len(tid_list))),
        'templs': templ_list,
    }
    # import pdb;pdb.set_trace()

    accept_gtid_list = []
    accept_gid_list  = []
    accept_path_list = []
    global_accept_path_list = []
    it = 0
    it_empty = 0
    erasedmapfile2 = ma

    # while(len(accept_gtid_list) < len(templ_list)):
    while(len(accept_gtid_list) < chain_num):

        print(f"> {it} {len(accept_gtid_list)}/{chain_num}")

        dock_met_path = ma.replace('.mrc', f'_aem_it{it}.csv')
        if not os.path.exists(dock_met_path):
            dock_df, dock_met_path = dock_all_templates_best(erasedmapfile2, group2templ, dock_met_path, accept_gtid_list, thr=thr, it=it, do_refine=do_refine)
        else:
            dock_df = pd.read_csv(dock_met_path)
        print(dock_df)
        print(f"> {it} do filter corr({dock1_cc_thr})")

        dock_df_ = dock_df.loc[dock_df['corr']>=dock1_cc_thr]
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


        accept_list, reject_list = nms_pdb(dock_df, accept_path_list, accept_gid_list, found_value=found_value, cc_thr=dock1_cc_thr, keep_rep_num=keep_rep_num, group2templ_info=group2templ_info)
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
        erasedmapfile2 = ma.replace('.mrc', f'_aem_erased_it{it}.mrc')
        if not os.path.exists(erasedmapfile2):
            print(f"> {it} erasing map... with {resolution_for_erase}({resolution})")
            print(f"> {it} erasing map... with {pdb_paths_for_erase})")
            erase_map_with_pdb(pdb_paths_for_erase, erasedmapfile1, erasedmapfile2, resolution=resolution_for_erase)
        it += 1
    print("global_accept_path_list: ")
    print(global_accept_path_list)
    out_pdb = ma.replace('.mrc', f'_aem0.pdb')
    combine_pdbs(global_accept_path_list, out_pdb)
    print(out_pdb)


seem_finished = False


### Functions for Backbone (pd_tr_ca.pdb) refinement
def get_pdb_regions(bb_pdb_file, chain_name, domain):
    prd_seq_dict, bb_frags = read_prd_pdb(bb_pdb_file, min_frag_size=5, full_padding=False)
    # chain_id = PDB_CHAIN_IDS.index(chain_name)
    chain_id = decode_atom_id(chain_name)
    chain_bb_prots = [ bb_prot for bb_prot in bb_frags if bb_prot.chain_index[0] == chain_id ]
    chain_bb_prots.sort(key=lambda x: x.residue_index[0])
    bb_prot = chain_bb_prots[0]
    for prot in chain_bb_prots[1:]:
        bb_prot = bb_prot + prot
    
    bb_prot = fill_protein_to_nongapped(bb_prot, {chain_name: prd_seq_dict[chain_name]}, full_padding=True)

    mask = np.zeros_like(bb_prot.atom_mask[:, 1])
    for s, e in domain:
        mask[s:e+1] = 1.0
    mask = mask.astype(np.bool_) & bb_prot.atom_mask[:, 1].astype(np.bool_)
    
    bb_prot_sub = bb_prot.filter_by_mask(mask)

    assert bb_pdb_file.endswith('_pd_tr_ca.pdb'), bb_pdb_file
    chain_pdb_file = re.sub('_pd_tr_ca.pdb$', f"_bb_rv{chain_name}.pdb", bb_pdb_file)
    af2_features.save_prot_as_pdb(bb_prot_sub, chain_pdb_file, prd_seq_dict[chain_name])

    return chain_pdb_file

def load_bb_atoms_as_dictionary(chain_pdb_file):
    residues = {}
    fr = open(chain_pdb_file)
    for r_line in fr:
        if r_line.startswith('ATOM'):
            res_index = int(r_line[22:26].strip())
            residues[res_index] = residues.get(res_index, []) + [r_line.strip()]
    return residues

def make_final_chain(pdb_file, domain_list, rv_data, outfile):
    '''

    :param pdb_file: chain pdb derived from domain adaptation
    :param domain_list: domains adopted for this chain
    :param rv_data: atoms to be replaced by alignment result. Dict
    :param outfile: final output pdb
    :return:
    '''
    f = open(pdb_file)

    w = open(outfile, 'w')
    keep_residues = {}
    substitute_residues = {}
    for domain in domain_list:
        for fg in domain:
            s, e = fg
            for i in range(s, e+1):
                keep_residues[i] = 1

    if rv_data is not None:
        for s, e in rv_data['domain']:
            for i in range(s, e+1):
                keep_residues[i] = 0
        substitute_residues = rv_data['substitute_residues']

    for line in f:
        if line.startswith('ATOM'):
            res_index = int(line[22:26].strip())
            atom_name = line[12:16].strip()
            chain_name = line[21]
            if res_index in substitute_residues:
                if atom_name == 'CA':
                    sub_lines = substitute_residues[res_index]
                    for s_line in sub_lines:
                        s_line_re = s_line[:21] + chain_name + s_line[22:]
                        w.write(s_line_re + '\n')
            elif res_index-1 in keep_residues:
                w.write(line)
        else:
            w.write(line)
    f.close()
    w.close()

def save_results(jsonfile, key, final_path, min_domain_cc=0.2, only_domain=True, ignore_domain=False,
                 config=None, is_final=False, df=None):
    """
    Save as pdb file from jsonfile
    """
    print(_info('save_results...'))
    if config is None:
        config = load_json_config(jsonfile)
    if df is None:
        df, has_unsolved_chain = sortcc_group(config, key=key, sort=True)
        print("Still Has unsolved chain: ", has_unsolved_chain)
        # print(df)
        df.to_csv(final_path.replace(".pdb", '_met.csv'))

    # if 'unet_align_confident' in config:    # bb combining is on
    #     bb_pdb_file = config['cryonet']
    #     bb_advanced_file = bb_pdb_file.replace('_pd_tr_ca.pdb', '_2_backbone_atoms.pdb')
    #     if not os.path.exists(bb_advanced_file):
    #         print(_warn(f"warning: bb results with CNO: {bb_advanced_file} not exists! Use Ca atoms only"))
    #         bb_advanced_file = bb_pdb_file
    bb_pdb_file = config['cryonet']
    gt_pdb_file = bb_pdb_file.replace('_pd_tr_ca.pdb', '.pdb')
    if not os.path.exists(gt_pdb_file):
        print(_warn(f"warning: ground truth pdb file: {gt_pdb_file} not exists!"))
        gt_pdb_file = None
     
    pdb_files = []
    pdb_file_rids = []
    pdb_file_seqs = []
    pdb_file_types = []
    iii = 0
    k_domain = 0
    gid_tmp = list(config[key])[0]
    gid_str = isinstance(gid_tmp, str)
    # for k in range(len(df)):
    for k in df.index:

        gid = str(df['gid'][k]) if gid_str else int(df['gid'][k])
        gra = df['gra'][k]
        if 'rid' in df.keys():
            rid = df['rid'][k]
        else:
            rid = k_domain

        cid = df['cid'][k]
        # import pdb;pdb.set_trace()
        tid = int(df['tid'][k])
        cc  = df['cc'][k]
        cc_peak  = df['cc_peak'][k]
        pa  = df['path'][k]
        dos = config['group2templ'][int(gid)][tid]['domain_list']
        group2templ_info = config['group2templ_info'][int(gid)]
        seq      = group2templ_info['seq']
        num_rep  = group2templ_info['rep_num']
        seq_type = group2templ_info['type']
        
        if dos is None or pa is None:
            print("==== dos is None or pa is None", cc)
            continue
        if ignore_domain_thr>0 and (cc<ignore_domain_thr or cc_peak < -0.1):
            try:
                config[key][gid]['path'][gra] = None
            except:
                pass
            df['path'][k] = None
            continue
        # import pdb;pdb.set_trace()
        if select_topk>0 and k_domain>=select_topk:
            config[key][gid]['path'][gra] = None
            df['path'][k] = None
            continue
        k_domain += 1
    

        rv_data = None
        if iii>=1:
            overlap, overlapped_paths = is_overlap(pa, pdb_files, found_value=0.5)
            if overlap:
                # print(_warn(f"warning: {path} overlaps with {overlapped_paths}"))
                df['path'][k] = None

                continue
        iii += 1
        pdb_files.append(pa)
        pdb_file_rids.append(rid)
        pdb_file_seqs.append(seq)
        pdb_file_types.append( seq_type )
        

    df.to_csv(final_path.replace(".pdb", '_met_final.csv'))

    if len(pdb_files) ==0:
        print("Error!!! Empty chain paths!!!")
    else:
        print(f'Combining chains: {len(pdb_files)}')
        i = 0
        final_dir = os.path.dirname(final_path)+"/final"
        if os.path.exists(final_dir):
            os.system(f"rm -rf {final_dir}/*")
        else:
            os.makedirs(final_dir)
        if gt_pdb_file is not None:
            print(f"id\tlen\tfound\tseq_match\tchain_match")
        for path in pdb_files:
            rid = pdb_file_rids[i]
            i += 1
            # print(f"{i:2d}  {path}")
            if is_final:
                final_name_i = os.path.basename(path)
                region_id = int(os.path.basename(os.path.dirname(os.path.dirname(path))))
                final_path_i = f"{final_dir}/{region_id:02d}_{rid+1:02d}_{final_name_i}"
                os.system(f"cp {path} {final_path_i}")
                # import pdb;pdb.set_trace()
                if gt_pdb_file is not None:
                    met = chain_comparison(path, gt_pdb_file, return_results=True)
                    # print(met)
                    print(f"{i:2d}\t{met['gt']:.0f}\t{met['found']:.3f}\t{met['seq_match']:.3f}\t{met['chain_match']:.3f}")


        combine_pdb(pdb_files, pdb_file_seqs, pdb_file_types, final_path)



def find_bb_aligned_results(bb_pdb_path, seq, domain, min_frag_size=15):
    """
    Find all intervals in aligned file, lying in specified seq (group) and region
    """
    
    assert os.path.exists(bb_pdb_path) and os.path.getsize(bb_pdb_path) > 0, f"{bb_pdb_path} not exists"
    prd_seq_dict, bb_frags = read_prd_pdb(bb_pdb_path, min_frag_size=min_frag_size, full_padding=False)
    
    ## Firstly, find the U-Net fragments according to the sequence
    matched_bb_chain_name = None
    for chain_name, chain_seq in prd_seq_dict.items():
        if seq == chain_seq:
            matched_bb_chain_name = chain_name
            break
    assert matched_bb_chain_name is not None, f"Expect seq {seq} in {bb_pdb_path}, but not found"
    # matched_bb_chain_id = PDB_CHAIN_IDS.index(matched_bb_chain_name)
    matched_bb_chain_id = decode_chain_id(matched_bb_chain_name)

    ## Secondly, check the U-Net chain and it must be unique
    chain_bb_prots = [ bb_prot for bb_prot in bb_frags if bb_prot.chain_index[0] == matched_bb_chain_id ]
    if len(chain_bb_prots) == 0:
        print(_warn(f"No templates found for {seq} in {bb_pdb_path}"))
        return None, None

    chain_bb_prots.sort(key=lambda x: x.residue_index[0])
    chain_frag_regions = [ [bb_prot.residue_index[0], bb_prot.residue_index[-1]] for bb_prot in chain_bb_prots ]
    for idx in range(1, len(chain_frag_regions)):
        if chain_frag_regions[idx-1][1] > chain_frag_regions[idx][0]:
            print(_error(f"Expect not overlapped regions, have found in {bb_pdb_path}({matched_bb_chain_name}): {chain_frag_regions}"))
            return None, None
    
    ## Thirdly, Collect all regions which is overlaped with domain
    num_bb_res = 0
    prot_reg_list = []
    for bb_prot in chain_bb_prots:
        # print(f"bb_pdb_path: {bb_pdb_path}; matched_bb_chain_id: {matched_bb_chain_id}, seq: {seq}")
        filled_bb_prot = fill_protein_to_nongapped(bb_prot, {matched_bb_chain_id: seq}, full_padding=True)
        bb_frag_s, bb_frag_e = bb_prot.residue_index[0], bb_prot.residue_index[-1]
        for reg_s, reg_e in domain:
            is_overlap = (reg_s < bb_frag_e and bb_frag_s < reg_e)
            if is_overlap:
                mask = filled_bb_prot.atom_mask[:, 1].astype(np.bool_)
                mask[:reg_s] = False
                mask[reg_e+1:] = False
                if mask.sum() > 3:
                    num_bb_res += mask.sum()
                    prot_reg = filled_bb_prot.filter_by_mask(mask)
                    prot_reg_list.append(prot_reg)
                else:
                    print(_warn(f"Warning: Chain {matched_bb_chain_id} in {bb_pdb_path} match with domain {domain} too little residues."))
    
    domain_size = sum([ e-s+1 for s,e in domain ])
    cover_score = round(num_bb_res / domain_size, 4)

    return cover_score, matched_bb_chain_name

def copy_config_to_workdir(jsonfile, map_path, work_dir):
    """
    Copy template pdb files to work directory
    """
    jsonfile_new = f"{work_dir}/templ.json"

    templ_dir_old = str(pathlib.Path(jsonfile).parent)+"/templ"
    templ_dir_new = f"{work_dir}/templ"
    # if not os.path.exists(templ_dir_new):
    #     os.system("cp -r "+templ_dir_old+" "+work_dir)
    map_path_old = map_path
    map_name = os.path.basename(map_path)
    map_path_new = os.path.join(work_dir, map_name)
    den_map = DensityInfo(map_path_old)
    den_map.save(map_path_new)
    # os.system(f"cp -r {map_path_old} {map_path_new}")

    # remove the path of template pdb files
    config = load_json_config(jsonfile)
    config['map_file'] = map_path_new
    config['cryonet'] = map_path_new.replace(".mrc", "_pd_tr_ca.pdb")

    for grp_idx, grp_info in config['group2templ'].items():
        for templ_info in grp_info:
            if templ_info['path'] is None:
                continue
            templ_path = templ_info['path']
            templ_name = os.path.basename(templ_path)
            templ_new_path = os.path.join(templ_dir_new, templ_name)
            # if not os.path.exists(templ_new_path):
            #     shutil.copy(templ_path, templ_new_path)
            templ_info['path'] = templ_new_path
            if templ_info['pAE'] is not None:
                pAE_path = templ_info['pAE']
                pAE_name = os.path.basename(pAE_path)
                pAE_new_path = os.path.join(templ_dir_new, pAE_name)
                templ_info['pAE'] = pAE_new_path
    dump_config_dict(config, jsonfile_new)
    return jsonfile_new

#############################
### Entry
#############################

def pipeline_templ(jsonfile, docking=True, stg2_dock1_cc_thr=0.6, stg2_dock2_cc_thr=0.2,
             stg3_cc_thr=0.2, resolution=3.0, force_rerun=False, ovlp_found=0.5, map_scale='auto', clip_res=20.0,
             clip_thr=0.0, min_domain_save_cc=0.2, map_path='', pdb_path='', thr=None,do_refine=True,
             build_with_group=True, ignore_domain=False,ignore_domain_thr=0, select_topk=-1, keep_rep_num=False):
    """
    Run superimpose, docking, domain adaptation pipeline to assemble the protein structure model.
    """
    bb_pdb_path = pdb_path
    config = load_json_config(jsonfile)
    # mrc_path = map_path
    print(map_path)
    if not os.path.exists(map_path):
        map_path = map_file_from_config(config, resolution)
    else:
        config['map_file'] = map_path

    map_name = os.path.basename(map_path)
    if thr is None or thr < 0:
        den_map = DensityInfo(map_path)
        thr = den_map.mean1s.item()

    work_dir = re.sub('.mrc$', "",  map_path)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    jsonfile_raw = jsonfile
    jsonfile = copy_config_to_workdir(jsonfile_raw, map_path, work_dir)


    

    config['cryonet_sup'] = {}
    bb_pdb_path = config['cryonet']
    
    template2path = config['group2templ']
    # work_dir = str(pathlib.Path(jsonfile).parent)

    # jsonfile_stage1 = re.sub('.json$', "_stage1_sup.json",  jsonfile) # Only replace the last .json
    # jsonfile_stage2 = re.sub('.json$', "_stage2_dock.json", jsonfile) # Only replace the last .json
    # jsonfile_stage3 = re.sub('.json$', "_stage3_da.json",   jsonfile) # Only replace the last .json

    jsonfile_stage1 = re.sub('.mrc$', "_aem1.json", map_path) # Only replace the last .json
    jsonfile_stage2 = re.sub('.mrc$', "_aem2.json", map_path) # Only replace the last .json
    jsonfile_stage3 = re.sub('.mrc$', "_aem3.json", map_path) # Only replace the last .json
    jsonfile_stage4 = re.sub('.mrc$', "_aem4.json", map_path) # Only replace the last .json

    final0_path = re.sub('.mrc$', "_aem0.pdb",  map_path)
    final1_path = re.sub('.mrc$', "_aem1.pdb",  map_path)
    final2_path = re.sub('.mrc$', "_aem2.pdb",  map_path)
    final_path  = re.sub('.mrc$', "_aem.pdb",   map_path)


    if force_rerun:
        prefix = map_path.replace('.mrc', f'_aem_*')
        
        os.system(f" rm {prefix}")
        # if os.path.exists(jsonfile_stage1):
        #     print(_warn(f"force_rerun specified, {jsonfile_stage1} deleted"))
        #     os.remove(jsonfile_stage1)
        if os.path.exists(jsonfile_stage2):
            print(_warn(f"force_rerun specified, {jsonfile_stage2} deleted"))
            os.remove(jsonfile_stage2)
        if os.path.exists(jsonfile_stage3):
            print(_warn(f"force_rerun specified, {jsonfile_stage3} deleted"))
            os.remove(jsonfile_stage3)
        if os.path.exists(jsonfile_stage4):
            print(_warn(f"force_rerun specified, {jsonfile_stage4} deleted"))
            os.remove(jsonfile_stage4)

    if pathlib.Path(jsonfile_stage2).exists():
        print(_warn(f'docking file {jsonfile_stage2} already exists.'))
        jsonfile = jsonfile_stage2
        config = load_json_config(jsonfile)
        key = 'cryonet_sup_dock'
    # elif pathlib.Path(jsonfile_stage1).exists():
    #     print(_warn(f'superimposing file {jsonfile_stage1} already exists.'))
    #     if docking:
    #         dock_all_chains(jsonfile_stage1, dock1_cc_thr=stg2_dock1_cc_thr,
    #                        dock2_cc_thr=stg2_dock2_cc_thr, resolution=resolution,
    #                        found_value=stage2_ovlp_found, map_scale=map_scale,
    #                        clip_res=clip_res, clip_thr=clip_thr, build_with_group=build_with_group)
    #         key = 'cryonet_sup_dock'
    #         jsonfile = jsonfile_stage2
    #     else:
    #         key = 'cryonet_sup'
    #         jsonfile = jsonfile_stage1
    else:

        # ================== Stage 1: Superimpose ==================== #
        

        # ================== Stage 2: Docking chains ==================== #
        dock_all_chains_best(jsonfile, thr=thr, config=config, dock1_cc_thr=stg2_dock1_cc_thr,
                        dock2_cc_thr=stg2_dock2_cc_thr, resolution=resolution,
                        found_value=ovlp_found, map_scale=map_scale,
                        clip_res=clip_res, clip_thr=clip_thr, build_with_group=build_with_group,
                        do_refine=do_refine, keep_rep_num=keep_rep_num)
        # key = 'cryonet_sup_dock'
        # jsonfile = jsonfile_stage2
        # save_results(jsonfile, key, final1_path, only_domain=False, ignore_domain=ignore_domain, is_final=True)
        # config = load_json_config(jsonfile)


    # if pathlib.Path(jsonfile_stage3).exists():
    #     print(_warn(f'docking file {jsonfile_stage3} already exists.'))
    #     jsonfile = jsonfile_stage3
    #     config = load_json_config(jsonfile)
    #     key = 'cryonet_sup_dock2'
    # else:
        # ================== Stage 2.1: Docking superimposed chains ==================== #
    # df_path = final1_path.replace("_sup1.pdb", '_s1_sup_v0_met_final.csv')
    # df_saved = pd.read_csv(df_path)
    # df = df_saved.loc[ df_saved['path'].notnull() ]
    # df = df.reset_index(drop=True)
    # dock_superimposed_chains(jsonfile, jsonfile_stage3, dock1_cc_thr=stg2_dock1_cc_thr,
    #                         dock2_cc_thr=stg2_dock2_cc_thr, resolution=resolution,
    #                         found_value=stage2_ovlp_found, map_scale=map_scale,
    #                         clip_res=clip_res, clip_thr=clip_thr, build_with_group=build_with_group,
    #                         df=df)
    
    # # import pdb; pdb.set_trace()
    # # save_results(jsonfile, key, final0_path, only_domain=False, ignore_domain=ignore_domain)
    # key = "cryonet_sup_dock2"
    # save_results(jsonfile_stage3, key, final_path, only_domain=False, ignore_domain=ignore_domain, is_final=True)

    # ================== Stage 3: dock last ==================== #
    # key = "cryonet_sup_dock2"

    # dock_all_chains(jsonfile_stage1, dock1_cc_thr=stg2_dock1_cc_thr,
    #                     dock2_cc_thr=stg2_dock2_cc_thr, resolution=resolution,
    #                     found_value=stage2_ovlp_found, map_scale=map_scale,
    #                     clip_res=clip_res, clip_thr=clip_thr, build_with_group=build_with_group)
    #     key = 'cryonet_sup_dock'
    #     jsonfile = jsonfile_stage2
    #     save_results(jsonfile, key, final1_path, only_domain=False, ignore_domain=ignore_domain, is_final=True)
    # ================== Stage 3: Domain Adaptation ==================== #

    # domain_adaptation_v2(jsonfile,        cc_thr=stg3_cc_thr,     radius=20, resolution=resolution, found_value=stage3_ovlp_found, 
    #                      ignore_domain=ignore_domain, ignore_domain_thr=ignore_domain_thr, select_topk=select_topk)
    # key = 'cryonet_sup_dock_da'
    # save_results(jsonfile_stage3, key, final1_path, min_domain_cc=min_domain_save_cc, only_domain=True, ignore_domain=ignore_domain)
    # domain_adaptation_v2(jsonfile_stage3, cc_thr=stg3_cc_thr+0.2, radius=10, resolution=resolution, found_value=stage3_ovlp_found, 
    #                      ignore_domain=ignore_domain, ignore_domain_thr=ignore_domain_thr, select_topk=select_topk)
    # save_results(jsonfile_stage3, key, final_path, min_domain_cc=min_domain_save_cc, only_domain=True, ignore_domain=ignore_domain)
    # if os.path.exists(final_path):
    #     cc_dict = den_model_cc_phenix(mrc_path, final_path, resolution=resolution)
    #     if (cc_dict['CC_mask'] > 0.1 and cc_dict['CC_peaks'] <=0) or cc_dict['CC_mask'] / abs(cc_dict['CC_peaks']) >1.5:
    #         erasedmapfile = re.sub(".mrc$", "_erased_da.mrc", mrc_path)

    #         rid = int(re.sub(".mrc$", "", os.path.basename(mrc_path)))
    #         parent_name = os.path.dirname(jsonfile_raw) 
    #         region_list_path = f"{parent_name}/regions.list"
    #         with open(region_list_path, 'r') as f:
    #             n_regions = len(f.readlines())
    #         new_regiong_path = f"{parent_name}/regions/{rid+n_regions}.mrc"
    #         os.system(f"cp {erasedmapfile} {new_regiong_path} && chmod +w {new_regiong_path}")
    #         print("New region saved to ", new_regiong_path)
    #     # print(cc_dict)
    #     print(f"{map_name} CC_mask: {cc_dict['CC_mask']} CC_volume: {cc_dict['CC_volume']} CC_peaks: {cc_dict['CC_peaks']} CC_box: {cc_dict['CC_box']} CC_mc: {cc_dict['CC_mc']} CC_sc: {cc_dict['CC_sc']} ")
    #     # den_exp = DensityInfo(mrc_path) 
    #     # prot = af2_features.read_pdb(final_path, full_padding=True)
    #     # den_simu = afprot2density(prot, voxel_size=den_exp.apix, atom_weight=14.0, res=resolution)
    #     # # cc_frag_dock = den_cc(den_exp, den_simu, cc_type=CC_TYPE)
    #     # cc_righ = den_cc(den_exp, den_simu, cc_type=CC_TYPE, overlap_type="right")
    #     # cc_left = den_cc(den_exp, den_simu, cc_type=CC_TYPE, overlap_type="left")
    #     # print(f"cc_righ: {cc_righ} cc_left: {cc_left} ")
    #     print(_info(f"Superimposing step is Complete. Results are saved in {final_path}"))
    # else:
    #     print(f"{map_name} CC_mask: None CC_volume: None CC_peaks: None CC_box: None CC_mc: None CC_sc: None ")
    #     print(_info(f"No model is generated. Please check the log file."))

def save_regions(jsonfile):
    j="230408_000903_11_0f4536f470b2ff1bc4d017192aa09a2b_3"
    j="230109_150923_11_78ce0ec536922a35d4abf5112725a994_3"
    work_dir = f"data/jobs/{j}"
    final_path = f'{work_dir}/sup.pdb'
    key="cryonet_sup_dock2"
    names = ['rid', 'gid', 'gra', 'cid', 'tid', 'cc', 'cc_peak', 'seq_match', 'path', 'seq_len', 'rep_num', 'stype']
    dfs = []
    # for r in range(1,51):
    for r in range(1,49):
        reg_dir = f"{work_dir}/regions/{r}"
        met_path = f"{reg_dir}/{r}_sup_met.csv"
        if os.path.exists(met_path):
            df = pd.read_csv(met_path, names=names,header=0)
            dfs.append(df)
        else:
            print(met_path," not exists.")
    
    df_c = pd.concat(dfs)
    df_c = df_c.reset_index(drop=True)
    df_c.to_csv(final_path.replace(".pdb", "_met.csv"))
    print(df_c)
    save_results(jsonfile, key, final_path, only_domain=False, ignore_domain=True, is_final=True, df=df_c)
    print(f"save into {final_path}")
    # import pdb;pdb.set_trace()

def rearange_pdb(inpath, outpath):
    fit_prot = AF2_Prot_Module.from_pdb_string(open(inpath).read())
    af2_features.save_prot_as_pdb(fit_prot, outpath)
    print("save to ", outpath)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Assemble templates into full model.')
    parser.add_argument('config_file',          type=str,            help="Config file")
    parser.add_argument('-r', '--resolution',   type=float,          default=3.0,       help='Resolution value.')
    parser.add_argument('--stg2_dock1_cc_thr',  type=float,          default=0.2,       help='Lowest CC value cutoff to skip fine-tune the superposed chains.')
    parser.add_argument('--stg2_dock2_cc_thr',  type=float,          default=0.1,       help='CC value cutoff for stage 2 DOCK 2.')
    parser.add_argument('--stg3_cc_thr',        type=float,          default=0.2,       help='CC value cutoff for stage 3 domain adaptation.')
    parser.add_argument('--ignore_domain_thr',  type=float,          default=0.5,       help='CC value cutoff for stage 3 domain adaptation to ignore templ.')
    parser.add_argument('--min_domain_save_cc', type=float,          default=0.2,       help='Minimum CC value for domain preservation.')
    parser.add_argument('--ovlp_found',         type=float,          default=0.5,       help='Max found value to define overlap in stage 3.')
    parser.add_argument('--thr',                type=float,          default=None,       help='Max found value to define overlap in stage 3.')
    parser.add_argument('--stage2_map_scale',   type=str,            default='false',   help='Scale map to 1.5 in step 2, acceptable values: auto, true, false')
    parser.add_argument('--dock_method',        type=str,            default='adpem',   help='Dock tools, acceptable values: adpem, phenix')
    parser.add_argument('--map_path',           type=str,            default='',        help='density map path')
    parser.add_argument('--pdb_path',           type=str,            default='',        help='pdb path (traced Ca)')
    parser.add_argument('--clip_res',           type=float,          default=20.0,      help='Simulation resolution in docking process.')
    parser.add_argument('--clip_thr',           type=str,            default='0.0',     help='Mask threshold in docking process. Acceptable values: float, int or string meidan / mean')
    parser.add_argument('--cctype',             type=str,            default='pearson', help='Acceptable: pearson, cosine, spearman')
    parser.add_argument('--ccmethod',           type=str,            default='phenix', help='Acceptable: cryonet, phenix')
    parser.add_argument('--ncpu',               type=int,            default=8,         help='Number of maximum CPUs to use.')
    parser.add_argument('--select_topk',        type=int,            default=-1,         help='Number of maximum CPUs to use.')
    parser.add_argument('--force_rerun',        action='store_true', help='Force re-run the whole process.')
    parser.add_argument('--ignore_domain',      action='store_true', help='Force re-run the whole process.')
    parser.add_argument('--build_with_group',   action='store_true', help='Build with group.')
    parser.add_argument('--save_reg',       action='store_true', help='save_regions.')
    parser.add_argument('--rearangepdb',       action='store_true', help='rearangepdb.')
    parser.add_argument('--no_refine',       action='store_true', help='rearangepdb.')
    parser.add_argument('--keep_rep_num',       action='store_true', help='rearangepdb.')
    parser.add_argument('--inpath',           type=str,            default='',        help='pdb path (traced Ca)')
    parser.add_argument('--outpath',           type=str,            default='',        help='pdb path (traced Ca)')


    args = parser.parse_args()

    jsonfile = args.config_file
    resolution = args.resolution
    stg2_dock1_cc_thr = args.stg2_dock1_cc_thr
    stg2_dock2_cc_thr = args.stg2_dock2_cc_thr
    stg3_cc_thr = args.stg3_cc_thr
    min_domain_save_cc = args.min_domain_save_cc
    force_rerun      = args.force_rerun
    ignore_domain    = args.ignore_domain
    ignore_domain_thr= args.ignore_domain_thr
    select_topk      = args.select_topk
    build_with_group = args.build_with_group
    ovlp_found = args.ovlp_found
    map_scale = args.stage2_map_scale
    map_path = args.map_path
    pdb_path = args.pdb_path
    thr = args.thr
    DOCK_METHOD = args.dock_method
    CC_TYPE = args.cctype
    CC_METHOD = args.ccmethod
    clip_res = args.clip_res
    clip_thr = args.clip_thr
    NCPU = args.ncpu
    save_reg = args.save_reg
    rearangepdb = args.rearangepdb
    do_refine = not args.no_refine
    keep_rep_num = args.keep_rep_num

    assert map_scale in ('auto', 'true', 'false')
    if map_scale == 'true':
        map_scale = True
    elif map_scale == 'false':
        map_scale = False

    try:
        clip_thr = float(clip_thr)
    except ValueError:
        assert clip_thr in ('median', 'mean')

    assert DOCK_METHOD in ('adpem', 'phenix')

    assert CC_METHOD in ('cryonet', 'phenix')

    assert CC_TYPE in ('pearson', 'cosine', 'spearman')

    if rearangepdb:
        rearange_pdb(args.inpath, args.outpath)
    elif save_reg:
        save_regions(jsonfile)
        
    else:
        pipeline_templ(jsonfile, docking=True, stg2_dock1_cc_thr=stg2_dock1_cc_thr, stg2_dock2_cc_thr=stg2_dock2_cc_thr, stg3_cc_thr=stg3_cc_thr, 
             min_domain_save_cc=min_domain_save_cc, resolution=resolution, force_rerun=force_rerun, ovlp_found=ovlp_found, map_scale=map_scale, clip_res=clip_res, thr=thr,
             clip_thr=clip_thr,map_path=map_path, pdb_path=pdb_path, build_with_group=True, ignore_domain=ignore_domain, 
             ignore_domain_thr=ignore_domain_thr, select_topk=select_topk, do_refine=do_refine, keep_rep_num=keep_rep_num)

