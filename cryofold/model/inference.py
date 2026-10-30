#!/usr/bin/env python

import pathlib, argparse, json, os, sys, tempfile, re, shutil, glob, pickle, gzip
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import List, Union, Optional
import os
import numpy as np
import torch

from cryofold.diff.structure.protein_chain import ProteinChain
from cryofold.diff.structure.protein_complex import ProteinComplex, protein_chain_list_to_protein_complex
from cryofold.model.sm import CF
from cryofold.model.api import (
    CFProtein,
    GenerationConfig,
)
import random
import torch.backends.cudnn as cudnn

import torch._dynamo
torch._dynamo.config.suppress_errors = True


from cryofold.data.msa_template import run_phmmer_seqdb, build_templates_from_phmmer_cifdb
from cryofold.np import protein, residue_constants
from cryofold.utils import General, _info, _warn, _error
from cryofold.model.backbone_imp import build_backbone_imp
from cryofold.model.refinement import do_refinement, do_refinement_best
from cryofold.model.refinement_s3 import refine


load_fasta  = General.load_fasta


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
            'seq_num': len(seq),
            'rep_num': rep_num,
            'type':    seq_type,
            'do_func': do_func,
            'pred_templ':[]
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
    

def load_atomic_model(model_path):
    """
    Load atomic model from path
    """
    if model_path.lower().endswith(('.pdb')):
        atomic_model = ProteinChain.from_pdb_full(model_path)
    else:
        atomic_model = ProteinChain.from_cif(model_path)
    return atomic_model

def cb_pred(pdb1, pdb2, mappath, resolution=3.0):
    pc1 = load_atomic_model(pdb1)
    pc2 = load_atomic_model(pdb2)

    tgt_inds = list(range(pc1.atom37_positions.shape[0]))
    prd_inds = list(range(pc2.atom37_positions.shape[0]))
    pc2_ali_path = pdb2.replace(".pdb", "_ali.pdb")
    # breakpoint()
    if len(tgt_inds) != len(prd_inds):
        print(f"{len(tgt_inds)} {len(prd_inds)}")
        breakpoint()
    assert len(tgt_inds) == len(prd_inds), f"{len(tgt_inds)} {len(prd_inds)}"

    pc2_ali = pc2.align(pc1, prd_inds, tgt_inds)
    pc2_ali.to_pdb(pc2_ali_path)
    outs = do_refinement(pc2_ali_path, mappath, outpath=None, resolution=resolution, 
                         islocal=True, return_path=False)
    
    return outs[0]['pc'], outs[0]['corr']
    
def cp_all_chains_best(config, mappath, final_path, resolution=4.0):

    group2templ_info = config['group2templ_info']
    chain_num = 0
    for k,v in  group2templ_info.items():
        chain_num += v['rep_num']

    template2path = config['group2templ']
    
    # sup_dir = os.path.dirname(final_path) +  f'/sup'
    # os.system(f"mkdir -p {sup_dir}")

    # wtempl_dir = os.path.dirname(final_path) +  f'/.templ'
    pc_list = []
    for gidx, v0 in  template2path.items():
        v = v0[:10] 
        # len_v = len(v)
        # print(k,v0)
        i=1
        for v1 in v0:
            # print(v1)
            ref_path = v1['path']
            templ_dir = os.path.dirname(ref_path)
            prd_ali_best = None
            prd_cor_best = 0
            pt_list = " ".join(group2templ_info[gidx]['pred_templ'])
            # print(pt_list)
            for prd_path in group2templ_info[gidx]['pred_templ']:
                # prd_path = group2templ_info[gidx]['pred_templ'][0]
                # prd_path = f'{templ_dir}/{prd_path}'.replace(".cif",".pdb")
                # out_dir  = f'{templ_dir}/sup/'
                # out_path = f'{sup_dir}/{gidx}_{i}.pdb'
                prd_ali, prd_ali_cc = cb_pred(ref_path, prd_path, mappath, resolution)
                if prd_ali_cc > prd_cor_best:
                    prd_ali_best = prd_ali 
                    prd_cor_best = prd_ali_cc
            if prd_ali_best is not None:
                pc_list.append(prd_ali_best)
            i += 1
    # os.system(f"rm -rf {wtempl_dir}")

    return pc_list
    

    
    # sys.exit()



def run_inference(input_fasta, input_pklgz, input_mrc, work_dir, output_path=None, exclude_pdbs=[], max_pdb_date=None, max_pdb_templ=100, 
                    ncores=10, min_pdb_templ_cov=0.1, min_identity=0.1, cus_templ_path="", msa_wo=False):
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
    mrc_path = Path(input_mrc)
    basename = mrc_path.name.replace(mrc_path.suffix,"")

    if output_path is None:
        output_path = f"{mrc_path.parent}/{basename}_cryofold.pdb"
        output_path = f"{mrc_path.parent}/{basename}_cf.pdb"

    if not msa_wo:
        features = pickle.load(gzip.open(input_pklgz, 'rb'))
    else:
        features = None
    
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
    if features is not None:
        if  len(features['aatype'].shape) == 2:
            aatype_fea = features['aatype'].argmax(axis=1)
        else:
            aatype_fea = features['aatype']
        # aatype_check = (aatype==aatype_fea).all()
        # assert aatype_check, f"sequence in fasta {input_fasta} and *.pkl.gz {input_pklgz} file do not match"

        aatype_check = len(aatype)==len(aatype_fea)
        # if not aatype_check:
        #     print(_error(f"Warning: sequence in fasta {input_fasta} and *.pkl.gz {input_pklgz} file do not match"))
        #     breakpoint()
        assert aatype_check, f"sequence in fasta {input_fasta} {len(aatype)} and *.pkl.gz {input_pklgz} {len(aatype_fea)} file do not match"

    #print("name2seq: ", len(name2seq))
    check_seq_dict(name2seq)

    print("Running model inference...")

    # -----------------------

    # model =  CF.from_pretrained("cf_v1.0", device=torch.device("cuda"))
    # model.eval()
    # protein_prompt = CFProtein(sequence=seqs_str)
    # structure_prediction_config = GenerationConfig(
    #     track="structure", # We want CF3 to generate tokens for the structure track
    #     num_steps=len(seqs_str) // 8,
    #     temperature=0.7,
    # )

    # try:
    #     structure_prediction,_ = model.generate(protein_prompt, structure_prediction_config)
    #     del structure_prediction
    # except :
    #     structure_prediction  = model.generate(protein_prompt, structure_prediction_config)
    #     del structure_prediction

    # -----------------------

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
        # debug 
        # breakpoint()
        os.system(f"rm -rf {out_templ_files_dir};mkdir -p {out_templ_files_dir}")
        templ_prots, templ_files, templ_ids = build_templates_from_phmmer_cifdb(phmmer_outfile, seq, templ_files_dir,
            out_templ_files_dir, min_coverage=min_pdb_templ_cov, min_identity=min_identity, dbtype='cifdb', 
            top_n=max_pdb_templ, auto_fix=False, verbose=0)
        # breakpoint()
        if templ_prots is not None:
            accept_ids = []
            # print("cadidates: ", templ_ids)
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
            
    if chain_num > len(used_ids):
        gt_ids = cifdb_name2seq.keys()
        # print(f"#seq: {chain_num}, #used: {len(used_ids)}, #templ: {len(gt_ids)}")

        # if len(gt_ids) == len(used_ids):
        #     print(f"all the chains in the templates are comfirmed, but not all the chains in the input fasta are comfirmed.")
        # else:
        #     print(f"not all the chains in the templates are comfirmed.")
        #     sys.exit()
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
    # single chain for each group
    single_chain_per_group = False
    if single_chain_per_group:
        for gid, group2templ in config_dict['group2templ'].items():
            if len(group2templ)>0 and 'path' in group2templ[0]:
                in_path = group2templ[0]['path']
                out_dir = os.path.dirname(in_path)
                out_path0 = f'{out_dir}/msa_idx_0.pdb'
                out_path1 = f'{out_dir}/msa_idx_1.pdb'
                build_backbone_imp(in_path, out_path0, out_path1)
    else:
        max_templ_num = 3 # max number of templates to use for each group
        for gid, group2templ in config_dict['group2templ'].items():
            config_dict['group2templ_info'][gid]['pred_templ'] = []

            tid = 0
            for templ_info in group2templ:
                in_path = templ_info['path']
                # print(gid, tid, in_path)
                out_dir = os.path.dirname(in_path)
                out_path0 = f'{out_dir}/msa_idx_{tid}.pdb'
                out_path1 = f'{out_dir}/msa_idx_{tid}_bb.pdb'
                if os.path.exists(out_path0) and os.path.exists(out_path1):
                    breakpoint()
                    continue
                # if gid == 1:
                #     breakpoint()
                build_backbone_imp(in_path, out_path0, out_path1)
                # breakpoint()
                if os.path.exists(out_path0):
                    config_dict['group2templ_info'][gid]['pred_templ'].append(out_path0)
                    tid += 1
                else:
                    print(f"Failed to build backbone for {in_path} {out_path0}")
                if tid >= max_templ_num:
                    break
    

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_cp_path = os.path.join(tmp_dir, basename+".pdb")
        pc_list = cp_all_chains_best(config_dict, input_mrc, tmp_cp_path, resolution=4.0)

        if len(pc_list) == 0:
            print(f"Failed to predict the complex, please check the input sequences at file: {input_fasta}.")
            breakpoint()
        else:
            pcompl = protein_chain_list_to_protein_complex(pc_list)
            pcompl.to_pdb(tmp_cp_path)
            
            if int(chain_num) != len(pc_list):
                output_path_p = Path(output_path).parent
                output_path2 = f"{output_path_p}/{basename}_cf_{len(pc_list)}-{chain_num}.pdb"
                pcompl.to_pdb(output_path2)
            # print("Refine file: ", tmp_cp_path)
            refine(input_mrc, tmp_cp_path, output_path)
            print("Predicted file: ", output_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build the template from PDB and AFDB')
    parser.add_argument('-i', '--input_fasta', type=str,            required=True, help="Protein sequence fasta file.")
    parser.add_argument('-w', '--work_dir',    type=str,            required=True, help='Working directory.')
    parser.add_argument('--exclude_pdbs',      type=str,            required=False, help='PDB list to excluded, multiple PDB seperated by comma')
    parser.add_argument('--ncores',            type=int,            default=8,      help='Max number of cores to processing templates')
    parser.add_argument('--min_pdb_templ_cov', type=float,          default=0.01,   help='Minimun coverage of PDB template of sequence')
    parser.add_argument('--min_identity',      type=float,          default=0.1,   help='Minimun coverage of PDB template of sequence')
    
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
    cus_templ_path  = args.cus_templ_path

   

    run_inference(input_fasta, work_dir, ncores=ncores, min_pdb_templ_cov=min_pdb_templ_cov, min_identity=min_identity, 
                  max_pdb_templ=max_pdb_templ, map_file=map_file, cus_templ_path=cus_templ_path)
   