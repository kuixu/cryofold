#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""

2022-07-26: Combined code from L.Y.G.(combine U-Net backbone with templates), B.H.L. (sort by length), P.L. (Overlap & Ab chain)
2022-09-19: Pan modified
    1. min_domain_save_cc to define cc value for domain;
    2. Replace kui's StructureBuilder with AF2_Protein & af2_features.py operations

"""

import shutil, os, sys, time, random, pickle, io, gzip, re, json, platform, tempfile, importlib, pathlib, argparse, multiprocessing, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import hashlib
import multiprocessing as mp
from typing import List, Union, Optional
from Bio.SVDSuperimposer import SVDSuperimposer
import torch
torch.set_num_threads(1)

from cryofold.libs.structure.hy36 import encode_atom_id, encode_chain_id, decode_atom_id, decode_chain_id, PDB_MAX_CHAINS

from cryofold.np import protein, residue_constants
from cryofold.np import residue_constants as rc

### Check dynamic library path and adpem executable
# if 'LD_LIBRARY_PATH' not in os.environ:
#     os.environ['LD_LIBRARY_PATH'] = ''

# conda_lib = os.path.join(os.environ['HOME'], 'usr/miniconda3/lib')
# if os.path.exists(conda_lib):
#     os.environ['LD_LIBRARY_PATH'] = conda_lib + ":" + os.environ['LD_LIBRARY_PATH'].lstrip(':')

cpx = '_internal/cpx/bin/cpythonx'
if not os.path.exists(cpx):
    cpx = 'tools/cpx/bin/cpythonx'
    assert os.path.exists(cpx)

refine_lo = '_internal/cpx/bin/adpem_opt'
if not os.path.exists(refine_lo):
    refine_lo = 'bin/adpem_opt'
    assert os.path.exists(refine_lo)

refine_gl = '_internal/cpx/bin/adpem_opt'
if not os.path.exists(refine_gl):
    refine_gl = 'bin/adpem'
    assert os.path.exists(refine_gl)

# status, output = subprocess.getstatusoutput('adpem')
# assert status == 1, f"Expect adpem in PATH, but got {output}"

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
# from cryonet.util.chain_comparison   import chain_comparison
from cryofold.libs.structure.density  import DensityInfo, mol_atom_density_np, mol_atom_density_th

from cryofold.np import protein           as AF2_Prot_Module
from cryofold.np import protein           as AF2_Protein
from cryofold.np import residue_constants as AF2_Res_Constants
# AF2_Prot_Module.extend_PDB_Chains(extend_to=500) # More chains

restypes_with_x = AF2_Res_Constants.restypes_with_x
restype_3to1    = AF2_Res_Constants.restype_3to1
restype_1to3    = AF2_Res_Constants.restype_1to3
# PDB_CHAIN_IDS   = AF2_Prot_Module.PDB_CHAIN_IDS

from cryofold.data import msa_template
from cryofold.utils import General, Colors

read_pdb_atom_line        = msa_template.read_pdb_atom_line        # Parse a pdb file ATOM line
read_pdb_seq              = msa_template.read_pdb_seq              # Read and parse SEQRES lines from pdb file
write_pdb_seq             = msa_template.write_pdb_seq             # Write sequence to SEQRES lines
fill_protein_to_nongapped = msa_template.fill_protein_to_nongapped # Fill gapped AF2_Protein to unggaped

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


def file_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()
#################################
### For json loading and parsing
#################################



# def dock_temp_raw(ma, ma_raw, pa, bb_pdb_path, resolution=3.0, gid=0, tid=0, cid=0, idock=0, stage='v0'):

#     outpath0 = re.sub('.pdb$', f"_{cid}_{idock}_{stage}_d0.pdb",  pa)
#     outpath1 = re.sub('.pdb$', f"_{cid}_{idock}_{stage}_d.pdb",  pa)
#     dockmapfile = ma
#     den_clipped = dockmapfile

#     if DOCK_METHOD == 'adpem':
#         dok_file = dock_pdb_into_map(pa, dockmapfile, outpath0, resolution=resolution)
#     else:
#         dok_file = outpath0
#         assert phenix_dock_in_map(pa, dockmapfile, resolution, 1, dok_file, quick=False, min_cc=0.2) == 1
    
#     fit_pdb_into_map(outpath0, ma_raw, outpath1)
#     os.system(f"rm {outpath0}")
#     dok_file = outpath1
#     cc_dock, cc_dock_peak = den_model_cc(den_clipped, dok_file, resolution=resolution, cc_type=CC_TYPE, 
#                                          cc_method=CC_METHOD, return_cc_peak=True)
#     met = chain_comparison(dok_file, bb_pdb_path, return_results=True)
#     met = {k:v.item() for k,v in met.items()}
#     results_dict = {
#         "cc":cc_dock, "cc_peak":cc_dock_peak, "path":dok_file, "gid":gid, "tid":tid, "cid":cid, "seq_match":met['seq_match'], "stype":1
#     }
#     return results_dict



# def phenix_dock_in_map(pdbfile, mapfile, resolution, nproc, outpdbfile, quick=False, min_cc=0.2):
#     """
#     Use phenix.dock_in_map to dock a single PDB chain to target_map
    
#     Parameters
#     -----------
#     pdbfile: str
#     mapfile: str
#     resolution: float
#     nproc: int
#     outpdbfile: str
#     quick: bool
    
#     Return
#     -----------
#     status: int
#         0 -- Failed
#         1 -- Success
#     """
#     from multiprocessing import Process
#     import tempfile, subprocess, os

#     assert os.path.exists(pdbfile), f'{pdbfile} does not exists'
#     assert os.path.exists(mapfile), f'{mapfile} does not exists'
#     assert pdbfile.endswith('.pdb')

#     pdbfile = os.path.abspath(pdbfile)
#     mapfile = os.path.abspath(mapfile)
#     outpdbfile = os.path.abspath(outpdbfile)

#     if os.path.exists(outpdbfile):
#         os.remove(outpdbfile)

#     with tempfile.TemporaryDirectory(prefix='dockinmap_') as work_dir:
#         if quick:
#             refine_cycle = 1
#         else:
#             refine_cycle = 3
#         cmd  = f"cd {work_dir}; phenix.dock_in_map {mapfile} search_model={pdbfile} resolution={resolution} "
#         cmd += f"refine_cycle={refine_cycle} quick={quick} use_symmetry=False nproc={nproc} pdb_out={outpdbfile} min_cc={min_cc}"
#         status, output = subprocess.getstatusoutput(cmd)

#     if os.path.exists(outpdbfile):
#         return 1
#     else:
#         print(_error(f"Phenix dock_in_map failed: {cmd}"))
#         print(f"Phenix output: {output}")
#         return 0

# def dock_pdb_into_map(pdbpath, mappath, outpath, resolution=3.0):
#     """
#     Use adpem to dock PDB atom model to 3D density map
#     """
#     den_exp = DensityInfo(mappath)

#     tmp_map = tempfile.NamedTemporaryFile(delete=True, suffix=".mrc")
#     tmp_pdb = tempfile.NamedTemporaryFile(delete=True, suffix=".pdb")
#     tmp_out = tempfile.NamedTemporaryFile(delete=True)
    
#     shutil.copy2(mappath, tmp_map.name)
#     shutil.copy2(pdbpath, tmp_pdb.name)
    
#     den_mean = max(den_exp.mean, 0)
#     # print(_info(f"adpem {mappath} {pdbpath}"))

#     cmd = 'adpem {} {} 16 {} {} -n 1 -o {} '.format(tmp_map.name, tmp_pdb.name, den_mean, resolution, tmp_out.name)
#     # print(_info(cmd))
#     output_results_str = subprocess.getoutput(cmd)
#     dok_file = tmp_out.name + "0001.pdb"
#     if pathlib.Path(dok_file).exists():
#         shutil.copy2(dok_file, outpath)
#     else:
#         print(_error(f"Warning: Docking error -- {cmd}, please check"))
#         outpath = pdbpath
#     seq_dict = af2_features.read_pdb_seq(pdbpath)
#     prot = af2_features.read_pdb(outpath)
#     af2_features.save_prot_as_pdb(prot, outpath, seq_dict)
#     return outpath

# def fit_pdb_into_map(pdbpath, mappath, outpath, resolution=3.0):
#     """
#     Use adpem to dock PDB atom model to 3D density map
#     """
#     fit_seq = AF2_Prot_Module.seq_from_pdb_seqres(pdbpath)
#     tmp_path0= outpath.replace('.pdb', '_tmp0.pdb')
#     tmp_path1= outpath.replace('.pdb', '_tmp1.pdb')
#     cmd = f'ChimeraX --nogui --cmd "open {pdbpath}; open {mappath};fitmap #1 inMap #2; save {tmp_path0} models #1 relModel #2;exit;" >> /dev/null 2>&1'
#     cmd = cmd+f'; real_refine_sup.sh {mappath} {tmp_path0} {tmp_path1} {resolution} >> /dev/null 2>&1;'
#     os.system(cmd)

#     fit_prot = AF2_Prot_Module.from_pdb_string(open(tmp_path1).read())
#     af2_features.save_prot_as_pdb(fit_prot, outpath, full_seq=fit_seq)
#     os.system(f"rm {tmp_path0} {tmp_path1}")

#     return outpath

def parse_adpem_err(dok_file):
    err_code = {}
    with open(dok_file) as f:
        lines = f.readlines()
        for line in lines:
            if "Error->Mask" in line:
                err_code['err_code'] = 1
                err_code['err_msg'] = "Error->Mask"
            # elif "Error->Mask" in line:
            #     err_code['err_code'] = 1
            #     err_code['err_msg'] = "Error->Mask"
            else:
                err_code['err_code'] = 0
                err_code['err_msg'] = "ok"
              
    return err_code

def parse_adpem_log(dok_file, n):
    corr_dict = {}
    with open(dok_file) as f:
        lines = f.readlines()
        dok_region = False
        i=-2
        for line in lines:
                
            if "--------------------------------------------------------------------------" in line:
                break
            if dok_region:
                i+=1
                if i>=1:
                    fields = line.strip().split()
                    # print(fields)
                    try:
                        corr_dict[i-1] = float(fields[8])
                    except IndexError:
                        print("fields: ",len(fields), dok_file)
                    assert i == int(fields[1])
            
            if i == n:
                break
            if line.startswith("adpem>        Psi   Theta     Phi         X        Y        Z    Corr"):
                i+=1
                dok_region = True
    return corr_dict

def parse_mmcc_log(cc_output, per_chain_cc=False):
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


 
class Docking:

    def __init__(self, pdbpath, mappath, outpath=None, thr=None, bb_pdb_path=None, resolution=3.0, do_docking=True):
        self.pdbpath = pdbpath
        self.mappath = mappath
        self.outpath = outpath
        self.thr = thr
        self.bb_pdb_path = bb_pdb_path
        self.resolution = resolution
        self.do_docking = do_docking

        
        self.outpath_tmp = tempfile.NamedTemporaryFile(delete=True).name
        # self.outpath_tmp = self.outpath.replace('.pdb', '.dok')

        if do_docking:
            self.outpath_name = self.outpath.replace('.pdb', '')
            
            # self.mappath_tmp = tempfile.NamedTemporaryFile(delete=True, suffix=".mrc").name
            # self.pdbpath_tmp = tempfile.NamedTemporaryFile(delete=True, suffix=".pdb").name

            self.mappath_tmp = f"/tmp/{file_md5(mappath)}.mrc"
            self.pdbpath_tmp = f"/tmp/{file_md5(pdbpath)}.pdb"

            if not pathlib.Path(self.pdbpath_tmp).exists():
                # shutil.copy2(mappath, tmp_map.name)
                # shutil.copy2(mappath, tmp_map)
                shutil.copy2(self.pdbpath, self.pdbpath_tmp)
            
            if not pathlib.Path(self.mappath_tmp).exists():
                # shutil.copy2(pdbpath, tmp_pdb.name)
                # shutil.copy2(pdbpath, tmp_pdb)
                shutil.copy2(self.mappath, self.mappath_tmp)
            self.seq_dict = af2_features.read_pdb_seq(self.pdbpath)

            self.cmd_dock = 'adpem '+self.mappath_tmp+' {} 16 '+str(self.thr)+' '+str(self.resolution)+' -n {} -o {} 2>&1 > {};rm '+self.mappath_tmp+';rm '+self.pdbpath_tmp+';'
            self.cmd_dock = 'adpem '+self.mappath_tmp+' {} 16 '+str(self.thr)+' '+str(self.resolution)+' -n {} -o {} 2>&1 > {};'
            self.cmd_chix = 'ChimeraX --nogui --cmd "open {}; open '+self.mappath_tmp+';fitmap #1 inMap #2; save {} models #1 relModel #2;exit;" >> /dev/null 2>&1;'
            self.cmd_refi = 'real_refine_sup.sh '+self.mappath+' {} {} '+str(self.resolution)+' >> {} 2>&1;'
        else:
            self.outpath_name = self.outpath_tmp

        self.cmd_mmcc = 'phenix.map_model_cc '+self.mappath+' {} resolution='+str(self.resolution)+ \
                        ' file_name_prefix='+self.outpath_tmp+' ignore_symmetry_conflicts=True scattering_table=it1992 2>&1 ;'

    def get_thr():
        if self.thr is None:
            den_exp = DensityInfo(self.mappath, datatype='numpy')
            self.thr = max(den_exp.mean, 0)

    def get_cmd_dock(self, pdbpath, outpath, logpath, n=1):
        return self.cmd_dock.format(pdbpath, n, outpath, logpath)
    
    def get_cmd_chix(self, pdbpath, outpath):
        return self.cmd_chix.format(pdbpath, outpath)
    
    def get_cmd_refine(self, pdbpath, outpath, logpath):
        return self.cmd_refi.format(pdbpath, outpath, logpath)
    
    def get_cmd_mmcc(self, pdbpath):
        return self.cmd_mmcc.format(pdbpath)
    

    def refine(self, pdbpath):
        tmp_path0 = pdbpath.replace('.pdb', '_r0.pdb')
        log_path0 = pdbpath.replace('.pdb', '_r0.log')
        cmd_chix = self.get_cmd_chix(pdbpath, tmp_path0)
        cmd_refi = self.get_cmd_refine(tmp_path0, pdbpath, log_path0)
        os.system(f'{cmd_chix} {cmd_refi}')

    def calculate_chain_match(self, pdbpath):
        cmd_mmcc = self.get_cmd_mmcc(pdbpath)

        status, output = subprocess.getstatusoutput(cmd_mmcc)
        # print(cmd_mmcc)
        cc_dict = parse_mmcc_log(output)
        
        if self.bb_pdb_path is not None:
            met = chain_comparison(pdbpath, self.bb_pdb_path, return_results=True)
            met = {k:v.item() for k,v in met.items() if k in ['found', 'seq_match', 'chain_match']}
            met['cc'] = cc_dict['CC_mask']
            met['cc_peak'] = cc_dict['CC_peaks']
        else:
            met = {"cc":cc_dict['CC_mask'], "cc_peak":cc_dict['CC_peaks'], 'found':0.0, 'seq_match':0.0, 'chain_match':0.0, }
        
        return met


    def run(self, n=1, corr_thr=0.4, do_refine=False, do_cm=False, return_df=False):
        assert n > 0 and n < 100
        
        logpath = self.outpath_tmp.replace(".pdb", ".dok")
        logpath = self.pdbpath.replace(".pdb", ".dok")
        cmd = self.get_cmd_dock(self.pdbpath_tmp, self.outpath_tmp, logpath, n)
        print(cmd)
        output_results_str = subprocess.getoutput(cmd)
        # import pdb;pdb.set_trace()
        err_code  = parse_adpem_err(logpath)
        assert err_code['err_code'] == 0, err_code['err_msg']

        corr_dict = parse_adpem_log(logpath, n)
        # print(corr_dict)
        outs = []
        results = []
        for i in range(len(corr_dict)):
            dok_file0 = self.outpath_tmp + f"00{i+1:0>2d}.pdb"
            dok_file1 = self.outpath_tmp + f"_{i+1:0>2d}.pdb"
            if corr_dict[i] > corr_thr:
                os.system(f"mv {dok_file0} {dok_file1}")
                outs.append(dok_file1)
                results.append({'id':i, 'path': dok_file1, 'corr': corr_dict[i]})  
            else:
                os.system(f"rm {dok_file0}")   

        if do_refine:
            # print("do_refine")
            for dokpdbpath_tmp in outs:
                self.refine(dokpdbpath_tmp)
        if do_cm:
            # print("do_cm")
            i = 0
            for dokpdbpath_tmp in outs:
                cm = self.calculate_chain_match(dokpdbpath_tmp)
                for k,v in cm.items():
                    results[i][k] = v
                i += 1

                

        outs_final = []
        i = 0
        for dokpdbpath_tmp in outs:
            outpath = dokpdbpath_tmp.replace(self.outpath_tmp, self.outpath_name)
            prot = af2_features.read_pdb(dokpdbpath_tmp)
            af2_features.save_prot_as_pdb(prot, outpath, self.seq_dict)
            outs_final.append(outpath)
            results[i]['path'] = outpath
            i += 1
        # import pdb;pdb.set_trace()
        if return_df:
            results = pd.DataFrame(results)
            # print(results)
        return results
    
    def run_best(self, n=1, do_refine=False, do_cm=False, return_df=False):
        assert n > 0
        
        logpath = self.outpath_tmp.replace(".pdb", ".dok")
        cmd = self.get_cmd_dock(self.pdbpath_tmp, self.outpath_tmp, logpath, n)
        print(cmd)
        output_results_str = subprocess.getoutput(cmd)
        # import pdb;pdb.set_trace()
        corr_dict = parse_adpem_log(logpath, n)
        # print(corr_dict)
        outs = []
        results = []
        for i in range(len(corr_dict)):
            dok_file0 = self.outpath_tmp + f"0{i+1:0>3d}.pdb"
            dok_file1 = self.outpath_tmp + f"_{i+1:0>3d}.pdb"
            # if corr_dict[i] > corr_thr:
            os.system(f"mv {dok_file0} {dok_file1}")
            outs.append(dok_file1)
            results.append({'id':i, 'path': dok_file1, 'corr': corr_dict[i]})
            # print({'id':i, 'path': dok_file1, 'corr': corr_dict[i]})  
            # else:
            #     os.system(f"rm {dok_file0}")   

        if do_refine:
            # print("do_refine")
            for dokpdbpath_tmp in outs:
                self.refine(dokpdbpath_tmp)
        if do_cm:
            # print("do_cm")
            i = 0
            for dokpdbpath_tmp in outs:
                cm = self.calculate_chain_match(dokpdbpath_tmp)
                for k,v in cm.items():
                    results[i][k] = v
                i += 1

                

        outs_final = []
        i = 0
        for dokpdbpath_tmp in outs:
            outpath = dokpdbpath_tmp.replace(self.outpath_tmp, self.outpath_name)
            prot = af2_features.read_pdb(dokpdbpath_tmp)
            af2_features.save_prot_as_pdb(prot, outpath, self.seq_dict)
            outs_final.append(outpath)
            results[i]['path'] = outpath
            i += 1
        # import pdb;pdb.set_trace()
        if return_df:
            results = pd.DataFrame(results)
            # print(results)
        return results

def do_docking(pdbpath, mappath, bb_pdb_path=None, resolution=3.0):
    
    outpath = pdbpath.replace(".pdb", f"_dt.pdb")
    dd = Docking(pdbpath, mappath, outpath, resolution=resolution, bb_pdb_path=bb_pdb_path)
    outs = dd.run(20, corr_thr=0.4, do_refine=True, do_cm=True, return_df=False)
    return outs

def do_docking_best(pdbpath, mappath, outpath, thr=None, bb_pdb_path=None, resolution=3.0, n=1, do_refine=True):
    
    # outpath = pdbpath.replace(".pdb", f"_.pdb")
    dd = Docking(pdbpath, mappath, outpath, thr=thr, resolution=resolution, bb_pdb_path=bb_pdb_path)
    outs = dd.run_best(n, do_refine=do_refine, return_df=False)
    return outs

def do_cm_calculation(mappath, df, bb_pdb_path=None, resolution=3.0):
    print(df)
    print("do_cm_calculation...")
    dd = Docking(None, mappath, outpath=None, resolution=resolution, bb_pdb_path=bb_pdb_path, do_docking=False)
    with mp.Pool( min(NCPU, os.cpu_count()) ) as pool:
        h_list = []
        for k in  range(0, len(df)):
            tpa = df['path'][k]
            h = pool.apply_async(dd.calculate_chain_match, (tpa,))
            h_list.append([h,k])
            
        for h,i in h_list:
            result = h.get()
            for k,v in result.items():
                # import pdb;pdb.set_trace()
                if k in df.columns:
                    df[k][i] = v
    sorted_indices = (df["cc"] + df["cc_peak"]*2+ df["seq_match"]*1).sort_values(ascending=False).index
    df = df.loc[sorted_indices]
    df = df.reset_index(drop=True)
    print(df)
    return df

if __name__ == "__main__":
    # pdbpath = "/
    ddir = "/data1/jobs/230408_000903_11_0f4536f470b2ff1bc4d017192aa09a2b_3/regions/10"
    pdbpath = f"{ddir}/templ/7y5e_m1:0-177.pdb"
    bb_pdb_path = f"{ddir}/10_pd_tr_ca.pdb"
    mappath = f"{ddir}/10.mrc"
    outpath = f"{ddir}/templ/7y5e_m1:0-177_dtest.pdb"
    resolution = 3.3
    # dd = Docking(pdbpath, mappath, outpath, resolution=resolution, bb_pdb_path=bb_pdb_path)
    # outs = dd.run(20, corr_thr=0.4, do_refine=True, do_cm=True, return_df=True)

    outs = do_docking(pdbpath, mappath, bb_pdb_path=None, resolution=resolution, stage='v0')
    print(outs)
    