#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""

2022-07-26: Combined code from L.Y.G.(combine U-Net backbone with templates), B.H.L. (sort by length), P.L. (Overlap & Ab chain)
2022-09-19: Pan modified
    1. min_domain_save_cc to define cc value for domain;

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
from cryofold.diff.structure.protein_chain import ProteinChain

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


def parse_refiner_err(dok_file):
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
    if len(err_code.keys())==0:
        err_code['err_code'] = 1
        err_code['err_msg'] = "Error->empty log"


    return err_code

def parse_refiner_log(dok_file, n):
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


def parse_refiner_log_local(dok_file):
    corr_dict = {}
    header = "adpem_opt> Final   CC"
    with open(dok_file) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(header):
                cc = line.replace(header, '').strip()
                corr_dict[0] = float(cc)
           
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


 
class Refinement:

    def __init__(self, pdbpath, mappath, outpath=None, thr=None, bb_pdb_path=None, resolution=4.0, do_refinement=True, islocal=False):
        self.pdbpath = pdbpath
        self.mappath = mappath
        self.outpath = outpath
        self.thr = thr
        self.bb_pdb_path = bb_pdb_path
        self.resolution = resolution
        self.do_refinement = do_refinement
        self.islocal = islocal
        self.refiner_lo = refine_lo
        self.refiner_gl = refine_gl
        self.get_thr()

        


        print(f"pdbpath: {pdbpath}, mappath: {mappath}, outpath: {outpath}")
        self.outpath_tmp = tempfile.NamedTemporaryFile(delete=True).name
        # self.outpath_tmp = self.outpath.replace('.pdb', '.dok')

        if do_refinement:
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

            # self.cmd_dock = self.refiner+' '+self.mappath_tmp+' {} 16 '+str(self.thr)+' '+str(self.resolution)+' -n {} -o {} 2>&1 > {};rm '+self.mappath_tmp+';rm '+self.pdbpath_tmp+';'
            # self.cmd_dock = self.refiner+' '+self.mappath_tmp+' {} 16 '+str(self.thr)+' '+str(self.resolution)+' -n {} -o {} 2>&1 > {};'
            if islocal:
                self.refiner = refine_lo
                self.refiner_cmd = self.refiner+' '+self.mappath_tmp+' '+str(self.thr)+' '+str(self.resolution)+' {} -o {} 2>&1 > {};rm '+self.mappath_tmp+';rm '+self.pdbpath_tmp+';'
                self.refiner_cmd = self.refiner+' '+self.mappath_tmp+' '+str(self.thr)+' '+str(self.resolution)+' {} -o {} 2>&1 > {};'


            else:
                self.refiner = refine_gl
                self.refiner_cmd = self.refiner+' '+self.mappath_tmp+' {} 16 '+str(self.thr)+' '+str(self.resolution)+' -n {} -o {} 2>&1 > {};rm '+self.mappath_tmp+';rm '+self.pdbpath_tmp+';'
                self.refiner_cmd = self.refiner+' '+self.mappath_tmp+' {} 16 '+str(self.thr)+' '+str(self.resolution)+' -n {} -o {} 2>&1 > {};'

            assert os.path.exists(self.refiner), f'{self.refiner} does not exists'
            
            # self.cmd_chix = 'ChimeraX --nogui --cmd "open {}; open '+self.mappath_tmp+';fitmap #1 inMap #2; save {} models #1 relModel #2;exit;" >> /dev/null 2>&1;'
            # self.cmd_refi = 'real_refine_sup.sh '+self.mappath+' {} {} '+str(self.resolution)+' >> {} 2>&1;'
        else:
            self.outpath_name = self.outpath_tmp

        self.cmd_mmcc = 'phenix.map_model_cc '+self.mappath+' {} resolution='+str(self.resolution)+ \
                        ' file_name_prefix='+self.outpath_tmp+' ignore_symmetry_conflicts=True scattering_table=it1992 2>&1 ;'

    def get_thr(self):
        if self.thr is None:
            den_exp = DensityInfo(self.mappath, datatype='numpy')
            self.thr = max(den_exp.mean, 0)

    def get_refine_cmd(self, pdbpath, outpath, logpath, n=1):
        if self.islocal:
            return self.refiner_cmd.format(pdbpath, outpath, logpath)
        else:
            return self.refiner_cmd.format(pdbpath, n, outpath, logpath)
    
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


    def run(self, n=1, do_refine=False, do_cm=False, return_df=False):
        assert n > 0 and n < 100
        
        logpath = self.outpath_tmp.replace(".pdb", ".dok")
        logpath = self.pdbpath.replace(".pdb", ".dok")
        cmd = self.get_refine_cmd(self.pdbpath_tmp, self.outpath_tmp, logpath, n)
        print(cmd)
        output_results_str = subprocess.getoutput(cmd)
        # import pdb;pdb.set_trace()
        err_code  = parse_refiner_err(logpath)
        assert err_code['err_code'] == 0, err_code['err_msg']
        outs = []
        results = []
        if self.islocal:
            corr_dict = parse_refiner_log_local(logpath)
            i = 0
            dok_file0 = self.outpath_tmp + f"00{i+1:0>2d}.pdb"
            dok_file1 = self.outpath_tmp + f"_{i+1:0>2d}.pdb"
            # if corr_dict[i] > corr_thr:
            os.system(f"mv {dok_file0} {dok_file1}")
            outs.append(dok_file1)
            results.append({'id':i, 'path': dok_file1, 'corr': corr_dict[i]})  
            # else:
            #     os.system(f"rm {dok_file0}") 
        else:
            corr_dict = parse_refiner_log(logpath, n)
            for i in range(len(corr_dict)):
                dok_file0 = self.outpath_tmp + f"00{i+1:0>2d}.pdb"
                dok_file1 = self.outpath_tmp + f"_{i+1:0>2d}.pdb"
                # if corr_dict[i] > corr_thr:
                os.system(f"mv {dok_file0} {dok_file1}")
                outs.append(dok_file1)
                results.append({'id':i, 'path': dok_file1, 'corr': corr_dict[i]})  
                # else:
                #     os.system(f"rm {dok_file0}") 
        print(corr_dict)
        
        
        if len(results)==0:
            return [{'id':i, 'path': self.pdbpath, 'corr': 0}]  
        print(results)
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
        if self.islocal:
            assert len(outs) == 1
            outpath = self.outpath
            # outs[0].replace(self.outpath_tmp, self.outpath_name)
            prot = ProteinChain.from_pdb(outs[0])
            prot.to_pdb(outpath)
            os.system(f"rm {outs[0]}") 

            # prot = af2_features.read_pdb(dokpdbpath_tmp)
            # af2_features.save_prot_as_pdb(prot, outpath, self.seq_dict)
            outs_final.append(outpath)
            results[i]['path'] = outpath
            
        else:

            for dokpdbpath_tmp in outs:
                outpath = dokpdbpath_tmp.replace(self.outpath_tmp, self.outpath_name)
                
                # import pdb;pdb.set_trace()
                # prot = ProteinChain.from_pdb_full(dokpdbpath_tmp)
                prot = ProteinChain.from_pdb(dokpdbpath_tmp)
                prot.to_pdb(outpath)
                os.system(f"rm {dokpdbpath_tmp}") 

                # prot = af2_features.read_pdb(dokpdbpath_tmp)
                # af2_features.save_prot_as_pdb(prot, outpath, self.seq_dict)
                outs_final.append(outpath)
                results[i]['path'] = outpath
                i += 1
        # import pdb;pdb.set_trace()
        if return_df:
            results = pd.DataFrame(results)
            # print(results)
        return results
    
    def run_best(self, do_refine=False, do_cm=False, return_df=False):
        return self.run(n=1, do_refine=do_refine, do_cm=do_cm, return_df=return_df)
        
        # logpath = self.outpath_tmp.replace(".pdb", ".dok")
        # cmd = self.get_refine_cmd(self.pdbpath_tmp, self.outpath_tmp, logpath, n)
        # print(cmd)
        # output_results_str = subprocess.getoutput(cmd)
        # # import pdb;pdb.set_trace()
        # corr_dict = parse_refiner_log(logpath, n)
        # # print(corr_dict)
        # outs = []
        # results = []
        # for i in range(len(corr_dict)):
        #     dok_file0 = self.outpath_tmp + f"0{i+1:0>3d}.pdb"
        #     dok_file1 = self.outpath_tmp + f"_{i+1:0>3d}.pdb"
        #     # if corr_dict[i] > corr_thr:
        #     os.system(f"mv {dok_file0} {dok_file1}")
        #     outs.append(dok_file1)
        #     results.append({'id':i, 'path': dok_file1, 'corr': corr_dict[i]})
        #     # print({'id':i, 'path': dok_file1, 'corr': corr_dict[i]})  
        #     # else:
        #     #     os.system(f"rm {dok_file0}")   

        # if do_refine:
        #     # print("do_refine")
        #     for dokpdbpath_tmp in outs:
        #         self.refine(dokpdbpath_tmp)
        # if do_cm:
        #     # print("do_cm")
        #     i = 0
        #     for dokpdbpath_tmp in outs:
        #         cm = self.calculate_chain_match(dokpdbpath_tmp)
        #         for k,v in cm.items():
        #             results[i][k] = v
        #         i += 1

                

        # outs_final = []
        # i = 0
        # for dokpdbpath_tmp in outs:
        #     outpath = dokpdbpath_tmp.replace(self.outpath_tmp, self.outpath_name)
        #     prot = ProteinChain.from_pdb_full(dokpdbpath_tmp)
        #     prot.to_pdb(outpath)
        #     outs_final.append(outpath)
        #     results[i]['path'] = outpath
        #     i += 1
        # # import pdb;pdb.set_trace()
        # if return_df:
        #     results = pd.DataFrame(results)
        #     # print(results)
        # return results

def do_refinement(pdbpath, mappath, outpath=None, bb_pdb_path=None, resolution=3.0, islocal=False, stage='v0', n=20, den_thr=0.):
    if outpath is None:
        outpath = pdbpath.replace(".pdb", f"_dt.pdb")
    dd = Refinement(pdbpath, mappath, outpath, thr=den_thr, resolution=resolution, bb_pdb_path=bb_pdb_path, islocal=islocal)
    outs = dd.run(n, do_refine=False, do_cm=False, return_df=False)
    return outs

def do_refinement_best(pdbpath, mappath, outpath, bb_pdb_path=None, resolution=3.0, islocal=False, do_refine=True, den_thr=0):
    
    # outpath = pdbpath.replace(".pdb", f"_.pdb")
    dd = Refinement(pdbpath, mappath, outpath, thr=den_thr, resolution=resolution, bb_pdb_path=bb_pdb_path, islocal=islocal)
    outs = dd.run_best(do_refine=do_refine, return_df=False)
    return outs

def do_cm_calculation(mappath, df, bb_pdb_path=None, resolution=3.0):
    print(df)
    print("do_cm_calculation...")
    dd = Refinement(None, mappath, outpath=None, resolution=resolution, bb_pdb_path=bb_pdb_path, do_refinement=False)
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
    # ddir = "/data1/jobs/230408_000903_11_0f4536f470b2ff1bc4d017192aa09a2b_3/regions/10"
    # pdbpath = f"{ddir}/templ/7y5e_m1:0-177.pdb"
    # bb_pdb_path = f"{ddir}/10_pd_tr_ca.pdb"
    # mappath = f"{ddir}/10.mrc"
    # outpath = f"{ddir}/templ/7y5e_m1:0-177_dtest.pdb"
    # outpath = f"{ddir}/templ/7y5e_m1:0-177_dtest_local.pdb"
    # resolution = 3.3

    # outs = do_refinement(pdbpath, mappath, bb_pdb_path=None, resolution=resolution, stage='v0')
    # print(outs)

    ddir = "/home/zhangqflab/usr/ADPEM_v2.1_Linux_20180228/adpem_test/"
    pdbpath = f"{ddir}/chainA/adpEM0003.pdb"
    # pdbpath = f"{ddir}/chainA/2c7e_A.pdb"
    # bb_pdb_path = f"{ddir}/10_pd_tr_ca.pdb"
    mappath = f"{ddir}/emd1047.mrc"
    outpath = f"{ddir}/chainA/adpEM0003_local.pdb"
    resolution = 14.9

    # outs = do_refinement(pdbpath, mappath, outpath, resolution=resolution, stage='v0', islocal=True)
    outs = do_refinement(pdbpath, mappath, outpath, resolution=resolution, stage='v0', islocal=False, n=10, corr_thr=0.05)
    print(outs)

    