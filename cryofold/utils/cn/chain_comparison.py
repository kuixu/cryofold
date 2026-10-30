# Copyright (c) CryoNet Team, and its affiliates. All Rights Reserved

#!/usr/bin/env python
"""
# Author: XU Kui
# Created Time : Tue 29 June 2017 11:14:31 PM CST
# File Name: bbox_to_pdb.py
# Description:
    Input: -p npz file
    Output: PDB file
    decription: using bbox center as the CA with scale and offset to generate PDB file
"""
import pdb
import argparse
import os, sys, subprocess
import glob
import numpy as np
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from cryofold.libs.structure.pdbfile import PdbReader
from cryofold.utils.cn.metrics import AAMetrics, calc_tmscore
# from cryofold.util import TMscore

class HyperParameter:
    def __init__(self):
        self.metric=2
        self.found_dis = 3
        if self.metric == 1 :
            self.avg_dis         = 4.867
            self.std_dis         = 1.093 + 0.1 +1
            self.visit_thres     = 2
            self.metric_string   = "boxcenter"
            self.found_dis       = 3
        elif self.metric == 2 :
            self.metric_string   = "ca"
            self.found_dis       = 3
            self.avg_dis_a       = [
                0,               # 0-order
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
        else:
            self.avg_dis         = 0.11
            self.std_dis         = 0.1
            self.visit_thres     = 0.6
            self.metric_string   = "iou"
            self.found_dis       = 0.11

def generate_chi_cmds(matches, pd_pr):
    if pd_pr.path.endswith(".pdb"):
        outpath = pd_pr.path.replace(".pdb", ".cmd")
    else:
        outpath = pd_pr.path + ".cmd"
    cmds="background solid white;color #8787ceceebeb #1;sel #1:"
    for m in matches:
        chain  = pd_pr.get('chainames')[m].strip()
        resnum = pd_pr.get('resnums')[m]
        cmds += "{}.{},".format(resnum, chain)
    cmds = cmds[:-1]
    cmds += ";select invert sel; color red sel;~sel;"
    print(cmds)

def pdb_to_boxes(pdb_path):
    pr = PdbReader(pdb_path, fast=True)
    return pr.to_boxes()



def chain_comparison(gt_pdb_path, pd_pdb_path, return_results = False, return_matches=False, gt_npz=False):
    assert os.path.exists(gt_pdb_path), "The reference file {} not found".format(gt_pdb_path)
    assert os.path.exists(pd_pdb_path), "The comparison file {} not found".format(pd_pdb_path)
    pdb_name = os.path.basename(pd_pdb_path).replace(".pdb", "")
    gt_pr = PdbReader(gt_pdb_path, fast=True)
    gt_boxes = gt_pr.to_boxes()
    pd_pr = PdbReader(pd_pdb_path, fast=True)
    pd_boxes = pd_pr.to_boxes()
    if gt_boxes is None or pd_boxes is None:
        info = "[AAMetrics] gt: {} selected: {} found: {} seq_match: {} chain_match: {} path_score: {}".format(
            len(gt_boxes), None, 0.0, 0.0, 0.0, 9999.9)
        print(info)
        return 
    # print(tmscore_met)
    hyper_args = HyperParameter()
    met = AAMetrics(gt_boxes, pd_boxes, args=hyper_args)
    initial_path = np.array(list(range(len(pd_boxes))))
    # print(met.compute())
    if return_results:
        return met.compute()
    else:
        tmscore_met = calc_tmscore(gt_pdb_path, pd_pdb_path)
        # tmscore_met = {}
        # tmscore = False
        tmscore = True
        # tmscore_met = TMscore.calc_TM_score(gt_pdb_path, pd_pdb_path, verbose=False)

        met.log(metric=hyper_args.metric_string, thr=hyper_args.found_dis,\
            found=True, seq_match=True, chain_match=True, rmsd=True, path=initial_path, \
            use_pdb=True, tmscore=tmscore, lddt=True, tmscore_met=tmscore_met, \
            title=pdb_name)
        if return_matches:
            generate_chi_cmds(met.matches, pd_pr)
        # pdb.set_trace()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('files', metavar='2', type=str, nargs="+", 
                        help='two files to compare, gt file in front')
    parser.add_argument('-r', '--return_matches', action='store_true')
    parser.add_argument('-a', '--comp_all', action='store_true')
    args = parser.parse_args()
    
    gt_pdb_path = args.files[0]
    pd_pdb_path = args.files[1]
    comp_all = 0
    if len(sys.argv) >= 4:
        comp_all = sys.argv[3]
    if args.comp_all:
        print("pd:", pd_pdb_path)
        for pd_path in glob.glob(pd_pdb_path.replace("path.pdb","path*.pdb")):
            print(pd_path)
            chain_comparison(gt_pdb_path, pd_path)
    else:
        chain_comparison(gt_pdb_path, pd_pdb_path, return_matches=args.return_matches)
