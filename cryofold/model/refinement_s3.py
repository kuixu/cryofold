from pathlib import Path
import os
pkg_dir = os.path.abspath(str(Path(__file__).parent.parent))
import sys
sys.path.insert(0, pkg_dir)
import argparse
import json
import shutil
import torch
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser
import numpy as np
from cryofold.utils.cfrf.fasta_utils import read_fasta
from cryofold.model.cfrf.inference_no_seq import infer
from cryofold.utils.cfrf.fasta_utils import is_valid_fasta_ending
from cryofold.utils.cfrf.misc_utils import filter_useless_warnings, Args
import time



def add_args(parser):
    main_args = parser.add_argument_group(
        "Main arguments",
        description="If you are not very familiar with this software, you can just fill in this part of the parameters. These are also the main parameters of this software."
    )  
    main_args.add_argument( #密度图
        '--map-path',
        "-v",
        "--v",
        help="input cryo-em density map",
        type=str,
        default='/home/jsptgpu/data/share/xukui/6cvm_A/6cvm_A.mrc',# hfy,添加默认参数，避免调试手动输入
        # required=True
    )
    # by hfy: 增加选项：直接使用ca模型，和密度图一起作为输入
    main_args.add_argument( # 碳原子
        "--ca-cif-path",
        "-p",
        help="Path to the C-alpha CIF model file to skip the first stage and directly use for the second stage of reasoning",
        type=str,
        default='/home/jsptgpu/data/share/xukui/6cvm_A/6cvm.A.cif'
    )
    main_args.add_argument(
        "--output-path",
        "-o",
        "--o",
        help="output directory",
        type=str,
        default="output",
    )
    main_args.add_argument(
        "--device",
        "-d",
        "--d",
        help="compute device, pick one of {cpu, cuda:number}. "
             "Default set to use cuda.",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    additional_args = parser.add_argument_group(
        "Additional arguments",
        description="Adjusting these additional parameters can help you build protein models more efficiently."
    )
    additional_args.add_argument(
        "--mask-path",
        "-m",
        "--m",
        help="Providing the mask map corresponding to the original density map can mask out the redundant regions in the original density map.",
        type=str,
    )
    additional_args.add_argument(
        '--fasta-path',
        "-f",
        "--f",
        help="The FASTA database containing all sequences",
        type=str,
    )
    additional_args.add_argument(
        "--crop-length",
        "-n",
        "--n",
        help="500=10G,1024=18G.",
        type=int,
        default=500,
    )
    additional_args.add_argument(
        "--keep-intermediate-results",
        '-k',
        '--k',
        action="store_true",
        help="Keep intermediate results, ie see_alpha_output and CryNet_round_x"
    )

    return parser

def refine(map_path, ca_path, out_path):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    infer_args = Args({
        "repeat_per_residue": 1,
        "mask_threshold": 0,
        "seq_attention_batch_size": 300,
        "crop_length": 500,
        "model_dir": "params/cfrf.pth",
        "aggressive_pruning": True,
        "end_flag": True,
        "output_path": out_path,
        "map_path": map_path,
        "struct": ca_path,
        "device": device,
    })


    # CC = infer(infer_args) 

    iter_num = 20
    best_CC = float('-inf')
    best_round = 0
    last_best_round = 0
    patience = 5
    i = 0  
    while i < iter_num:
        # current_outpath = outpath.replace(f"_rf.cif", f"_rf_{i + 1}.cif")
        # infer_args.struct = ca_path  # 当前的 C-alpha 原子信息文件路径 -> struct
        # infer_args.output_path = out_path  # 此次循环 C-alpha 原子的模型文件输出目录
        if (i + 1) >= iter_num:  # 设计标志位，这下面的推理函数中有用到
            infer_args.end_flag = True
        else:
            infer_args.end_flag = False
        # infer_args.round = i + 1  
        CC = infer(infer_args)  

        if CC > best_CC:
            best_CC = CC
            best_round = i + 1
            last_best_round = i + 1
        elif i - last_best_round >= patience:
            # print(f"Early stopping at round {i + 1} with best CC {best_CC}")
            num_rounds = i + 1
            break
        # current_ca_cif_path = outpath.replace("_rf.cif", f"_rf_{i + 1}.cif")
        i += 1





@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parsed_args = add_args(parser).parse_args()
    filter_useless_warnings()

    device = torch.device(parsed_args.device)
    
    map_path = parsed_args.map_path
    out_path = parsed_args.output_path
    ca_path = parsed_args.ca_cif_path
    start_time = time.time()

    refine(map_path, ca_path, out_path, parsed_args)
    end_time = time.time()-start_time

    print(str(int(end_time))+'s')

    
  



if __name__ == "__main__":
    # main()
    map_path = sys.argv[1]
    ca_path = sys.argv[2]
    out_path = sys.argv[3]

    start_time = time.time()

    refine(map_path, ca_path, out_path,)
    end_time = time.time()-start_time

    print(str(int(end_time))+'s')

