#!/usr/bin/env python

import os
import argparse
import tempfile
import shutil
import numpy as np
import random

import torch
import torch.backends.cudnn as cudnn
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from cryofold.diff.structure.protein_chain import ProteinChain
from cryofold.model.sm import CF
from cryofold.model.api import CFProtein, GenerationConfig
from cryofold.model.packsc import pack_sidechain

# WINDOW_LENGTH = 256
WINDOW_LENGTH = 768
EXTEND_NUM = 100

WINDOW_LENGTH = 384
EXTEND_NUM = 48

def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    # Set CuDNN to use deterministic algorithms
    cudnn.deterministic = True
    cudnn.benchmark = False
set_seed(999,0)


def align_indexes(motif_inds, length, NUM=32):#16):
    O = [i for i in range(length)]
    # left = list(set(O) - set(motif_inds))
    left = sorted(set(O) - set(motif_inds))
    
    groups = []   # need to align indexes
    current_group = []

    for number in left:
        if not current_group or (current_group and current_group[-1] + 1 != number):
            if current_group:  
                groups.append(current_group)
            current_group = [number]  
        else:
            current_group.append(number) 

    if current_group:
        groups.append(current_group)  
 
    all_align_indexes = [] #all align indexes
    # mask_indexes = np.array(groups, dtype=object) #mask indexes
    mask_indexes = groups #mask indexes
    # breakpoint()
    for mask_index in mask_indexes:
        left_align_indexes = [] #left align indexes
        right_align_indexes = [] #right align indexes
        # left_number = mask_index[0]-NUM #left number
        left_number = mask_index[0]-1 #left number
        # break_flag = False
        
        # while len(left_align_indexes) < NUM:
        #     print("len left_align_indexes",len(left_align_indexes))
        #     if break_flag:
        #         break
        #     for i in range(len(motif_inds)):
        #         if left_number - i in mask_indexes:
        #             left_align_indexes.append(left_number - i)
        #         if left_number - i < 0:
        #             break_flag = True
        #             break
        for i in range(NUM):
            if left_number - i < 0:
                break
            elif left_number - i in motif_inds:
                left_align_indexes.append(left_number - i)
            else:
                pass
        # right_number = mask_index[-1] + 1 #right number
        # break_flag = False
        # while len(right_align_indexes) < NUM:
        #     print("len right_align_indexes",len(right_align_indexes))
        #     if break_flag:
        #         break
        #     for i in range(len(motif_inds)):
        #         print("right_number + i",right_number + i)
        #         print("mask_indexes",mask_indexes)
        #         if right_number + i in mask_indexes:
        #             right_align_indexes.append(right_number + i)
        #         if right_number + i > length - 1:
        #             break_flag = True
        #             break
        right_number = mask_index[-1] + 1 #right number
        for i in range(NUM):
            if right_number + i >= length:
                break
            elif right_number + i in motif_inds:
                right_align_indexes.append(right_number + i)
            else:
                pass
        align_indexes = left_align_indexes[::-1] + right_align_indexes #all align indexes
        all_align_indexes.append(align_indexes)
        # if len(align_indexes)!=0:
        #     breakpoint()
        #     all_align_indexes.append(align_indexes)
    
    return all_align_indexes, groups


def remove_model_lines(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    with open(output_file_path, 'w') as file:
        for line in lines:
            if not line.startswith('MODEL'):
                file.write(line)


def impute_interval(WINDOW_LENGTH, miss_inds,length, EXTEND_NUM=20):

    operate_list = []

    groups = []   # need to align indexes
    current_group = []

    for number in miss_inds:
        if not current_group or (current_group and current_group[-1] + 1 != number):
            if current_group:  
                groups.append(current_group)
            current_group = [number]  
        else:
            current_group.append(number)  

    if current_group:
        groups.append(current_group)
    
    def get_interval(first_index, last_index, EXTEND_NUM=20):
        curr_interval = np.array([first_index-EXTEND_NUM , last_index+EXTEND_NUM])
        curr_interval = np.clip(curr_interval,0,length).astype(int)
        return curr_interval

    first_index = None
    last_index = None
    for group in groups:
        if first_index is None:
            first_index = group[0]
            last_index = group[-1]
            if last_index - first_index > WINDOW_LENGTH:
                operate_list.append(get_interval(first_index, last_index))
                first_index = None
                last_index = None
        else:
            if group[-1] - first_index > WINDOW_LENGTH:
                
                operate_list.append(get_interval(first_index, last_index))
                first_index = group[0]
                last_index = group[-1]
            else:
                last_index = group[-1]
    if first_index is not None:
        curr_interval = np.array([first_index-EXTEND_NUM , last_index+EXTEND_NUM])
        curr_interval = np.clip(curr_interval,0,length).astype(int)
        operate_list.append(curr_interval)

    return operate_list


def increment_list_elements(lst,num):
    if not isinstance(lst, list) and not isinstance(lst, tuple):
        return lst + num
    return [increment_list_elements(item,num) for item in lst]



def build_backbone_imp(template_path, out_path0, out_path1):
    if template_path.lower().endswith(('.pdb')):
        # print("loading pdb ...")
        # remove_model_lines(template_path, template_path)
        template_chain = ProteinChain.from_pdb_full(template_path)
        # print("pdb loaded")
    else:
        # print("loading cif ...")
        template_chain = ProteinChain.from_cif(template_path)
        # print("cif loaded")
    # template_chain.residue_index
    # with template_chain
    basename = os.path.basename(template_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_fix_path = os.path.join(tmp_dir, basename.replace(".pdb", "_fix.pdb"))
        tmp_imp_path = os.path.join(tmp_dir, basename.replace(".pdb", "_imp.pdb"))

        # need to support empty/gap region, make pack_sidechain respect seqres and residue id
        # template_chain = pack_sidechain(template_path, out_path=tmp_fix_path)
        # template_chain = ProteinChain.from_pdb_full(tmp_fix_path)


        template_chain = ProteinChain.from_pdb_full(template_path)

        
        # fix sc
        atom_idx =np.array([0,1,2,4])
        # atom_idx =np.array([1])
        # mask = template_chain.atom37_mask[:,:3].all(axis=1)
        mask = template_chain.atom37_mask[:,atom_idx].all(axis=1)
        miss_inds = np.where(1-mask)[0]

        # print("miss_inds",miss_inds)

        # breakpoint()
        ol = impute_interval(WINDOW_LENGTH,miss_inds,len(template_chain.sequence), EXTEND_NUM)

        if template_chain.sequence == template_chain.sequence_fragment:
            # print("not need esm.")
            template_chain.to_pdb(out_path0)
        else:
            # print("esm3 imp...")

            model =  CF.from_pretrained("cf_v1.0", device=torch.device("cuda"))
            for interval in ol:
                # print("interval:",interval)
                # template_chain = ProteinChain.from_pdb_full(args.template_path)[interval[0]:interval[1]]
                # sequence_prompt = template_chain.sequence
                sequence_prompt = template_chain.sequence[interval[0]:interval[1]]
                # coordinates_prompt = template_chain.atom37_positions
                coordinates_prompt = template_chain.atom37_positions[interval[0]:interval[1],...]

                protein_prompt = CFProtein(sequence=sequence_prompt, coordinates=torch.tensor(coordinates_prompt))
                # breakpoint()
                structure_prediction_config = GenerationConfig(
                    track="structure", # We want CF3 to generate tokens for the structure track
                    num_steps=len(sequence_prompt) // 8,
                    temperature=0.7,
                )

                # try:
                #     structure_prediction,_ = model.generate(protein_prompt, structure_prediction_config)
                # except :
                #     breakpoint()
                structure_prediction  = model.generate(protein_prompt, structure_prediction_config)
            
                imputed_chain = structure_prediction.to_protein_chain()
                imputed_chain.to_pdb(out_path1)

                do_replace = True
                if do_replace:
                    mask = template_chain.atom37_mask[:,atom_idx].all(axis=1)
                    motif_inds = np.where(mask)[0]

                    
                    motif_inds = np.unique(np.clip(motif_inds, interval[0], interval[1]-1).astype(int))
                    motif_inds = motif_inds - np.array(interval[0])
                    # print("motif_inds",motif_inds)

                    indexes = align_indexes(motif_inds,length=len(sequence_prompt))
                    # breakpoint()
                    indexes = increment_list_elements(indexes,interval[0])
                    # indexes = [np.array(index)+np.array(interval[0]) for index in indexes]
                    for i in range(len(indexes[0])):
                        ind_rep = indexes[1][i]
                        ind_ali = indexes[0][i]
                        if len(ind_rep)==0 or len(ind_ali)==0:
                            breakpoint()
                            continue
                        # breakpoint()
                        # align_imputed_chain = imputed_chain.align(template_chain, mobile_inds=indexes[0][i], target_inds=indexes[0][i])
                        nonan_inds = np.where(~np.isnan(template_chain[ind_ali].atom37_positions[:,1,0]))[0]
                        target_inds = np.array(ind_ali)[nonan_inds]
                        mobile_inds = (ind_ali-np.array(interval[0]))[nonan_inds ]

                        try:
                            align_imputed_chain = imputed_chain.align(template_chain, mobile_inds=mobile_inds, target_inds=target_inds)
                        except:
                            print("align error")
                            breakpoint()
                        imputed_atom37_pos = align_imputed_chain.atom37_positions[ind_rep -np.array(interval[0]),:,:]
                        # print("imputed_atom37_pos: ",imputed_atom37_pos[:,1,0])
                        template_chain.atom37_positions[ind_rep,:,:] = imputed_atom37_pos
                    # breakpoint()
                    allindex = set(range(interval[0],interval[1]))
                    replace_index = list(allindex - set(motif_inds + np.array(interval[0])))
                    template_chain.atom37_mask[replace_index,:] = imputed_chain.atom37_mask[replace_index-np.array(interval[0]),:]
                    
            nan_inds = np.where(np.isnan(template_chain.atom37_positions[:,0,0]))[0]
            nan_inds = np.where(np.isnan(template_chain.atom37_positions[:,4,0]))[0]
            # print(f"nan_inds: {len(nan_inds)}",nan_inds)

            # breakpoint()
            # template_chain.to_pdb(tmp_imp_path)
            template_chain.to_pdb(out_path0)
        # pack_sidechain(tmp_imp_path, out_path=out_path0)
        # breakpoint()

    # print(f"saved at {args.out_path0}")

    # cat_pdb_files(tmpdirname, args.out_path0)

    # shutil.rmtree(tmpdirname)

if __name__== "__main__":

    parser = argparse.ArgumentParser(description='test backbone')

    parser.add_argument('--template_path','-i', type=str, default='/13175_A:1-401.pdb', help='input')
    parser.add_argument('--out_path0','-o', type=str, default='13175_test.pdb', help='output')
    parser.add_argument('--out_path1','-p', type=str, default='13175_test.pdb', help='output')
    
    
    args = parser.parse_args()

    build_backbone_imp(args.template_path, args.out_path0, args.out_path1)
