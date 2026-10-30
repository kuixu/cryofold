"""
MIT License

Copyright (c) 2022 Kiarash Jamali

This file is modified from: [https://github.com/3dem/model-angelo/blob/main/model_angelo/gnn/inference.py].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
"""
import os.path
import sys
from collections import namedtuple
import torch
import tqdm
import numpy as np
from scipy.spatial import cKDTree

from cryofold.utils.cfrf.mrc_tools import load_map,make_model_grid
from cryofold.utils.cfrf.save_pdb_utils import final_results_to_cif ,final_results_to_protein_complex
from cryofold.utils.cfrf.fasta_utils import is_valid_fasta_ending
from cryofold.utils.cfrf.protein import Protein, get_protein_empty_except, load_protein_from_prot, get_protein_from_file_path
from cryofold.utils.cfrf.affine_utils import init_random_affine_from_translation, get_affine_translation, get_affine, get_affine_rot
from cryofold.utils.cfrf.pdb_untils import load_cas_from_structure
from cryofold.model.cfrf.CryFolder_no_seq import CryFolder
from  cryofold.utils.cfrf.density_v2 import DensityInfo
MRCObject = namedtuple("MRCObject", ["grid", "voxel_size", "global_origin"])

def get_module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device

def init_protein_from_see_alpha(see_alpha_file: str) -> Protein:
    ca_locations = torch.from_numpy(load_cas_from_structure(see_alpha_file)).float()
    rigidgroups_gt_frames = np.zeros((len(ca_locations), 1, 3, 4), dtype=np.float32)
    rigidgroups_gt_frames[:, 0] = init_random_affine_from_translation(
        ca_locations
    ).numpy()
    rigidgroups_gt_exists = np.ones((len(ca_locations), 1), dtype=np.float32)
    prot = get_protein_from_file_path(see_alpha_file)
    # breakpoint()
    return get_protein_empty_except(
        rigidgroups_gt_frames=rigidgroups_gt_frames,
        rigidgroups_gt_exists=rigidgroups_gt_exists,
        aatype=prot.aatype,
        residue_index=prot.residue_index,
        chain_index=prot.chain_index,
    )
def argmin_random(count_tensor: torch.Tensor):
    rand_idxs = torch.randperm(len(count_tensor))
    corr_idxs = torch.arange(len(count_tensor))[rand_idxs]
    random_argmin = count_tensor[rand_idxs].argmin()
    original_argmin = corr_idxs[random_argmin]
    return original_argmin
def update_protein_gt_frames(
    protein: Protein, update_indices: np.ndarray, update_affines: np.ndarray
) -> Protein:
    protein.rigidgroups_gt_frames[update_indices][:, 0] = update_affines
    return protein

def get_inference_data(protein, grid_data, idx, crop_length=300): 
    grid = ((grid_data.grid - np.mean(grid_data.grid)) / np.std(grid_data.grid)).astype( # 归一化
        np.float32
    )
    backbone_frames = protein.rigidgroups_gt_frames[:, 0]  # (num_res, 3, 4) 
    ca_positions = get_affine_translation(backbone_frames)
    picked_indices = np.arange(len(ca_positions), dtype=int)
    # 当蛋白质的残基数量 len(ca_positions) 大于 crop_length 时，使用 idx 作为随机起始残基索引 random_res_index。
    if len(ca_positions) > crop_length:
        random_res_index = idx
        # 然后创建一个 cKDTree 对象 kd，用于对 ca_positions 进行快速空间搜索
        kd = cKDTree(ca_positions)
         #通过 kd.query 方法，找到距离 ca_positions[random_res_index] 最近的 crop_length 个点的索引，更新 picked_indices。这一步实现了对残基的截取，使得最终用于推理的数据量符合 crop_length 的限制。
        _, picked_indices = kd.query(ca_positions[random_res_index], k=crop_length)

    return {
        "affines": torch.from_numpy(backbone_frames[picked_indices]),
        "cryo_grids": torch.from_numpy(grid[None]),  # Add channel dim
        "cryo_global_origins": torch.from_numpy(
            grid_data.global_origin.astype(np.float32)
        ),
        "cryo_voxel_sizes": torch.from_numpy(grid_data.voxel_size.astype(np.float32)),
        "indices": torch.from_numpy(picked_indices), # 选取的残基索引，转换为张量 
        "num_nodes": len(picked_indices), # 记录了当前用于推理的数据中包含的氨基酸数量（也就是节点数量，从命名和数据结构角度推测，把氨基酸看作节点进行处理）
    }
@torch.no_grad()
def run_inference_on_data(
    module,
    data,
    run_iters: int = 3,
):
    device = get_module_device(module)

    affines = data["affines"].to(device)
    result = module(
        positions=get_affine_translation(affines),
        batch=None,
        cryo_grids=[data["cryo_grids"].to(device)],
        cryo_global_origins=[data["cryo_global_origins"].to(device)],
        cryo_voxel_sizes=[data["cryo_voxel_sizes"].to(device)],
        init_affine=affines,
        run_iters=run_iters,
    )
    result.to("cpu")
    return result


def init_empty_collate_results(num_residues, device="cpu"):
    result = {}
    result["counts"] = torch.zeros(num_residues, device=device) # 氨基酸数量
    result["pred_positions"] = torch.zeros(num_residues, 3, device=device) # 位置
    result["pred_affines"] = torch.zeros(num_residues, 3, 4, device=device) # 每个氨基酸的仿射变换信息
    result["pred_torsions"] = torch.zeros(num_residues, 83, 2, device=device) # 氨基酸的扭转角度？
    result["aa_logits"] = torch.zeros(num_residues, 20, device=device) # 每个氨基酸的类型
    result["local_confidence"] = torch.zeros(num_residues, device=device)
    result["existence_mask"] = torch.zeros(num_residues, device=device)
    result["edge_logits"] = torch.zeros(
        num_residues, 20, 2,device=device
    )
    result["edge_index"] = torch.zeros(
        num_residues, 20,dtype=torch.long, device=device
    )
    return result

# 将神经网络的预测结果整合到已有的累积结果中，并根据这些结果更新蛋白质的相关信息
def collate_nn_results(
    # results：当前神经网络预测得到的结果，也是一个字典，包含了预测的位置、扭转角、氨基酸 logits 等信息。
    # indices：一个索引数组，用于指定在 collated_results 中哪些位置需要更新, 这里的索引是从 0 开始的,其实本质就是上一步神经网络选取处理的那些氨基酸。
    # protein：一个蛋白质对象，可能包含蛋白质的结构信息，函数会根据整合后的结果更新该对象。
    # end_flag：布尔标志，默认为 False，用于控制不同的逻辑分支，影响结果的整合方式。
    # crop_length：整数值，默认为 300，用于计算 num_pred_residues。
    # repeat_num：整数，默认为 3，用于控制重复计数的逻辑，决定某些结果的更新次数。
    collated_results, results, indices, protein,end_flag=False,crop_length=300,repeat_num:int=3
):
    num_pred_residues = crop_length//3
    if end_flag:
        repeat_logits = (collated_results["counts"][indices[:num_pred_residues]] > -1) # 无论计数是多少，对应的 repeat_logits 位置就为 True 对应aggresive_repeat
    else:
        repeat_logits = (collated_results["counts"][indices[:num_pred_residues]]<repeat_num) # 只有计数小于 repeat_num 的位置会被考虑，从而限制了某些结果的更新次数，防止某个位置被过度更新
    # 对满足 repeat_logits 条件的索引位置，将 collated_results["counts"] 中对应的值加 1，用于记录每个位置被更新的次数
    collated_results["counts"][indices[:num_pred_residues][repeat_logits]] += 1 
    # 对满足 repeat_logits 条件的索引位置，将 collated_results["pred_positions"] 中对应的值加上当前预测的位置，用于记录每个位置的预测位置信息
    collated_results["pred_positions"][indices[:num_pred_residues][repeat_logits]] += results[
        "pred_positions"
    ][-1][:num_pred_residues][repeat_logits]
    # 对满足 repeat_logits 条件的索引位置，将 collated_results["pred_torsions"] 中对应的值加上当前预测的扭转角度，用于记录每个位置的预测扭转角度信息
    collated_results["pred_torsions"][indices[:num_pred_residues][repeat_logits]] += torch.nn.functional.normalize(
        results["pred_torsions"][:num_pred_residues][repeat_logits], p=2, dim=-1
    )
    #     取平均值
    curr_pos_avg = (
        collated_results["pred_positions"][indices[:num_pred_residues][repeat_logits]]
        / collated_results["counts"][indices[:num_pred_residues][repeat_logits]][..., None]
    )
    # 取平均值，更新 collated_results["pred_affines"] 中对应的值，用于记录每个位置的预测仿射变换信息
    collated_results["pred_affines"][indices[:num_pred_residues][repeat_logits]] = get_affine(
        get_affine_rot(results["pred_affines"][-1][:num_pred_residues][repeat_logits]).cpu(),
        curr_pos_avg
    )
    # 对满足 repeat_logits 条件的索引位置，将 collated_results["aa_logits"] 中对应的值加上当前预测的氨基酸类型 logits，用于记录每个位置的预测氨基酸类型信息
    collated_results["aa_logits"][indices[:num_pred_residues][repeat_logits]] += results[
        "cryo_aa_logits"
    ][-1][:num_pred_residues][repeat_logits]
    # 对满足 repeat_logits 条件的索引位置，将 collated_results["local_confidence"] 中对应的值更新为当前预测的局部置信度，用于记录每个位置的预测局部置信度信息
    collated_results["local_confidence"][indices[:num_pred_residues][repeat_logits]] = results[
        "local_confidence_score"
    ][-1][:num_pred_residues][repeat_logits]

    collated_results["existence_mask"][indices[:num_pred_residues][repeat_logits]] = results[
        "pred_existence_mask"
    ][-1][:num_pred_residues][repeat_logits]
    edge_index = results["cryo_edges"][-1][0].reshape(crop_length,20)
    collated_results["edge_index"][indices[:num_pred_residues][repeat_logits]] = indices[edge_index][:num_pred_residues][repeat_logits]
    collated_results["edge_logits"][indices[:num_pred_residues][repeat_logits]] = results[
        "cryo_edge_logits"
    ][-1][:num_pred_residues][repeat_logits].softmax(dim=2)[...,1:]
    protein = update_protein_gt_frames(
        protein,
        indices[:num_pred_residues].numpy(),
        collated_results["pred_affines"][indices[:num_pred_residues]].numpy(),
    )
    return collated_results, protein

def get_final_nn_results(collated_results):
    final_results = {}

    final_results["pred_positions"] = (
        collated_results["pred_positions"] / collated_results["counts"][..., None]
    )
    final_results["pred_torsions"] = (
        collated_results["pred_torsions"] / collated_results["counts"][..., None, None]
    )
    final_results["pred_affines"] = get_affine(
        get_affine_rot(collated_results["pred_affines"]),
        final_results["pred_positions"],
    )
    final_results["aa_logits"] = (
        collated_results["aa_logits"] / collated_results["counts"][..., None]
    )

    final_results["local_confidence"] = collated_results["local_confidence"]
    final_results["existence_mask"] = collated_results["existence_mask"]

    final_results["raw_aa_entropy"] = (
        final_results["aa_logits"].softmax(dim=-1).log().sum(dim=-1)
    )
    final_results["normalized_aa_entropy"] = final_results["raw_aa_entropy"].add(
        -final_results["raw_aa_entropy"].min()
    )
    final_results["normalized_aa_entropy"] = final_results["normalized_aa_entropy"].div(
        final_results["normalized_aa_entropy"].max()
    )

    final_results["edge_logits"] = collated_results["edge_logits"]
    final_results["edge_index"] = collated_results["edge_index"]

    return dict([(k, v.numpy()) for (k, v) in final_results.items()])


def infer(args):
    device = torch.device(args.device)
    state_dict = torch.load(args.model_dir, map_location=torch.device('cpu'), weights_only=False)
    # 构建模型，并且设置模型为推理阶段
    module = CryFolder(256,num_layers_former=21,num_layers_ipa=7,attention_heads=8)
    module.load_state_dict(state_dict)
    module.to(device)
    module.eval()
    protein = None # 用来接收碳原子模型结构，赋值给protein这个变量
    if args.struct.endswith("cif") or args.struct.endswith("pdb"):
        protein = init_protein_from_see_alpha(args.struct) # 从给定的文件（see_alpha_file）中提取蛋白质结构里 Cα 原子位置信息
    else:
        raise "Failed in refinement stage, no structure model file provided"
    # breakpoint()
    aatype = protein.aatype
    args.crop_length = min(args.crop_length, len(protein.aatype)-1 )
    if protein is None:
        raise RuntimeError(f"File {args.struct} is not a supported file format.")
    grid_data = None # 用来接收密度图对象
    if args.map_path.endswith("map") or args.map_path.endswith("mrc"):
        density_info = DensityInfo(mrc_path=args.map_path)

        grid_torch = density_info.density
        grid = grid_torch.numpy().T
        voxel_size = density_info.voxel_size
        voxel_size=np.array([voxel_size,voxel_size,voxel_size])
        global_origin = density_info.offset
        global_origin = global_origin.numpy()

        # return (grid_np, voxel_size, global_origin_np)

        # grid, voxel_size, global_origin = load_map(args.map_path, multiply_global_origin=True)
        grid, voxel_size, global_origin = make_model_grid(
            grid,
            voxel_size,
            global_origin,
            target_voxel_size=1.1,
        )
        grid_data = MRCObject(grid,voxel_size,global_origin)
    if grid_data is None:
        raise RuntimeError(
            f"Grid volume file {args.map_path} is not a cryo_em density map file format."
        )
    num_res = len(protein.rigidgroups_gt_frames) 
    # print("input res:",num_res)
    collated_results = init_empty_collate_results(
        num_res,
        device="cpu",
    )

    residues_left = num_res 
    total_steps = num_res * args.repeat_per_residue 
    steps_left_last = total_steps

    aggressive_repeat = True
    # performing inference on each residue of the protein structure, updating the results iteratively until all residues have been processed
    while residues_left > 0:
        idx = argmin_random(collated_results["counts"]) # counts是氨基酸数量，Random Selection of Residue
        data = get_inference_data(protein, grid_data, idx, crop_length=args.crop_length) # 每次处理crop_length个残基

        results = run_inference_on_data( # 在神经网络中进行推理
            module, data
        )
        collated_results, protein = collate_nn_results(
            collated_results,
            results,
            data["indices"],
            protein,
            end_flag=True if aggressive_repeat else args.end_flag,
            crop_length=args.crop_length,
            repeat_num=args.repeat_per_residue
        )
        residues_left = (
            num_res
            - torch.sum(collated_results["counts"] > args.repeat_per_residue - 1).item() # 如果该氨基酸已经被更新3次，则证明该氨基酸已经被处理完毕，从而减少了重复处理的次数
        )
        steps_left = (
            total_steps
            - torch.sum(
                collated_results["counts"].clip(0, args.repeat_per_residue)  # 对 collated_results["counts"] 张量进行裁剪操作。它将张量中小于 0 的值设为 0，大于 args.repeat_per_residue 的值设为 args.repeat_per_residue。这一步是为了确保在计算剩余步骤数时，每个残基的更新次数不会超过最大允许次数 args.repeat_per_residue。
            ).item()
        )
        # pbar.update(n=int(steps_left_last - steps_left))
        steps_left_last = steps_left
    # pbar.close()

    # print("input:",torch.sum(collated_results["counts"]))

    final_results = get_final_nn_results(collated_results)
    # output_path = os.path.join(args.output_dir, "model_net.cif")
    output_path = args.output_path
    # final_results_to_cif(
    #     final_results,
    #     output_path,
    #     sequences=None,
    #     aatype=aatype,
    #     verbose=True,
    #     aggressive_pruning=args.aggressive_pruning,
    #     mask_threshold=args.mask_threshold,
    #     end_flag=args.end_flag,
    #     protein=protein,
    # )
    mol_den = final_results_to_protein_complex(
        final_results,
        output_path,
        sequences=None,
        aatype=aatype,
        verbose=True,
        aggressive_pruning=args.aggressive_pruning,
        mask_threshold=args.mask_threshold,
        end_flag=args.end_flag,
        protein=protein,
    )

    t_ov, m_ov = density_info.overlap_right(mol_den)
    ## CC loss
    t_ov_1 = t_ov - t_ov.mean()
    m_ov_1 = m_ov - m_ov.mean()
    CC = (t_ov_1 * m_ov_1).sum() / ((t_ov_1**2).sum().sqrt() * (m_ov_1**2).sum().sqrt())
    return CC


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map-path", "--m", required=True, help="The path to the input map")
    parser.add_argument(
        "--struct", "--s", required=True, help="The path to the structure file"
    )
    parser.add_argument("--model-dir", required=True, help="Where the model at")
    parser.add_argument("--output-dir", default=".", help="Where to save the results")
    parser.add_argument("--device", default="cpu", help="Which device to run on")
    parser.add_argument(
        "--crop-length", type=int, default=400, help="How many points per batch"
    )
    parser.add_argument(
        "--repeat-per-residue",
        default=3,
        type=int,
        help="How many times to repeat per residue",
    )
    parser.add_argument(
        "--aggressive-pruning",
        action="store_true",
        help="Only build parts of the model that have a good match with the sequence. "
        + "Will lower recall, but quality of build is higher",
    )
    args = parser.parse_args()
    infer(args)
