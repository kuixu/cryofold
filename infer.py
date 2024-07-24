# Copyright 2023 The CryoFold team
# Copyright 2021 AlQuraishi Laboratory
# 
# Licensed under the MIT License.

import argparse
from copy import deepcopy
from datetime import date
import logging
import math
import numpy as np
import os,gzip

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

import pickle
import random
import sys
import time
import torch
import re

torch.set_grad_enabled(False)
from cryofold.model.model import CryoFold
from cryofold.config import model_config, NUM_RES
from cryofold.data import templates, feature_pipeline, data_pipeline
from cryofold.np import residue_constants, protein
from cryofold.data import data_loader 
from cryofold.utils.load_model import load_models_from_command_line
from cryofold.utils.protein_utils import save_dict_as_pdb

from cryofold.utils.tensor_utils import (
    tensor_tree_map,
)
from scripts.utils import add_data_args


def run_model(model, batch, tag, args):
    with torch.no_grad(): 
        # Disable templates if there aren't any in the batch
        model.config.template.enabled = model.config.template.enabled and any([
            "template_" in k for k in batch
        ])

        logger.info(f"Running inference for {tag}...")
        t = time.perf_counter()
        out = model(batch)
        inference_time = time.perf_counter() - t
        logger.info(f"Inference time: {inference_time}")
   
    return out




def main(args):
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    config = model_config(args.config_preset)
    config.data.cryoem.debug = args.debug
    config.data.cryoem.axis = args.axis
    config.data.cryoem.dataset = args.dataset
    config.data.cryoem.resolution = args.resolution
    config.data.cryoem.inference = True 
    config.data.cryoem.synmap = not args.use_mrc 
    config.data.cryoem.map_path = args.map_path
    
    

    
    if(args.trace_model):
        if(not config.data.predict.fixed_size):
            raise ValueError(
                "Tracing requires that fixed_size mode be enabled in the config"
            )

    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)
    

    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    out_name = args.map.replace(".mrc", "")
    # TODO data_loader is in preparing
    try:
        feature_dict = data_loader.prepare_data(args.map, args.seq)
    except:
        print("TODO data_loader is in preparing.")

    output_name = f'{out_name}_{args.config_preset}'

    # print(feature_dict.keys())
    processed_feature_dict = feature_processor.process_features(
        feature_dict, mode='predict',
    )

    processed_feature_dict = data_loader.add_cryoem(processed_feature_dict, config.data.cryoem, args.pdb_chain)
    # import pdb;pdb.set_trace()
    print("Density: ", processed_feature_dict['cryoem_density'].shape)
    print("AminoAcids: ", processed_feature_dict['aatype'].shape)
    processed_feature_dict = {
        k:torch.as_tensor(v, device=args.model_device) 
        for k,v in processed_feature_dict.items()
    }

    for model, output_directory in load_models_from_command_line(args, config, CryoFold): 
        out = run_model(model, processed_feature_dict, out_name, args)
        # Toss out the recycling dimensions --- we don't need them anymore
        processed_feature_dict = tensor_tree_map(
            lambda x: np.array(x[..., -1].cpu()), 
            processed_feature_dict
        )
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

        save_dict_as_pdb(processed_feature_dict, out, output_directory, output_name)

        
        
        if args.save_outputs:
            output_dict_path = os.path.join(
                output_directory, f'{output_name}_output_dict.pkl'
            )
            with open(output_dict_path, "wb") as fp:
                pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Model output written to {output_dict_path}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq", type=str, default="",
        help="""Name of the sequence file""",
    )
    parser.add_argument(
        "--map", type=str, default="",
        help="""Name of the cryo-EM/ET density map file""",
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--gpu_device", type=str, default="cpu",
        help="""ID of the GPU device on which to run the model. For running on CPU, set "cpu"."""
    )
    parser.add_argument(
        "--config_preset", type=str, default="cryofold_v1",
        help="""Name of a model config preset defined in cryofold/config.py"""
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="""Path to CryoFold checkpoint. Can be either a DeepSpeed 
             checkpoint directory or a .pt file"""
    )
    parser.add_argument(
        "--cpus", type=int, default=8,
        help="""Number of CPUs to run MSA pipeline"""
    )
    parser.add_argument(
        "--random_seed", type=str, default=None
    )
    add_data_args(parser)
    args = parser.parse_args()


    if(args.gpu_device == "cpu" and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. For running on GPU, please 
            specify --gpu_device."""
        )

    main(args)
