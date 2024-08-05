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
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict
)
import random
import sys
import time
import torch
import re


def get_model_basename(model_path):
    return os.path.splitext(
                os.path.basename(
                    os.path.normpath(model_path)
                )
            )[0]


def make_output_directory(output_dir, model_name, multiple_model_mode):
    if multiple_model_mode:
        prediction_dir = os.path.join(output_dir, "predictions", model_name)
    else:
        prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    return prediction_dir


def count_models_to_evaluate(cryofold_checkpoint_path):
    model_count = 0
    model_count += len(cryofold_checkpoint_path.split(","))
    return model_count


def load_models_from_command_line(args, config, CryoFold):
    # Create the output directory

    multiple_model_mode = count_models_to_evaluate(args.cryofold_checkpoint_path) > 1
    if multiple_model_mode:
        logger.info(f"evaluating multiple models")

    if args.cryofold_checkpoint_path:
        for path in args.cryofold_checkpoint_path.split(","):
            model = CryoFold(config)
            # model = model.eval()
            checkpoint_basename = get_model_basename(path)
            # import pdb;pdb.set_trace()
            if os.path.isdir(path):
                # A DeepSpeed checkpoint
                logger.info(
                    f"Loading CryoFold parameters from DeepSpeed checkpointing..."
                )
                ckpt_path = os.path.join(
                    path,
                    checkpoint_basename + ".pt",
                )

                if not os.path.isfile(ckpt_path):
                    convert_zero_checkpoint_to_fp32_state_dict(
                        path,
                        ckpt_path,
                    )
                d = torch.load(ckpt_path)
                # model.load_state_dict(d["ema"]["params"], strict=False)
                model.load_state_dict(d["ema"]["params"])
            else:
                # import pdb;pdb.set_trace()

                ckpt_path = path
                d = torch.load(ckpt_path, map_location='cpu')
                # print(d.keys()) #model.state_dict()['
                if "ema" in d:
                    # The public weights have had this done to them already
                    d = d["ema"]["params"]
                if "module.model." in list(d.keys())[0]:
                    d = {k[len("module.model."):]:v for k,v in d.items()}
                # model.load_state_dict(d)
                model.load_state_dict(d, strict=False)
            
            model = model.to(args.model_device)
            logger.info(
                f"Loaded CryoFold parameters at {path}..."
            )
            output_directory = make_output_directory(args.output_dir, checkpoint_basename, multiple_model_mode)
            yield model, output_directory
    
    if not args.cryofold_checkpoint_path:
        raise ValueError(
            "At least one of cryofold_checkpoint_path must "
            "be specified."
        )

