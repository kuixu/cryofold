import copy
from typing import Mapping, Tuple, List, Optional, Dict, Sequence

import ml_collections
import numpy as np
import torch

from cryofold.data import input_pipeline


FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


def np_to_tensor_dict(
    np_example: Mapping[str, np.ndarray],
    features: Sequence[str],
) -> TensorDict:
    """Creates dict of tensors from a dict of NumPy arrays.

    Args:
        np_example: A dict of NumPy feature arrays.
        features: A list of strings of feature names to be returned in the dataset.

    Returns:
        A dictionary of features mapping feature names to features. Only the given
        features are returned, all other ones are filtered out.
    """ 
    print("np_example",np_example.keys())
    print("features",features)
    # for k, v in np_example.items():
    #     print(k,"", v)
    # tensor_dict = {
    #     k: torch.tensor(v) for k, v in np_example.items() if (k in features and (v.dtype !=np.str_ and  v[0].dtype !=np.str_ ))
    # }
    tensor_dict = {}

    for k, v in np_example.items():
        try:
            if k not in features:
                continue
                
            is_string_type = (v.dtype == np.str_ or v[0].dtype == np.str_)
            
            if not is_string_type:
                tensor = torch.tensor(v)
                tensor_dict[k] = tensor
                
        except Exception as e:
            print(f" {k} : {str(e)}")
            continue

    # print(f" {len(tensor_dict)} ")
    # for k, v in tensor_dict.items():
    #     print(f"k: {k}, shape: {v.shape}, type: {v.dtype}")

    return tensor_dict


def make_data_config(
    config: ml_collections.ConfigDict,
    mode: str,
    num_res: int,
) -> Tuple[ml_collections.ConfigDict, List[str]]:
    cfg = copy.deepcopy(config)
    mode_cfg = cfg[mode]
    with cfg.unlocked():
        if mode_cfg.crop_size is None:
            mode_cfg.crop_size = num_res

    feature_names = cfg.common.unsupervised_features

    if cfg.common.use_templates:
        feature_names += cfg.common.template_features

    # print("mode: ", mode)
    breakpoint()
    if cfg[mode].supervised:
        feature_names += cfg.supervised.supervised_features
    print("feature_names: ", feature_names)
    return cfg, feature_names


def np_example_to_features(
    np_example: FeatureDict,
    config: ml_collections.ConfigDict,
    mode: str,
):
    np_example = dict(np_example)
    # breakpoint()
    # num_res = int(np_example["seq_length"][0])
    num_res = np_example["seq_length"].item()
    cfg, feature_names = make_data_config(config, mode=mode, num_res=num_res)
    # import pdb;pdb.set_trace()
    if "deletion_matrix_int" in np_example:
        np_example["deletion_matrix"] = np_example.pop(
            "deletion_matrix_int"
        ).astype(np.float32)
    print("feature_names: ", mode,feature_names)
    print("np_example: ", mode,np_example.keys())

    tensor_dict = np_to_tensor_dict(
        np_example=np_example, features=feature_names
    )
    print("mode: ", mode,tensor_dict.keys())
    # print("mode: ", mode)
    with torch.no_grad():
        features = input_pipeline.process_tensors_from_config(
            tensor_dict,
            cfg.common,
            cfg[mode],
        )

    return {k: v for k, v in features.items()}


class FeaturePipeline:
    def __init__(
        self,
        config: ml_collections.ConfigDict,
    ):
        self.config = config

    def process_features(
        self,
        raw_features: FeatureDict,
        mode: str = "train", 
    ) -> FeatureDict:
        return np_example_to_features(
            np_example=raw_features,
            config=self.config,
            mode=mode,
        )
