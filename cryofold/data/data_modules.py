import copy
from functools import partial
import json
import logging
import os,io
import pickle
from typing import Optional, Sequence, List, Any

import ml_collections as mlc
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import RandomSampler

from cryofold.data import (
    data_loader,
    feature_pipeline,
)
from cryofold.utils.tensor_utils import tensor_tree_map, dict_multimap



class CryoFoldSingleDataset(torch.utils.data.Dataset):
    def __init__(self,
        data_dir: str,
        name_list: str,
        max_template_date: str,
        config: mlc.ConfigDict,
        max_template_hits: int = 4,
        obsolete_pdbs_file_path: Optional[str] = None,
        template_release_dates_cache_path: Optional[str] = None,
        shuffle_top_k_prefiltered: Optional[int] = None,
        filter_path: Optional[str] = None,
        mode: str = "train", 
        _output_raw: bool = False,
    ):
        """
            Args:
                data_dir:
                    A path to a directory containing mmCIF files (in train
                    mode) or FASTA files (in inference mode).
                
                template_mmcif_dir:
                    Path to a directory containing template mmCIF files.
                config:
                    A dataset config object. See cryofold.config
                
                max_template_hits:
                    An upper bound on how many templates are considered. During
                    training, the templates ultimately used are subsampled
                    from this total quantity.
                template_release_dates_cache_path:
                    Path to the output of scripts/generate_mmcif_cache.
                obsolete_pdbs_file_path:
                    Path to the file containing replacements for obsolete PDBs.
                shuffle_top_k_prefiltered:
                    Whether to uniformly shuffle the top k template hits before
                    parsing max_template_hits of them. Can be used to
                    approximate DeepMind's training-time template subsampling
                    scheme much more performantly.
                mode:
                    "train", "val", or "predict"
        """
        super(CryoFoldSingleDataset, self).__init__()
        self.data_dir = data_dir
        self.config = config
        self.mode = mode
        self.name_list = name_list
        self.filter_path = filter_path
        self._output_raw = _output_raw

        self.supported_exts = [".cif", ".core", ".pdb"]

        valid_modes = ["train", "eval", "predict"]
        if(mode not in valid_modes):
            raise ValueError(f'mode must be one of {valid_modes}')

        self.filter_chain_ids = []
        if(filter_path is not None):
            with open(filter_path) as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip()=="":
                        continue
                    # fields = line.rstrip().split("\t")
                    fields = line.rstrip().split()
                    if len(fields)==0:
                        continue
                    name = fields[0]
                    self.filter_chain_ids.append(name)
            print("Filtering list: ", len(self.filter_chain_ids))

        self._chain_ids = []
        with open(self.name_list) as f:
            lines = f.readlines()
            for line in lines:
                # fields = line.rstrip().split("\t")
                fields = line.rstrip().split()
                name = fields[0]
                if name not in self.filter_chain_ids:
                    self._chain_ids.append(name)
        self.num = len(self._chain_ids)
        # print(f"{valid_modes} dataset: {len(self._chain_ids)}")
        # import pdb;pdb.set_trace()

        self.chain_data_caches = []
        self._chain_id_to_idx_dict = {
            chain: i for i, chain in enumerate(self._chain_ids)
        }

        
        self.set_ffdata()
        self.feature_pipeline = feature_pipeline.FeaturePipeline(config) 
    
    def set_ffdata(self):
        self.msa_ffindex = data_loader.get_ffindex(self.config.cryoem.dataset, "msa")
        self.pdb_ffindex = data_loader.get_ffindex(self.config.cryoem.dataset, "pdb")

    def chain_id_to_idx(self, chain_id):
        return self._chain_id_to_idx_dict[chain_id]

    def idx_to_chain_id(self, idx):
        return self._chain_ids[idx]
    
    def idx_to_pid_cid(self, idx):
        return self._chain_ids[idx].rsplit('_', 1)

    def add_to_filter_list(self, pdb_chain):
        with open(self.filter_path, 'a') as f:
            f.write(pdb_chain+"\n")

    def __get_random_item__(self):
        new_idx = np.random.randint(self.num)
        return self.__getitem__(new_idx) 

    def __getitem__(self, idx):
        # idx = 79361
        name = self.idx_to_chain_id(idx)
        pdb_chain = self._chain_ids[idx]
        if(self.mode == 'train' or self.mode == 'eval'):
            spl = name.rsplit('_', 1)
            if(len(spl) == 2):
                file_id, chain_id = spl
            else:
                print(f"wrong chain name: {pdb_chain} ")
                file_id, = spl
                chain_id = None

            # import pdb;pdb.set_trace()
            data = data_loader.get_data(idx, pdb_chain, self.msa_ffindex, self.pdb_ffindex, self.chain_data_caches, args=self.config.cryoem)
            if data is None:
                self.add_to_filter_list(pdb_chain+" 0")
                return self.__get_random_item__() 
            mask_all = data['all_atom_mask'][:,1].sum()
            # mask_gt  = data['atom14_gt_exists'][:,1].sum()
            is_resample = mask_all<30
            if is_resample:
                self.add_to_filter_list(pdb_chain+" 1")
                return self.__get_random_item__() 

        is_resample = True
        no_resample = 0
        while(is_resample):
            feats = self.feature_pipeline.process_features(
                data, self.mode 
            )
            # import pdb;pdb.set_trace()
            mask_all = feats['all_atom_mask'][:,1,0].sum()
            mask_gt  = feats['atom14_gt_exists'][:,1,0].sum()
            is_resample = min(mask_all, mask_gt)<30
            no_resample += 1
            if no_resample>3:
                self.add_to_filter_list(pdb_chain+" 2")
                print(f"no_resample: {pdb_chain} {no_resample} {mask_all} {mask_gt}")
                return self.__get_random_item__() 


        # if self.args.synmap:
        feats = data_loader.add_cryoem(feats, self.config.cryoem, pdb_chain)
        
        # print(feats["aatype"].shape)
        feats["batch_idx"] = torch.tensor([idx for _ in range(feats["aatype"].shape[-1])], dtype=torch.int64, device=feats["aatype"].device)

        return feats

    def __len__(self):
        return len(self._chain_ids) 


def deterministic_train_filter(
    chain_data_cache_entry: Any,
    max_resolution: float = 9.,
    max_single_aa_prop: float = 0.8,
) -> bool:
    # Hard filters
    resolution = chain_data_cache_entry.get("resolution", None)
    if(resolution is not None and resolution > max_resolution):
        return False

    seq = chain_data_cache_entry["seq"]
    counts = {}
    for aa in seq:
        counts.setdefault(aa, 0)
        counts[aa] += 1
    largest_aa_count = max(counts.values())
    largest_single_aa_prop = largest_aa_count / len(seq)
    if(largest_single_aa_prop > max_single_aa_prop):
        return False

    return True


def get_stochastic_train_filter_prob(
    chain_data_cache_entry: Any,
) -> List[float]:
    # Stochastic filters
    probabilities = []
    
    cluster_size = chain_data_cache_entry.get("cluster_size", None)
    if(cluster_size is not None and cluster_size > 0):
        probabilities.append(1 / cluster_size)
    
    chain_length = len(chain_data_cache_entry["seq"])
    probabilities.append((1 / 512) * (max(min(chain_length, 512), 256)))

    # Risk of underflow here?
    out = 1
    for p in probabilities:
        out *= p

    return out


class CryoFoldDataset(torch.utils.data.Dataset):
    """
        Implements the stochastic filters applied during AlphaFold's training.
        Because samples are selected from constituent datasets randomly, the
        length of an CryoFoldFilteredDataset is arbitrary. Samples are selected
        and filtered once at initialization.
    """
    def __init__(self,
        datasets: Sequence[CryoFoldSingleDataset],
        epoch_len: int,
        chain_data_cache_paths: List[str],
        generator: torch.Generator = None,
        _roll_at_init: bool = True,
    ):
        self.datasets = datasets
        self.epoch_len = epoch_len
        self.generator = generator
        
        self.chain_data_caches = []
        for path in chain_data_cache_paths:
            with open(path, "r") as fp:
                self.chain_data_caches.append(json.load(fp))
        
        self.datasets[0].chain_data_caches = self.chain_data_caches[0]

        def looped_shuffled_dataset_idx(dataset_len):
            while True:
                # Uniformly shuffle each dataset's indices
                weights = [1. for _ in range(dataset_len)]
                shuf = torch.multinomial(
                    torch.tensor(weights),
                    num_samples=dataset_len,
                    replacement=False,
                    generator=self.generator,
                )
                for idx in shuf:
                    yield idx

        def looped_samples(dataset_idx):
            max_cache_len = epoch_len
            dataset = self.datasets[dataset_idx]
            idx_iter = looped_shuffled_dataset_idx(len(dataset))

            while True:
                idx = []
                for _ in range(max_cache_len):
                    candidate_idx = next(idx_iter)
                    idx.append(candidate_idx)

                cache = [i for i in idx]
                for datapoint_idx in cache:
                    yield datapoint_idx

        self.samples = looped_samples(0) 

        if(_roll_at_init):
            self.reroll()

    def __getitem__(self, idx):
        dataset_idx, datapoint_idx = self.datapoints[idx]
        return self.datasets[dataset_idx][datapoint_idx]

    def __len__(self):
        return self.epoch_len

    def reroll(self):
        dataset_idx = 0
        self.datapoints = []
        for i in range(self.epoch_len):
            datapoint_idx = next(self.samples)
            self.datapoints.append((dataset_idx, datapoint_idx))


class CryoFoldBatchCollator:
    def __call__(self, prots):
        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, prots) 


class CryoFoldDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, config, stage="train", generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.stage = stage    

        if(generator is None):
            generator = torch.Generator()
        
        self.generator = generator
        self._prep_batch_properties_probs()

    def _prep_batch_properties_probs(self):
        keyed_probs = []
        stage_cfg = self.config[self.stage]

        max_iters = self.config.common.max_recycling_iters
        if(stage_cfg.supervised):
            clamp_prob = self.config.supervised.clamp_prob
            keyed_probs.append(
                ("use_clamped_fape", [1 - clamp_prob, clamp_prob])
            )
        
        if(stage_cfg.uniform_recycling):
            recycling_probs = [
                1. / (max_iters + 1) for _ in range(max_iters + 1)
            ]
        else:
            recycling_probs = [
                0. for _ in range(max_iters + 1)
            ]
            recycling_probs[-1] = 1.
        
        keyed_probs.append(
            ("no_recycling_iters", recycling_probs)
        )

        keys, probs = zip(*keyed_probs)
        max_len = max([len(p) for p in probs])
        padding = [[0.] * (max_len - len(p)) for p in probs] 
        
        self.prop_keys = keys
        self.prop_probs_tensor = torch.tensor(
            [p + pad for p, pad in zip(probs, padding)],
            dtype=torch.float32,
        )

    def _add_batch_properties(self, batch):
        samples = torch.multinomial(
            self.prop_probs_tensor,
            num_samples=1, # 1 per row
            replacement=True,
            generator=self.generator
        )

        aatype = batch["aatype"]
        # import pdb;pdb.set_trace()
        batch_dims = aatype.shape[:-2]
        recycling_dim = aatype.shape[-1]
        no_recycling = recycling_dim
        for i, key in enumerate(self.prop_keys):
            sample = int(samples[i][0])
            sample_tensor = torch.tensor(
                sample, 
                device=aatype.device, 
                requires_grad=False
            )
            orig_shape = sample_tensor.shape
            sample_tensor = sample_tensor.view(
                (1,) * len(batch_dims) + sample_tensor.shape + (1,)
            )
            sample_tensor = sample_tensor.expand(
                batch_dims + orig_shape + (recycling_dim,)
            )
            batch[key] = sample_tensor

            if(key == "no_recycling_iters"):
                no_recycling = sample 
        
        resample_recycling = lambda t: t[..., :no_recycling + 1]
        batch = tensor_tree_map(resample_recycling, batch)

        return batch

    def __iter__(self):
        it = super().__iter__()

        def _batch_prop_gen(iterator):
            for batch in iterator:
                yield self._add_batch_properties(batch)

        return _batch_prop_gen(it)


class CryoFoldDataModule(pl.LightningDataModule):
    def __init__(self,
        config: mlc.ConfigDict,
        max_template_date: str,
        train_data_dir: Optional[str] = None,
        train_chain_data_cache_path: Optional[str] = None,
        train_list: Optional[str] = None,
        val_data_dir: Optional[str] = None,
        val_list: Optional[str] = None,
        predict_data_dir: Optional[str] = None,
        train_filter_path: Optional[str] = None,
        obsolete_pdbs_file_path: Optional[str] = None,
        template_release_dates_cache_path: Optional[str] = None,
        batch_seed: Optional[int] = None,
        train_epoch_len: int = 50000, 
        **kwargs
    ):
        super(CryoFoldDataModule, self).__init__()

        self.config = config
        self.max_template_date = max_template_date
        self.train_data_dir = train_data_dir
        self.train_chain_data_cache_path = train_chain_data_cache_path
        self.train_list = train_list
        self.val_data_dir = val_data_dir
        self.val_list = val_list
        self.predict_data_dir = predict_data_dir
        self.train_filter_path = train_filter_path
        self.template_release_dates_cache_path = (
            template_release_dates_cache_path
        )
        self.obsolete_pdbs_file_path = obsolete_pdbs_file_path
        self.batch_seed = batch_seed
        self.train_epoch_len = train_epoch_len

        if(self.train_data_dir is None and self.predict_data_dir is None):
            raise ValueError(
                'At least one of train_data_dir or predict_data_dir must be '
                'specified'
            )
        self.training_mode = self.train_data_dir is not None


    def setup(self, stage=None):
        # Most of the arguments are the same for the three datasets 
        dataset_gen = partial(CryoFoldSingleDataset,
            max_template_date=self.max_template_date,
            config=self.config,
            template_release_dates_cache_path=
                self.template_release_dates_cache_path,
            obsolete_pdbs_file_path=
                self.obsolete_pdbs_file_path,
        )

        if(self.training_mode):
            train_dataset = dataset_gen(
                data_dir=self.train_data_dir,
                filter_path=self.train_filter_path,
                max_template_hits=self.config.train.max_template_hits,
                shuffle_top_k_prefiltered=
                    self.config.train.shuffle_top_k_prefiltered,
                mode="train",
                name_list=self.train_list,
            )

            datasets = [train_dataset]
            chain_data_cache_paths = [
                self.train_chain_data_cache_path,
            ]

            if(self.batch_seed is not None):
                generator = torch.Generator()
                generator = generator.manual_seed(self.batch_seed + 1)
            
            self.train_dataset = CryoFoldDataset(
                datasets=datasets,
                epoch_len=self.train_epoch_len,
                chain_data_cache_paths=chain_data_cache_paths,
                generator=generator,
                _roll_at_init=False,
            )
            # import pdb;pdb.set_trace()
            if self.val_data_dir == "None":
                self.val_data_dir = None

            if(self.val_data_dir is not None):
                # self.eval_dataset = dataset_gen(
                #     data_dir=self.val_data_dir,
                #     filter_path=None,
                #     max_template_hits=self.config.eval.max_template_hits,
                #     mode="eval",
                #     name_list=self.val_list,
                # )
                self.eval_dataset = dataset_gen(
                    data_dir=self.val_data_dir,
                    filter_path=None,
                    max_template_hits=self.config.train.max_template_hits,
                    mode="eval",
                    name_list=self.val_list,
                )
                self.eval_dataset.chain_data_caches = self.train_dataset.chain_data_caches[0]
                
            else:
                self.eval_dataset = None
        else:           
            self.predict_dataset = dataset_gen(
                data_dir=self.predict_data_dir,
                filter_path=None,
                max_template_hits=self.config.predict.max_template_hits,
                mode="predict",
            )

    def _gen_dataloader(self, stage):
        generator = torch.Generator()
        if(self.batch_seed is not None):
            generator = generator.manual_seed(self.batch_seed)

        dataset = None
        if(stage == "train"):
            dataset = self.train_dataset
            # Filter the dataset, if necessary
            dataset.reroll()
        elif(stage == "eval"):
            dataset = self.eval_dataset
        elif(stage == "predict"):
            dataset = self.predict_dataset
        else:
            raise ValueError("Invalid stage")

        batch_collator = CryoFoldBatchCollator()
        # print("num_workers:", self.config.data_module.data_loaders.num_workers)
        # print("batch_size:", self.config.data_module.data_loaders.batch_size)
        dl = CryoFoldDataLoader(
            dataset,
            config=self.config,
            stage=stage,
            generator=generator,
            batch_size=self.config.data_module.data_loaders.batch_size,
            num_workers=self.config.data_module.data_loaders.num_workers,
            collate_fn=batch_collator,
            pin_memory=True
        )

        return dl

    def train_dataloader(self):
        return self._gen_dataloader("train") 

    def val_dataloader(self):
        if(self.eval_dataset is not None):
            return self._gen_dataloader("eval")
        return None

    def predict_dataloader(self):
        return self._gen_dataloader("predict") 


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, batch_path):
        with open(batch_path, "rb") as f:
            self.batch = pickle.load(f)

    def __getitem__(self, idx):
        return copy.deepcopy(self.batch)

    def __len__(self):
        return 1000


class DummyDataLoader(pl.LightningDataModule):
    def __init__(self, batch_path):
        super().__init__()
        self.dataset = DummyDataset(batch_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset)
