import copy
import importlib
import ml_collections as mlc


def set_inf(c, inf):
    for k, v in c.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            c[k] = inf


def enforce_config_constraints(config):
    def string_to_setting(s):
        path = s.split('.')
        setting = config
        for p in path:
            setting = setting[p]

        return setting

    mutually_exclusive_bools = [
        (
            "model.template.average_templates", 
            "model.template.offload_templates"
        ),
        (
            "globals.use_lma",
            "globals.use_flash",
        ),
    ]

    for s1, s2 in mutually_exclusive_bools:
        s1_setting = string_to_setting(s1)
        s2_setting = string_to_setting(s2)
        if(s1_setting and s2_setting):
            raise ValueError(f"Only one of {s1} and {s2} may be set at a time")

    fa_is_installed = importlib.util.find_spec("flash_attn") is not None
    if(config.globals.use_flash and not fa_is_installed):
        raise ValueError("use_flash requires that FlashAttention is installed")


def model_config(name, train=False, low_prec=False):
    c = copy.deepcopy(config)
    #   CryoFold model
    # lr_scheduler
    c.lr_scheduler.max_lr = 0.0001
    c.lr_scheduler.start_decay_after_n_steps = 5000
    c.lr_scheduler.decay_every_n_steps = 5000
    
    # model frozen
    c.model.input_embedder.freeze = False
    c.model.template.freeze = False
    c.model.extra_msa.freeze = False
    c.model.evoformer_stack.freeze = False
    c.model.cryoem.cryoformer_embedder.freeze = False
    c.model.cryoem.cryoformer_encoder.freeze = False
    c.model.cryoem.cryoformer_decoder.freeze = False
    c.model.backbone_frame.freeze = False

    # model arch
    c.globals.use_single = True
    c.globals.fuse_single = False
    c.model.cryoem.enabled = True
    c.model.heads.amino_acid.enabled = True
    c.model.heads.secondary_structure.enabled = True
    c.model.backbone_frame.enabled = True
    c.model.heads.hungarian_matcher.enabled = True
    c.model.heads.cryoem_segmentation.enabled = True
    c.model.structure_module.enabled = True # use structure_module
    c.model.structure_module.use_cryoem_backbone = True
    c.model.structure_module.use_backbone_update = True
    c.model.structure_module.trans_scale_factor = 1
    c.data.train.max_extra_msa = 128
    c.data.train.crop_size = 384
    c.data.train.max_msa_clusters = 512
    # model weight
    c.loss.violation.weight = 1.
    c.loss.experimentally_resolved.weight = 0.01
    # stage2
    c.loss.amino_acid.weight = 10.0
    c.loss.secondary_structure.weight = 10.0
    c.loss.cryoem_segmentation.weight = 5.0
    c.loss.backbone_frame.weight = 0.1
    c.loss.normed_ca_init.weight = 50.0
    c.loss.normed_ca_final.weight = 50.0
    c.loss.masked_msa.weight = 10.0
    # aug
    c.data.cryoem.rotation = True
    c.data.cryoem.res_rand = True
    c.data.cryoem.no_padding = False
    c.data.cryoem.multimer = True
    
    # ema
    c.ema.decay = 0.9
    
    
    if low_prec:
        c.globals.eps = 1e-4
        # If we want exact numerical parity with the original, inf can't be
        # a global constant
        set_inf(c, 1e4)

    enforce_config_constraints(c)

    return c


c_z = mlc.FieldReference(128, field_type=int)
c_m = mlc.FieldReference(256, field_type=int)
c_t = mlc.FieldReference(64, field_type=int)
c_e = mlc.FieldReference(64, field_type=int)
c_s = mlc.FieldReference(384, field_type=int)
blocks_per_ckpt = mlc.FieldReference(None, field_type=int)
chunk_size = mlc.FieldReference(4, field_type=int)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
tm_enabled = mlc.FieldReference(False, field_type=bool)

aa_enabled = mlc.FieldReference(False, field_type=bool)
ss_enabled = mlc.FieldReference(False, field_type=bool)
bf_enabled = mlc.FieldReference(False, field_type=bool)
cs_enabled = mlc.FieldReference(False, field_type=bool)
hm_enabled = mlc.FieldReference(False, field_type=bool)
sm_enabled = mlc.FieldReference(True, field_type=bool)
sgl_enabled = mlc.FieldReference(False, field_type=bool)
cha_enabled = mlc.FieldReference(False, field_type=bool)
max_recycling_iters = mlc.FieldReference(3, field_type=int)

 


eps = mlc.FieldReference(1e-8, field_type=float)
templates_enabled = mlc.FieldReference(True, field_type=bool)
embed_template_torsion_angles = mlc.FieldReference(True, field_type=bool)
tune_chunk_size = mlc.FieldReference(True, field_type=bool)

NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

config = mlc.ConfigDict(
    {
        # Recurring FieldReferences that can be changed globally here
        "globals": {
            "blocks_per_ckpt": blocks_per_ckpt,
            "chunk_size": chunk_size,
            # Use Staats & Rabe's low-memory attention algorithm. Mutually
            # exclusive with use_flash.
            "use_lma": False,
            # Use FlashAttention in selected modules. Mutually exclusive with 
            # use_lma.
            "use_flash": False,
            "offload_inference": False,
            "use_single": sgl_enabled,
            "use_chain": cha_enabled,
            "fuse_single": True,
            "c_z": c_z,
            "c_m": c_m,
            "c_t": c_t,
            "c_e": c_e,
            "c_s": c_s,
            "eps": eps,
        },
        "lr_scheduler": {
            "last_epoch": -1, 
            "verbose": False,
            "base_lr": 0.,
            "max_lr": 0.001,
            "warmup_no_steps": 10, # default: 1000
            "start_decay_after_n_steps": 50000,
            "decay_every_n_steps":  50000,
            "decay_factor": 0.95,
        },
        "data": {
            "common": {
                "feat": {
                    "aatype": [NUM_RES],
                    "sstype": [NUM_RES],
                    "all_atom_mask": [NUM_RES, None],
                    "all_atom_positions": [NUM_RES, None, None],
                    "all_atom_mask_14": [NUM_RES, None, None],
                    "all_atom_positions_14": [NUM_RES, None, None],
                    "alt_chi_angles": [NUM_RES, None],
                    "atom14_alt_gt_exists": [NUM_RES, None],
                    "atom14_alt_gt_positions": [NUM_RES, None, None],
                    "atom14_atom_exists": [NUM_RES, None],
                    "atom14_atom_is_ambiguous": [NUM_RES, None],
                    "atom14_gt_exists": [NUM_RES, None],
                    "atom14_gt_positions": [NUM_RES, None, None],
                    "atom37_atom_exists": [NUM_RES, None],
                    "backbone_rigid_mask": [NUM_RES],
                    "backbone_rigid_tensor": [NUM_RES, None, None],
                    "bert_mask": [NUM_MSA_SEQ, NUM_RES],
                    "chi_angles_sin_cos": [NUM_RES, None, None],
                    "chi_mask": [NUM_RES, None],
                    "extra_deletion_value": [NUM_EXTRA_SEQ, NUM_RES],
                    "extra_has_deletion": [NUM_EXTRA_SEQ, NUM_RES],
                    "extra_msa": [NUM_EXTRA_SEQ, NUM_RES],
                    "extra_msa_mask": [NUM_EXTRA_SEQ, NUM_RES],
                    "extra_msa_row_mask": [NUM_EXTRA_SEQ],
                    "is_distillation": [],
                    "msa_feat": [NUM_MSA_SEQ, NUM_RES, None],
                    "msa_mask": [NUM_MSA_SEQ, NUM_RES],
                    "msa_row_mask": [NUM_MSA_SEQ],
                    "no_recycling_iters": [],
                    "debug": [],
                    "pseudo_beta": [NUM_RES, None],
                    "pseudo_beta_mask": [NUM_RES],
                    "residue_index": [NUM_RES],
                    "chain_index": [NUM_RES],
                    "residx_atom14_to_atom37": [NUM_RES, None],
                    "residx_atom37_to_atom14": [NUM_RES, None],
                    "resolution": [],
                    "rigidgroups_alt_gt_frames": [NUM_RES, None, None, None],
                    "rigidgroups_group_exists": [NUM_RES, None],
                    "rigidgroups_group_is_ambiguous": [NUM_RES, None],
                    "rigidgroups_gt_exists": [NUM_RES, None],
                    "rigidgroups_gt_frames": [NUM_RES, None, None, None],
                    "seq_length": [],
                    "seq_mask": [NUM_RES],
                    "target_feat": [NUM_RES, None],
                    "template_aatype": [NUM_TEMPLATES, NUM_RES],
                    "template_all_atom_mask": [NUM_TEMPLATES, NUM_RES, None],
                    "template_all_atom_positions": [
                        NUM_TEMPLATES, NUM_RES, None, None,
                    ],
                    "template_alt_torsion_angles_sin_cos": [
                        NUM_TEMPLATES, NUM_RES, None, None,
                    ],
                    "template_backbone_rigid_mask": [NUM_TEMPLATES, NUM_RES],
                    "template_backbone_rigid_tensor": [
                        NUM_TEMPLATES, NUM_RES, None, None,
                    ],
                    "template_mask": [NUM_TEMPLATES],
                    "template_pseudo_beta": [NUM_TEMPLATES, NUM_RES, None],
                    "template_pseudo_beta_mask": [NUM_TEMPLATES, NUM_RES],
                    "template_sum_probs": [NUM_TEMPLATES, None],
                    "template_torsion_angles_mask": [
                        NUM_TEMPLATES, NUM_RES, None,
                    ],
                    "template_torsion_angles_sin_cos": [
                        NUM_TEMPLATES, NUM_RES, None, None,
                    ],
                    "true_msa": [NUM_MSA_SEQ, NUM_RES],
                    "use_clamped_fape": [],
                },
                "masked_msa": {
                    "profile_prob": 0.1,
                    "same_prob": 0.1,
                    "uniform_prob": 0.1,
                },
                "max_recycling_iters": max_recycling_iters,
                "msa_cluster_features": True,
                "reduce_msa_clusters_by_max_templates": False,
                "resample_msa_in_recycling": True,
                "template_features": [
                    "template_all_atom_positions",
                    "template_sum_probs",
                    "template_aatype",
                    "template_all_atom_mask",
                ],
                "unsupervised_features": [
                    "aatype",
                    "residue_index",
                    "chain_index",
                    "msa",
                    "num_alignments",
                    "seq_length",
                    "between_segment_residues",
                    "deletion_matrix",
                    "no_recycling_iters",
                    "debug",
                    "cryoem_density",
                    "cryoem_offset",
                    "cryoem_cropidx",
                    "cryoem_apix",
                    "cryoem_mask",
                    "cryoem_orisize",
                    "cryoem_size",
                    "cryoem_seglabel",
                    "normed_ca_positions",
                ],
                "use_templates": templates_enabled,
                "use_template_torsion_angles": embed_template_torsion_angles,
            },
            "supervised": {
                "clamp_prob": 0.9,
                "supervised_features": [
                    "all_atom_mask",
                    "all_atom_positions",
                    "all_atom_mask_14",
                    "all_atom_positions_14",
                    "resolution",
                    "sstype",
                    "use_clamped_fape",
                    "is_distillation",
                ],
            },
            "predict": {
                "fixed_size": True,
                "subsample_templates": False,  # We want top templates.
                "masked_msa_replace_fraction": 0.15,
                "max_msa_clusters": 512,
                "max_extra_msa": 1024,
                "max_template_hits": 20,
                "max_templates": 20,
                "crop": False,
                "crop_size": None,
                "supervised": True,
                "uniform_recycling": False,
            },
            "eval": {
                "fixed_size": True,
                "subsample_templates": False,  # We want top templates.
                "masked_msa_replace_fraction": 0.15,
                "max_msa_clusters": 128,
                "max_extra_msa": 1024,
                "max_template_hits": 4,
                "max_templates": 4,
                "crop": False,
                "crop_size": None,
                "supervised": True,
                "uniform_recycling": False,
            },
            "train": {
                "fixed_size": True,
                "subsample_templates": True,
                "masked_msa_replace_fraction": 0.15,
                "max_msa_clusters": 128,
                "max_extra_msa": 1024,
                "max_template_hits": 4,
                "max_templates": 4,
                "shuffle_top_k_prefiltered": 20,
                "crop": True,
                "crop_size": 256,
                "supervised": True,
                "clamp_prob": 0.9,
                "max_distillation_msa_clusters": 1000,
                "uniform_recycling": True,
                "distillation_prob": 0.75,
            },
            "data_module": {
                "use_small_bfd": False,
                "data_loaders": {
                    "batch_size": 1,
                    "num_workers": 16,
                },
            },
            "cryoem": {
                "synmap": True,
                "res_rand": False,
                "max_recycling_iters": max_recycling_iters,
                "resolution": 3.0,
                "apix": 2/3,
                "no_padding": True,
                "cube_width": 128,
                "axis": 3,
                "rotation": False,
                "debug": False,
                "inference": False,
                "dataset": "pdb",
                "multimer": False,
                "map_path": ""
            },
        },
        
        "model": {
            "freeze": False,
            "_mask_trans": False,
            "input_embedder": {
                "tf_dim": 22,
                "msa_dim": 49,
                "c_z": c_z,
                "c_m": c_m,
                "c_s": c_s,
                "relpos_k": 32,
                "freeze": False,
                "use_single": sgl_enabled,
                "use_chain": cha_enabled,

            },
            "recycling_embedder": {
                "c_z": c_z,
                "c_m": c_m,
                "c_s": c_s,
                "min_bin": 3.25,
                "max_bin": 20.75,
                "no_bins": 15,
                "inf": 1e8,
                "freeze": False,
                "use_single": sgl_enabled,
                "use_chain": cha_enabled,
            },
            "template": {
                "distogram": {
                    "min_bin": 3.25,
                    "max_bin": 50.75,
                    "no_bins": 39,
                },
                "template_angle_embedder": {
                    # DISCREPANCY: c_in is supposed to be 51.
                    "c_in": 57,
                    "c_out": c_m,
                },
                "template_pair_embedder": {
                    "c_in": 88,
                    "c_out": c_t,
                },
                "template_pair_stack": {
                    "c_t": c_t,
                    # DISCREPANCY: c_hidden_tri_att here is given in the supplement
                    # as 64. In the code, it's 16.
                    "c_hidden_tri_att": 16,
                    "c_hidden_tri_mul": 64,
                    "no_blocks": 2,
                    "no_heads": 4,
                    "pair_transition_n": 2,
                    "dropout_rate": 0.25,
                    "blocks_per_ckpt": blocks_per_ckpt,
                    "tune_chunk_size": tune_chunk_size,
                    "inf": 1e9,
                },
                "template_pointwise_attention": {
                    "c_t": c_t,
                    "c_z": c_z,
                    # DISCREPANCY: c_hidden here is given in the supplement as 64.
                    # It's actually 16.
                    "c_hidden": 16,
                    "no_heads": 4,
                    "inf": 1e5,  # 1e9,
                },
                "inf": 1e5,  # 1e9,
                "eps": eps,  # 1e-6,
                "enabled": templates_enabled,
                "embed_angles": embed_template_torsion_angles,
                "use_unit_vector": False,
                # Approximate template computation, saving memory.
                # In our experiments, results are equivalent to or better than
                # the stock implementation. Should be enabled for all new
                # training runs.
                "average_templates": False,
                # Offload template embeddings to CPU memory. Vastly reduced
                # memory consumption at the cost of a modest increase in
                # runtime. Useful for inference on very long sequences.
                # Mutually exclusive with average_templates.
                "offload_templates": False,
                "freeze": False,
            },
            "extra_msa": {
                "extra_msa_embedder": {
                    "c_in": 25,
                    "c_out": c_e,
                },
                "extra_msa_stack": {
                    "c_m": c_e,
                    "c_z": c_z,
                    "c_hidden_msa_att": 8,
                    "c_hidden_opm": 32,
                    "c_hidden_mul": 128,
                    "c_hidden_pair_att": 32,
                    "no_heads_msa": 8,
                    "no_heads_pair": 4,
                    "no_blocks": 4,
                    "transition_n": 4,
                    "msa_dropout": 0.15,
                    "pair_dropout": 0.25,
                    "clear_cache_between_blocks": False,
                    "tune_chunk_size": tune_chunk_size,
                    "inf": 1e9,
                    "eps": eps,  # 1e-10,
                    "ckpt": blocks_per_ckpt is not None,
                },
                "enabled": True,
                "freeze": False,
            },
            "cryoem":{
                "cryoformer_embedder": {
                    "hidden_dim": 384,
                    "backbone": "resnet50t2",
                    "frozen_backbone":False,
                    "dilation": False,
                    "density_head": True,
                    "freeze": False,

                },
                "cryoformer_encoder":{
                    "d_model":384, 
                    "no_head":8, 
                    "no_blocks":8,
                    "dim_feedforward":2048,
                    "dropout":0.1,
                    "activation": "relu",
                    "freeze": False,
                },
                "cryoformer_decoder":{
                    "d_model":384, 
                    "no_head":8, 
                    "no_blocks": 8,
                    "dim_feedforward":2048,
                    "dropout":0.1,
                    "activation": "relu",
                    "return_intermediate": True,
                    "freeze": False,
                },
                "enabled": False,
                "freeze": False,
            },
            "evoformer_stack": {
                "c_m": c_m,
                "c_z": c_z,
                "c_hidden_msa_att": 32,
                "c_hidden_opm": 32,
                "c_hidden_mul": 128,
                "c_hidden_pair_att": 32,
                "c_s": c_s,
                "no_heads_msa": 8,
                "no_heads_pair": 4,
                "no_blocks": 48,
                "transition_n": 4,
                "msa_dropout": 0.15,
                "pair_dropout": 0.25,
                "blocks_per_ckpt": blocks_per_ckpt,
                "clear_cache_between_blocks": False,
                "tune_chunk_size": tune_chunk_size,
                "inf": 1e9,
                "eps": eps,  # 1e-10,
                "freeze": False,
            },
            "structure_module": {
                "c_s": c_s,
                "c_z": c_z,
                "c_ipa": 16,
                "c_resnet": 128,
                "no_heads_ipa": 12,
                "no_qk_points": 4,
                "no_v_points": 8,
                "dropout_rate": 0.1,
                "no_blocks": 8,
                "no_transition_layers": 1,
                "no_resnet_blocks": 2,
                "no_angles": 7,
                "use_cryoem_backbone": False,
                "use_backbone_update": True,
                "trans_scale_factor": 10,
                "epsilon": eps,  # 1e-12,
                "inf": 1e5,
                "enabled": sm_enabled,
                "freeze": False,
            },
            "backbone_frame": {
                "c_s": c_s,
                "c_out_q": 4,
                "c_out_t": 3,
                "enabled": bf_enabled,
                "freeze": False,

            },
            "heads": {
                "lddt": {
                    "no_bins": 50,
                    "c_in": c_s,
                    "c_hidden": 128,
                    "enabled": sm_enabled,
                },
                "distogram": {
                    "c_z": c_z,
                    "no_bins": aux_distogram_bins,
                },
                "tm": {
                    "c_z": c_z,
                    "no_bins": aux_distogram_bins,
                    "enabled": tm_enabled,
                },
                "masked_msa": {
                    "c_m": c_m,
                    "c_out": 23,
                },
                "amino_acid": 
                {
                    "c_s": c_s,
                    "c_out": 21,
                    "enabled": aa_enabled,
                },
                "secondary_structure": {
                    "c_s": c_s,
                    "c_out": 4,
                    "enabled": ss_enabled,
                },
                
                "cryoem_segmentation": {
                    "hidden_dim":64, 
                    "hidden_dim_cf": 384, 
                    "dropout": 0.1, 
                    "patch_c1":512, 
                    "patch_c2": 2048,
                    "c_out": 24,
                    "enabled": cs_enabled,
                },
                "hungarian_matcher": {
                    "cost_aa": 1.0,
                    "cost_ss": 1.0,
                    "cost_ca": 5.0,
                    "enabled": hm_enabled,
                },
                "experimentally_resolved": {
                    "c_s": c_s,
                    "c_out": 37,
                },
            },
        },
        "relax": {
            "max_iterations": 0,  # no max
            "tolerance": 2.39,
            "stiffness": 10.0,
            "max_outer_iterations": 20,
            "exclude_residues": [],
        },
        "loss": {
            "distogram": {
                "min_bin": 2.3125,
                "max_bin": 21.6875,
                "no_bins": 64,
                "eps": eps,  # 1e-6,
                "weight": 0.3,
            },
            "hungarian_matcher": {
                "weight": 1.0,
                "enabled": hm_enabled,

            },
            "amino_acid": {
                "eps": eps,  # 1e-8,
                "weight": 1.0,
                "enabled": aa_enabled,
            },
            "secondary_structure": {
                "eps": eps,  # 1e-8,
                "weight": 1.0,
                "enabled": ss_enabled,
            },
            "backbone_frame": {
                "eps": eps,  # 1e-8,
                "weight": 0.1,
                "enabled": bf_enabled,
            },
            "normed_ca_init": {
                "eps": eps,  # 1e-8,
                "weight": 0.1,
            },
            "normed_ca_final": {
                "eps": eps,  # 1e-8,
                "weight": 0.1,
            },
            "cryoem_segmentation":{
                "eps": eps,  # 1e-8,
                "weight": 1.0,
                "enabled": cs_enabled,
            },
            "experimentally_resolved": {
                "eps": eps,  # 1e-8,
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "weight": 0.0,
            },
            "fape": {
                "backbone": {
                    "clamp_distance": 10.0,
                    "loss_unit_distance": 10.0,
                    "weight": 0.5,
                },
                "sidechain": {
                    "clamp_distance": 10.0,
                    "length_scale": 10.0,
                    "weight": 0.5,
                },
                "eps": 1e-4,
                "weight": 1.0,
            },
            "structure_module": {
                "enabled": sm_enabled,
            },
            "plddt_loss": {
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "cutoff": 15.0,
                "no_bins": 50,
                "eps": eps,  # 1e-10,
                "weight": 0.01,

            },
            "masked_msa": {
                "eps": eps,  # 1e-8,
                "weight": 2.0,
            },
            "supervised_chi": {
                "chi_weight": 0.5,
                "angle_norm_weight": 0.01,
                "eps": eps,  # 1e-6,
                "weight": 1.0,
            },
            "violation": {
                "violation_tolerance_factor": 12.0,
                "clash_overlap_tolerance": 1.5,
                "eps": eps,  # 1e-6,
                "weight": 0.0,
            },
            "tm": {
                "max_bin": 31,
                "no_bins": 64,
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "eps": eps,  # 1e-8,
                "weight": 0.,
                "enabled": tm_enabled,
            },
            "eps": eps,
        },
        "ema": {"decay": 0.999},
    }
)
