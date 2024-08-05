
from tkinter import E
import numpy as np
import pickle
import os, io, json, gzip
import torch

from cryofold.data.data_transforms import atom37_to_torsion_angles
from cryofold.data.ffindex import FFindex

from cryofold.utils.rigid_utils import rot_by_axis
from cryofold.utils.protein_utils import save_as_pdb
from cryofold.np import constants

from cryofold.data import cryo_utils
from cryofold.np.density import DensityInfo


chain_data_cache_path = "data/dataset_split/chain_data_cache.json"
# pdb_ffindex_prefix = "data/ffindex/PDB_3D_concat_10W"
# msa_ffindex_prefix = "data/ffindex/augument_features_concat_10W"
DATASET = {
    'antibody':{
        "msa": "data/ffindex/ab_msa",
        "pdb": "data/ffindex/ab_3d",
    },
    'casp':{
        "msa": "data/ffindex/casp_msa",
        "pdb": "data/ffindex/casp_pdb",
    },
    'pdb':{
        "msa": "data/ffindex/augument_features_concat_10W",
        "pdb": "data/ffindex/PDB_3D_concat_10W",
    },
}

def get_ffindex(dataset, type):
    ffindex_prefix = DATASET[dataset][type]
    return FFindex(ffindex_prefix+".ffdata", ffindex_prefix+".ffindex", dynamic_file_handle=True)


def get_msa(ffindex, key, pkl_path="", max_seq=20480):
    """
    Get a sample from FFindex. If there are too many sequence in MSA, random sample from it
    
    Parameters
    ----------
    ffindex: FFindex of augument_features.ffdata
    key: str
    
    Return
    ----------
    np_example: dict
    """
    if pkl_path!="":
        print("Loading from pkl file...")
        np_example = pickle.load(gzip.open(pkl_path,"r"))  
    else:
        np_example = pickle.load(io.BytesIO(ffindex.get(key, decompress=True, decode=False)) )
    num_seq, num_res = np_example['msa'].shape
    np_example.update({'pos_index': np.arange(num_res).reshape(-1,1)})

    dd=(np_example['msa']==21).sum(1)
    select_seqs = (dd<=int(num_res*0.3))
    if select_seqs.sum() > 100:
        np_example['msa'] = np_example['msa'][select_seqs]
        np_example['deletion_matrix_int'] = np_example['deletion_matrix_int'][select_seqs]
        np_example['num_alignments'][:] = select_seqs.sum()
        # print(key, "msa",num_seq, np_example['msa'].shape[0], np_example['msa'].shape[0]/num_seq)
        
    return np_example

def get_data(idx, pdb_chain, msa_ffindex=None, pdb_ffindex=None, chain_data_caches=None, infer=False, args=None):
    # if self.args.infer and not self.args.eval:
    #     batchY = {
    #         'id':  np.array([idx]),
    #         'midx':  np.array([self.mono_idx]),
    #         'debug': np.array([0]),
    #     }
    #     return batchY
    
    if msa_ffindex is None:
        msa_ffindex = get_ffindex(args.dataset, "msa")
    if pdb_ffindex is None:
        pdb_ffindex = get_ffindex(args.dataset, "pdb")
    msa_templ = load_features_from_ffdata(idx, pdb_chain, msa_ffindex, args)
    if msa_templ is None:
        return None
    targets   = load_targets_from_ffdata(idx, pdb_chain, pdb_ffindex, args)
    # print("chain_data_caches",len(chain_data_caches))
    if chain_data_caches is None:
        with open(chain_data_cache_path, "r") as fp:
            chain_data_caches = json.load(fp)

    # import pdb;pdb.set_trace()
    # print("pdb_chain: ", pdb_chain)
    # print("pdb_chain: ", pdb_chain, pdb_chain in chain_data_caches)
    targets.update(msa_templ)
    targets['is_distillation'] = np.array(0., dtype=np.float32)
    # try:
    if pdb_chain in chain_data_caches:
        targets['resolution'] = np.array([chain_data_caches[pdb_chain]['resolution']])
    else:
        targets['resolution'] = np.array([4.0])
    # except:
    #     import pdb;pdb.set_trace()
    targets["no_recycling_iters"] = np.array(args.max_recycling_iters)
    if args.debug:
        targets["debug"] = np.array(1.)
    else:
        targets["debug"] = np.array(0.)
    # targets["no_recycling_iters"] = np.array([3., 3., 3., 3.,])   
    # if infer:
    #     add_cryoem(targets, args)
    torsion_angles_fn = atom37_to_torsion_angles()

    prots = {}
    prots['aatype'] = torch.from_numpy(targets['aatype']).argmax(-1)
    prots['all_atom_positions'] = torch.from_numpy(targets['all_atom_positions'])
    prots['all_atom_mask'] = torch.from_numpy(targets['all_atom_mask'])
    try:
        angles_new = torsion_angles_fn(prots)
    except:
        print(f"wrong angles: {pdb_chain}")
        return None
    for k in ['torsion_angles_sin_cos', 'alt_torsion_angles_sin_cos', 'torsion_angles_mask']:
        targets[k] = angles_new[k].numpy()

    
    save_gt = False
    output_path = f"./predictions/{pdb_chain}_{args.axis}.pdb"
    if save_gt and not os.path.exists(output_path):
        occupancies = np.tile(targets['sstype'].reshape(-1,1,1), [1,37,1]) + 1
        save_as_pdb(
            targets['aatype'].argmax(-1),
            targets['residue_index'],
            targets['all_atom_positions'],
            targets['all_atom_mask'],
            out_file=output_path,
            b_factors=None,
            asym_id=None,
            occupancies=occupancies,
            ss_coding=[3,1,2],
            write_ss=True,
            gap_ter_threshold=9.0
        )
        print("saved gt file: ", output_path)

    return targets

    
def load_features_from_ffdata(idx, pdb_chain, msa_ffindex=None, args=None):
    """
    Loading msa tempaltes
    
    Keys: 
        'aatype'
        'between_segment_residues'
        'domain_name'
        'residue_index'
        'seq_length'
        'sequence'
        'deletion_matrix_int'
        'msa'
        'num_alignments'
        'template_aatype'
        'template_all_atom_masks'
        'template_all_atom_positions'
        'template_domain_names'
        'template_e_value'
        'template_neff'
        'template_prob_true'
        'template_release_date'
        'template_score'
        'template_similarity'
        'template_sequence'
        'template_sum_probs'
        'template_confidence_scores'
        'ss_pred'
        'ss_conf'
        'chain_index'
        'pos_index'
        'template_all_atom_mask'
    """
    if args.multimer:
        return load_features_from_ffdata_multimer(idx, pdb_chain, msa_ffindex, args)

    pdbname, chainname = pdb_chain.rsplit('_', 1)

    msa_templ = get_msa(msa_ffindex, pdbname+".pkl.gz")
    if args.dataset == "casp":
    # if 'chain_index' not in msa_templ:
        # import pdb;pdb.set_trace()
        msa_templ['chain_index'] = np.array(['A'] *len( msa_templ['aatype']))


    try:
        chain_idx = np.where(np.unique(msa_templ['chain_index'])==chainname)[0][0]
    except:
        print(pdbname, chainname)
    seq_indices = np.in1d(msa_templ['chain_index'], chainname.split(',')).nonzero()[0]
    # for validation
    msa_templ['seq_indices'] = seq_indices

    msa_templ = cryo_utils.filter_inputs_by_seq_idx(msa_templ, seq_indices)
    msa_templ['template_all_atom_mask'] = msa_templ['template_all_atom_masks']
    del msa_templ['template_all_atom_masks']
    del_keys = ['template_domain_names',
                'template_e_value',
                'template_neff',
                'template_prob_true',
                'template_release_date',
                'template_score',
                'template_similarity',
                'template_sequence',
                # 'template_sum_probs',
                'template_confidence_scores']
    for k in del_keys:
        if k in msa_templ:
            del msa_templ[k]
    if args.dataset != "casp":
        try:
            msa_templ['template_sum_probs'] = msa_templ['template_sum_probs'][chain_idx] 
        except:
            print(pdb_chain, "miss matched number of the chain.")
            try:
                msa_templ['template_sum_probs'] = msa_templ['template_sum_probs'][0] 
            except:
                return None
    # import pdb;pdb.set_trace()
    msa_templ['chain_index'] = np.zeros(len(seq_indices))
    return msa_templ

def load_features_from_ffdata_multimer(idx, pdb_chain, msa_ffindex=None, args=None):
    """
    Loading features for multimer
    """
    pc = pdb_chain.rsplit('_')
    pdbname, chainnames = pc[0], pc[1:]
    # import pdb;pdb.set_trace()
    if args.dataset == "antibody":
        key_name = pdb_chain
    elif args.dataset == "casp":
        key_name = pdbname
    else:
        key_name = pdbname

    msa_templ = get_msa(msa_ffindex, key_name+".pkl.gz")
    if args.dataset == "casp":
    # if 'chain_index' not in msa_templ:
        # import pdb;pdb.set_trace()
        msa_templ['chain_index'] = np.array(['A'] *len( msa_templ['aatype']))

    seq_indices = np.in1d(msa_templ['chain_index'], chainnames).nonzero()[0]
    # for validation
    msa_templ['seq_indices'] = seq_indices
    try:
        msa_templ = cryo_utils.filter_inputs_by_seq_idx(msa_templ, seq_indices)
    except:
        print(pdbname, chainnames)
        import pdb;pdb.set_trace()
        return None
    msa_templ['template_all_atom_mask'] = msa_templ['template_all_atom_masks']
    del msa_templ['template_all_atom_masks']
    del_keys = ['template_domain_names',
                'template_e_value',
                'template_neff',
                'template_prob_true',
                'template_release_date',
                'template_score',
                'template_similarity',
                'template_sequence',
                # 'template_sum_probs',
                'template_confidence_scores']
    for k in del_keys:
        if k in msa_templ:
            del msa_templ[k]
    template_sum_probs = []
    for chainname in chainnames:
        chain_idx = np.where(np.unique(msa_templ['chain_index'])==chainname)[0][0]
        template_sum_probs_idx = msa_templ['template_sum_probs'][chain_idx] 
        template_sum_probs.append(template_sum_probs_idx)
    msa_templ['template_sum_probs'] = np.mean(template_sum_probs, axis=0)
    # if args.dataset != "casp":
    #     try:
    #         msa_templ['template_sum_probs'] = 
    #     except:
    #         print(pdb_chain, "miss matched number of the chain.")
    #         try:
    #             msa_templ['template_sum_probs'] = msa_templ['template_sum_probs'][0] 
    #         except:
    #             return None
    chain_names_uq = np.unique(list(chainnames))
    CHAIN_TO_ID = { chain_names_uq[i]:i for i in range(len(chain_names_uq))}
    chainidx = np.array([CHAIN_TO_ID[s] for s in msa_templ['chain_index']])
    # import pdb;pdb.set_trace()
    msa_templ['chain_index'] = chainidx
    seq_length = len(chainidx)
    msa_templ['seq_length'] = np.array([seq_length]*seq_length)
    return msa_templ

def load_targets_from_ffdata(idx, pdb_chain, pdb_ffindex=None, args=None):
    """
    Loading targets
    
    Keys: 
        'all_atom_positions'
        'all_atom_masks'
        'sequence'
        'aatype'
        'domain_names'
        'rotation'
        'translation'
        'torsion_angles_sin_cos'
        'alt_torsion_angles_sin_cos'
        'torsion_angles_mask'
        'chain_index'
        'ss'
        'sse'
        'all_atom_mask'

    """
    if args.multimer:
        return load_targets_from_ffdata_multimer(idx, pdb_chain, pdb_ffindex, args)

    pdbname, chainname = pdb_chain.rsplit('_', 1)


    pdb_ffdata = pdb_ffindex.get(pdbname+".pkl.gz", decode=False)
    targets = pickle.load(io.BytesIO(pdb_ffdata))
    # import pdb;pdb.set_trace()
    
    if 'template_all_atom_positions' not in targets:
        if 'all_atom_mask' in targets:
            targets['all_atom_masks'] = targets['all_atom_mask']
        if 'chain_names' not in targets:
            targets['chain_names'] = np.array(['A']*len(targets['aatype']))
        if 'ss' not in targets:
            targets['ss'] = np.array(['H']*len(targets['aatype']))
        if 'rotation' not in targets:
            targets['rotation'] = np.array([np.eye(3)]*len(targets['aatype']))
        if 'translation' not in targets:
            targets['translation'] = targets['all_atom_positions'][:,1]
        targets = {
            'template_all_atom_positions': targets['all_atom_positions'],
            'template_all_atom_masks': targets['all_atom_masks'],
            # 'template_sequence': targets['sequence'],
            'template_aatype': targets['aatype'],
            'rotation': targets['rotation'],
            'translation': targets['translation'],
            # 'torsion_angles_sin_cos': targets['torsion_angles_sin_cos'],
            # 'alt_torsion_angles_sin_cos': targets['alt_torsion_angles_sin_cos'],
            # 'torsion_angles_mask': targets['torsion_angles_mask'],
            'chain_index': targets['chain_names'],
            'ss': targets['ss']
        }
    
    targets = {k.replace("template_",""):v for k,v in targets.items()}
    targets['sstype'] = get_sseids_at(targets['ss'])
    # import pdb;pdb.set_trace()
    if len(targets['aatype'].shape) ==2:
        aatype_hb = np.argmax(targets['aatype'], 1)
        aatype = np.array([constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[r] for r in aatype_hb])

    seq_indices = np.in1d(targets['chain_index'], chainname.split(',')).nonzero()[0]
    # for validation 
    targets['seq_indices'] = seq_indices
    targets = cryo_utils.filter_targets_by_seq_idx(targets, seq_indices)

    targets['all_atom_mask'] = targets['all_atom_masks']

    del targets['aatype']
    del targets['all_atom_masks']
    if 'sequence' in targets:
        del targets['sequence']


    # rotation augmentation
    rand_axis = args.axis
    # rand_axis = 2
    targets['axis'] = np.array([rand_axis])

    if args.rotation:
        # rand_axis = (np.random.randint(4)+targets['all_atom_mask'][:,0].sum())%4
        rand_axis = idx % 4 
        # rand_axis = torch.randint(4,(1,),generator=g).item()
        # print("rand_axis: ", rand_axis, idx)
    # import pdb;pdb.set_trace()
    if rand_axis <3:
        # import pdb;pdb.set_trace()
        # print("rot with: ", rand_axis)
        all_atom_positions = targets['all_atom_positions']
        all_atom_mask      = targets['all_atom_mask']
        rot = rot_by_axis(rand_axis)
        pos = all_atom_positions[all_atom_mask[:,1]>0]
        cen = pos[:,1].mean(0)
        all_atom_positions = ((all_atom_positions-cen) @ rot + cen)  * all_atom_mask[:,:,None]
        targets['all_atom_positions'] = all_atom_positions
        targets['translation'] = all_atom_positions[:,1]
        targets['axis'] = np.array([rand_axis])
    # elif rand_axis ==4:
    #     all_atom_positions = (all_atom_positions + 50)  * all_atom_mask[:,:,None]                    
    #     targets['template_all_atom_positions'] = all_atom_positions
    #     targets['translation'] = all_atom_positions[:,1]
    # import pdb;pdb.set_trace()


        
    return targets


def load_targets_from_ffdata_multimer(idx, pdb_chain, pdb_ffindex=None, args=None):
    pc = pdb_chain.rsplit('_')
    pdbname, chainnames = pc[0], pc[1:]
    # import pdb;pdb.set_trace()

    if args.dataset == "antibody":
        key_name = pdb_chain
    elif args.dataset == "casp":
        key_name = pdbname
    else:
        key_name = pdbname

    pdb_ffdata = pdb_ffindex.get(key_name+".pkl.gz", decode=False)
    targets = pickle.load(io.BytesIO(pdb_ffdata))
    # import pdb;pdb.set_trace()
    
    if 'template_all_atom_positions' not in targets:
        if 'all_atom_mask' in targets:
            targets['all_atom_masks'] = targets['all_atom_mask']
        if 'chain_names' not in targets:
            targets['chain_names'] = np.array(['A']*len(targets['aatype']))
        if 'ss' not in targets:
            targets['ss'] = np.array(['H']*len(targets['aatype']))
        if 'rotation' not in targets:
            targets['rotation'] = np.array([np.eye(3)]*len(targets['aatype']))
        if 'translation' not in targets:
            targets['translation'] = targets['all_atom_positions'][:,1]
        targets = {
            'template_all_atom_positions': targets['all_atom_positions'],
            'template_all_atom_masks': targets['all_atom_masks'],
            # 'template_sequence': targets['sequence'],
            'template_aatype': targets['aatype'],
            'rotation': targets['rotation'],
            'translation': targets['translation'],
            # 'torsion_angles_sin_cos': targets['torsion_angles_sin_cos'],
            # 'alt_torsion_angles_sin_cos': targets['alt_torsion_angles_sin_cos'],
            # 'torsion_angles_mask': targets['torsion_angles_mask'],
            'chain_index': targets['chain_names'],
            'ss': targets['ss']
        }
    
    targets = {k.replace("template_",""):v for k,v in targets.items()}
    targets['sstype'] = get_sseids_at(targets['ss'])
    # import pdb;pdb.set_trace()
    if len(targets['aatype'].shape) ==2:
        aatype_hb = np.argmax(targets['aatype'], 1)
        aatype = np.array([constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[r] for r in aatype_hb])

    seq_indices = np.in1d(targets['chain_index'], chainnames).nonzero()[0]
    # seq_indices = np.in1d(targets['chain_index'], chainname.split(',')).nonzero()[0]
    # for validation 
    targets['seq_indices'] = seq_indices
    targets = cryo_utils.filter_targets_by_seq_idx(targets, seq_indices)

    targets['all_atom_mask'] = targets['all_atom_masks']

    del targets['aatype']
    del targets['chain_index']
    del targets['all_atom_masks']
    if 'sequence' in targets:
        del targets['sequence']


    # rotation augmentation
    rand_axis = args.axis
    # rand_axis = 2
    targets['axis'] = np.array([rand_axis])

    if args.rotation:
        # rand_axis = (np.random.randint(4)+targets['all_atom_mask'][:,0].sum())%4
        rand_axis = idx % 4 
        # rand_axis = torch.randint(4,(1,),generator=g).item()
        # print("rand_axis: ", rand_axis, idx)
    # import pdb;pdb.set_trace()
    if rand_axis <3:
        # import pdb;pdb.set_trace()
        # print("rot with: ", rand_axis)
        all_atom_positions = targets['all_atom_positions']
        all_atom_mask      = targets['all_atom_mask']
        rot = rot_by_axis(rand_axis)
        pos = all_atom_positions[all_atom_mask[:,1]>0]
        cen = pos[:,1].mean(0)
        all_atom_positions = ((all_atom_positions-cen) @ rot + cen)  * all_atom_mask[:,:,None]
        targets['all_atom_positions'] = all_atom_positions
        targets['translation'] = all_atom_positions[:,1]
        targets['axis'] = np.array([rand_axis])
    # elif rand_axis ==4:
    #     all_atom_positions = (all_atom_positions + 50)  * all_atom_mask[:,:,None]                    
    #     targets['template_all_atom_positions'] = all_atom_positions
    #     targets['translation'] = all_atom_positions[:,1]
    # import pdb;pdb.set_trace()


        
    return targets


def get_sseids_at(ss):
    if not isinstance(ss, np.ndarray):
        ss = ss.decode()
    sseid_np = np.array([constants.SSE_TO_ID[s] for s in ss])
    return sseid_np



def get_spatial_cropping_ind(den_dict, batch, cube_width):
    """
    crop residues (backbone) in cube, 
    
    """
    atom_cds = batch['all_atom_positions'][:,1,:,0]
    atom_msk = batch['all_atom_mask'][:,1,0]
    atom_msk = np.tile(atom_msk.reshape(-1,1), (1, 3))

    offset  = den_dict['cryoem_offset'] + den_dict['cryoem_cropidx']
    apix    = den_dict['cryoem_apix']
    if type(offset) == torch.Tensor:
        offset = offset.numpy()
    if type(atom_cds) == torch.Tensor:
        atom_cds = atom_cds.numpy()
    # atom_cds = (atom_cds - np.tile(offset, (len(atom_cds), 1))) * apix
    # import pdb;pdb.set_trace()

    atom_cds = (atom_cds/ apix - offset) 

    atom_cds = atom_cds * atom_msk

    tmp = atom_cds.reshape(-1, 3)
    cds_range = (3.8, cube_width-3.8)
    cds_range = (0, cube_width)

    
    # cropped_ind = np.where(
    #     (tmp.min(axis=1)>=cds_range[0]) & 
    #     (tmp.max(axis=1)<cds_range[1]) &
    #     atom_msk[:,1].astype(np.int64)==1 
    # )[0]
    # import pdb;pdb.set_trace()
    cropped_ind = np.where(
        (tmp.min(axis=1)<cds_range[0]) |
        (tmp.max(axis=1)>cds_range[1])
    )[0]
    
    return cropped_ind

def synmap(feats, args, pdb_chain):
    seq_mask = feats['seq_mask']
    N_res, N_recycle = seq_mask.shape
    dens = cryo_utils.load_density(feats, args)
    if dens is None:
        print(pdb_chain)
        import pdb;pdb.set_trace()
    dens = {
        'cryoem_density' : dens['den_pad'],
        'cryoem_mask' : dens['den_mask'],
        'cryoem_orisize' : dens['den_whlraw'],
        'cryoem_offset': dens['densityinfo'].offset,
        'cryoem_cropidx': np.array([0.,0.,0.]),
        'cryoem_apix': np.array([dens['densityinfo'].apix]),
    }
    # print("density: ", dens['cryoem_density'].shape, dens['cryoem_orisize'])
    save_gt = False
    output_path = f"./predictions/{pdb_chain}_{args.axis}.mrc"
    
    if save_gt:
        ddd = DensityInfo(
            density=dens['cryoem_density'], 
            offset=dens['cryoem_offset'] + dens['cryoem_cropidx'],
            apix=dens['cryoem_apix'][0])
        # ddd = DensityInfo(density=dens['density'], offset=dens['density_offset'], apix=dens['density_apix'])
        ddd.save(output_path)

    seglabel = cryo_utils.segmentation_labels(dens, feats)
    # spatial masking  
    # all_atom_mask, seq_mask, atom14_gt_exists, all_atom_mask_14, 'rigidgroups_gt_exists', 
    # 'rigidgroups_group_exists', pseudo_beta_mask, backbone_rigid_mask, chi_mask
    spatial_mask = get_spatial_cropping_ind(dens, feats, args.cube_width)
    if len(spatial_mask)>0:    
        feats['all_atom_mask'][spatial_mask]=0
        feats['all_atom_mask_14'][spatial_mask]=0
        feats['atom14_gt_exists'][spatial_mask]=0
        feats['rigidgroups_gt_exists'][spatial_mask]=0
        feats['pseudo_beta_mask'][spatial_mask]=0
        feats['backbone_rigid_mask'][spatial_mask]=0
        # feats['chi_mask'][spatial_mask]=0

    # import pdb;pdb.set_trace()
    size = dens['cryoem_density'].shape
    crop_dict = {
        'cryoem_density': torch.from_numpy(dens['cryoem_density'])[None, :,:,:, None].repeat([1,1,1,1, N_recycle]),
        'cryoem_offset':  torch.from_numpy(dens['cryoem_offset'])[:,None].repeat([1,N_recycle]),
        'cryoem_cropidx': torch.from_numpy(dens['cryoem_cropidx'])[:,None].repeat([1,N_recycle]),
        'cryoem_apix':    torch.from_numpy(dens['cryoem_apix'])[:,None].repeat([1,N_recycle]),
        'cryoem_mask':    torch.from_numpy(dens['cryoem_mask'])[None,:,:,:,None].repeat([1,1,1,1, N_recycle]),
        'cryoem_orisize': torch.tensor(dens['cryoem_orisize'])[:,None].float().repeat([1,N_recycle]),
        'cryoem_size':    torch.tensor(size)[:,None].float().repeat([1,N_recycle]),
        'cryoem_seglabel':torch.from_numpy(seglabel)[None,:,:,:,None].float().repeat([1,1,1,1, N_recycle]),
    }
    feats.update(crop_dict)
    del dens, seglabel
    if len(seq_mask.shape) ==2:
        seq_mask = seq_mask[:,0]
        
    seq_mask = seq_mask>0
    # add normed all atom position
    size    = feats['cryoem_orisize']
    offset  = feats['cryoem_offset']
    cropidx = feats['cryoem_cropidx']
    apix    = feats['cryoem_apix']

    ca_pos  = feats['all_atom_positions'][:,1]
    size    = size[None].repeat([N_res, 1,1])   
    offset  = (offset + cropidx)[None].repeat([N_res, 1,1])
    # import pdb;pdb.set_trace()

    normed_ca_pos = ((ca_pos/apix - offset)/size)
    feats['normed_ca_positions'] = normed_ca_pos

    return feats


def add_cryoem(feats, args, pdb_chain):
    # if args.synmap:
    return synmap(feats, args, pdb_chain)
    # else:
        

