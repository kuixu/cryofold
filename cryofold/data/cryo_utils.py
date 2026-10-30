import io,os
import time
import torch
import numpy as np
import pickle
from cryofold.np import constants
from cryofold.np.density import DensityInfo, mol_atom_density


INPUT_LENGTH_DIM_DICT = {
    'aatype': 0, 'between_segment_residues': 0, 'residue_index': 0, 'chain_index': 0, 
    'pos_index': 0, 'seq_length': 0, 'deletion_matrix_int': 1, 'msa': 1, 'num_alignments': 0, 
    'template_aatype': 1, 'template_all_atom_masks': 1, 'template_all_atom_positions': 1, 
    'template_confidence_scores': 1}
GT_LENGTH_DIM_DICT = {
    'template_all_atom_positions': 0, 'template_all_atom_masks': 0, 'template_aatype': 0, 
    'all_atom_positions': 0, 'all_atom_masks': 0, 'aatype': 0, 'sstype': 0, 
    'rotation': 0, 'translation': 0, 'torsion_angles_sin_cos': 0, 'alt_torsion_angles_sin_cos': 0, 
    'torsion_angles_mask': 0, 'chain_index': 0}



def map_pdb_monogroup():
    filepath = "data/monogroup.txt"
    pdb_monogroup_dict = {}
    with open(filepath,'r') as f:
        for line in f.readlines():
            pdbid, monogroup = line.strip().split('\t')
            # print(pdbid, monogroup )
            pdb_monogroup_dict[pdbid] = monogroup.split(';')
    return pdb_monogroup_dict


def map_emdb_pdb():
    map_file = "data/EMDB_PDB.tsv"
    emdb_pdb_dict = {}
    with open(map_file,'r') as f:
        for line in f.readlines():
            emdid, pdbid, res =line.split('\t')
            emdb_pdb_dict[pdbid] = {
                "emdid": emdid,
                "res": res,
                "name": pdbid+"_"+emdid,
            }
    return emdb_pdb_dict


def map_seq_pdb():
    map_file = "data/seq_2_pdb.txt"
    seq_pdb_dict = {}
    with open(map_file,'r') as f:
        for line in f.readlines():
            fields =line.split('\t')
            v = fields[0]
            if 'AFDB:' in v:
                v = v.split('|')[0].replace("AFDB:","")
            for i in range(1,len(fields)):
                k = fields[i].split('=')[0]
                seq_pdb_dict[k] = v
    return seq_pdb_dict

def map_sse_pdb():
    sse_file = "data/PDB_ss.txt"
    # sse_pdb_dict = SeqIO.to_dict(SeqIO.parse(sse_file,"fasta"))
    sse_pdb_dict = pickle.load(open(sse_file+"_id.pkl","rb")) 
    return sse_pdb_dict

def iszero(x):
    assert len(x) ==3
    return x[0]==0 and x[1]==0 and x[2]==0  

def generate_slice_position(w, k=64, overlap=4):
    """
    param:
        w: int, width
        k: int, target size
        overlap: int, overlap region
    return:
        list
    """
    stride = k - overlap
    n = ( w // k) 
    plist = [ stride*i for i in range(n ) ]

    if w - n*stride > overlap:
        plist.append(w - k)
    return plist

def generate_random_crop_position(shape, k=64):
    """
    param:
        shape: (int,int,int)
        k: int, target size
        overlap: int, overlap region`
    return:
        list
    """
    assert len(shape) == 3, print("density shape error, ",shape)
    k  = k - 1
    s1 = [shape[0]-k, shape[1]-k, shape[2]-k]
    #print("s1",s1)
    if s1[0] <1:
        s1[0] = 1
    if s1[1] <1:
        s1[1] = 1
    if s1[2] <1:
        s1[2] = 1
    return [np.random.randint(s1[0]), np.random.randint(s1[1]), np.random.randint(s1[2])]

def get_atom_coords(batch):
    aatype = batch['aatype']
    if len(aatype.shape) ==2:
        # recycling
        aatype = batch['aatype'][:,0]
    coords14 = batch['all_atom_positions_14']
    mask14 = batch['all_atom_mask_14']
    if isinstance(aatype, torch.Tensor):
        aatype = aatype.numpy().squeeze()
    if isinstance(coords14, torch.Tensor):
        coords14 = coords14.numpy()
    if isinstance(mask14, torch.Tensor):
        mask14 = mask14.numpy()
    if len(coords14.shape)==4:
        coords14 = coords14[:,:,:,0]
    if len(mask14.shape)==4:
        mask14 = mask14[:,:,:,0]
    atom_coords = []
    atom_elems  = []
    for i in range(coords14.shape[0]):
        for j in range(14):
            if mask14[i,j,0] and not iszero(coords14[i,j]):
                atom_coords.append(coords14[i,j])
                restype = aatype[i]
                # print(i,j)
                # import pdb;pdb.set_trace()
                atom_elems.append(constants.restype_atom14_elem[restype][j])
    atom_coords = np.array(atom_coords)
    atom_weight = [constants.ATOM_WEIGHT[e][0] for e in atom_elems]
    # atom_weight = [ATOM_WEIGHT['C'][0] for e in atom_coords]
    # density, offset = mol_atom_density(atom_coords, atom_weight, res, voxel_size)
    # den = DensityInfo(density=density, offset=offset, apix=voxel_size)
    # if atom_coords.shape[0]==0:
    # import pdb;pdb.set_trace()
    return atom_coords, atom_weight

def get_monomer(dens, batch, args):

    den = dens['densityinfo']
    density = dens['den_pad']
    mask =  dens['den_mask']

    atom_coords, atom_weight = get_atom_coords(batch)

    res=8.0
    apix=1.0

    mol_density, offset = mol_atom_density(atom_coords, atom_weight, res, apix)


    thr = mol_density.mean()
    # print("thr: ", thr)
    mol_density[mol_density<thr]  = 0.0
    mol_density[mol_density>=thr] = 1.0
    mol_den = DensityInfo(density=mol_density, offset=offset, apix=apix)
    
    den.to_torch()
    msk_density, msk_offset = den.mask(mol_den)
    msk_den = DensityInfo(density=msk_density, offset=msk_offset, apix=apix)


    den_pad,  den_mask = msk_den.pad(args.cube_width, return_mask=True, 
        no_padding=args.no_padding, scale_factor=4)
    
    den_dict = {
        'densityinfo': msk_den,
        'den_pad': den_pad,
        'den_mask': den_mask,
    }
    return den_dict

def load_density(batch, args):
    path = args.map_path
    print("loading map: ", path)
    mol_den = DensityInfo(path, datatype="numpy")
    mol_den.scale(args.apix)
    res = args.resolution
    mol_den.norm()
    if args.inference:
        args.no_padding = True
        args.cube_width = max(mol_den.density.shape)
        den_pad,  den_mask = mol_den.pad(args.cube_width, return_mask=True, 
            no_padding=args.no_padding, scale_factor=4)
        # import pdb;pdb.set_trace()

        den_whlraw = np.array(den_pad.shape) 
        
        den_dict = {
            'densityinfo': mol_den,
            'den_pad': den_pad,
            'den_mask': den_mask,
            'den_res': res,
            'den_whlraw': den_whlraw,
        }
    else:
        # print("mol_density: ", mol_density.shape)
        den_pad,  den_mask = mol_den.pad(args.cube_width, return_mask=True, 
            no_padding=args.no_padding, scale_factor=4)
        # if max(mol_density.shape)>128:
        #     import pdb;pdb.set_trace()

        den_whlraw = np.clip(np.array(den_pad.shape),8,args.cube_width)

        
        den_dict = {
            'densityinfo': mol_den,
            'den_pad': den_pad[:args.cube_width, :args.cube_width, :args.cube_width, ],
            'den_mask': den_mask[:args.cube_width, :args.cube_width, :args.cube_width, ],
            'den_res': res,
            'den_whlraw': den_whlraw,
        }
        

    return den_dict

def padding_density(cubes, target_shape, value=0.0):
    # if isinstance(cubes, np.ndarray):
    m = np.array(target_shape) - cubes.shape
    if m[0] > 0:
        b = np.zeros([m[0], cubes.shape[1], cubes.shape[2]]) + value
        cubes = np.concatenate((cubes, b), axis=0)
    if m[1] > 0:
        b = np.zeros([cubes.shape[0], m[1], cubes.shape[2]]) + value
        cubes = np.concatenate((cubes, b), axis=1)
    if m[2] > 0:
        b = np.zeros([cubes.shape[0], cubes.shape[1], m[2]]) + value
        cubes = np.concatenate((cubes, b), axis=2)
    # elif isinstance(cubes, torch.Tensor):
    #     m = (torch.tensor(target_shape) - torch.tensor(cubes.shape)).int()
    #     if m[0] > 0:
    #         b = torch.zeros([m[0], cubes.shape[1], cubes.shape[2]]) + value
    #         cubes = torch.cat((cubes, b), dim=0)
    #     if m[1] > 0:
    #         b = torch.zeros([cubes.shape[0], m[1], cubes.shape[2]]) + value
    #         cubes = torch.cat((cubes, b), dim=1)
    #     if m[2] > 0:
    #         b = torch.zeros([cubes.shape[0], cubes.shape[1], m[2]]) + value
    #         cubes = torch.cat((cubes, b), dim=2)
    # else:
    #     raise "wrong data type"
    return cubes

def filter_inputs_by_seq_idx(inputs, seq_indices):
    for key in inputs:
        if key in INPUT_LENGTH_DIM_DICT:
            dim = INPUT_LENGTH_DIM_DICT[key]
            # print(key, inputs[key].shape, len(seq_indices))
            # print(seq_indices)
            if inputs[key].ndim > 0 and inputs[key].shape[0] > 0:
                inputs[key] = np.take(inputs[key], seq_indices, axis=dim)
        if not isinstance(inputs[key], np.ndarray):
            # inputs[key] = "".join(np.array(list(inputs[key].decode()))[seq_mask]).encode()
            continue
    return inputs

def filter_targets_by_seq_idx(targets, seq_indices):
    for key in targets:
        if key in GT_LENGTH_DIM_DICT:
            dim = GT_LENGTH_DIM_DICT[key]
            if targets[key].ndim > 0 and targets[key].shape[0] > 0:
                targets[key] = np.take(targets[key], seq_indices, axis=dim)
        if not isinstance(targets[key], np.ndarray):
            # targets_raw[key] = "".join(np.array(list(targets[key].decode()))[seq_mask]).encode()
            continue
    return targets
    
def get_spatial_cropping_ind(den_dict, batch, cube_width):
        """
        crop residues (backbone) in cube, 
        
        """
        atom_cds = batch['template_all_atom_positions'][:,1]
        atom_msk = batch['template_all_atom_masks'][:,1]
        atom_msk = np.tile(atom_msk.reshape(-1,1), (1, 3))

        offset  = den_dict['density_offset'] + den_dict['density_cropidx']
        apix    = den_dict['density_apix']
        # width   = den_dict['density_whl']
        if type(offset) == torch.Tensor:
            offset = offset.numpy()

        # atom_cds = (atom_cds - np.tile(offset, (len(atom_cds), 1))) * apix
        atom_cds = (atom_cds/ apix - offset) 

        atom_cds = atom_cds * atom_msk

        tmp = atom_cds.reshape(-1, 3)
        cds_range = (3.8, cube_width-3.8)

        # bbb = batch['all_atom_positions'][:,1]
        # print("map offset: ", offset)
        # print("cds min: ", bbb[:,].min(),bbb[:,1].min(),bbb[:,2].min())
        # print("cds max: ", bbb[:,].max(),bbb[:,1].max(),bbb[:,2].max())
        # print("atoms crop: ", tmp.shape)
        # print("crp min max: ", tmp.min(), tmp.max())
        # print("atoms crop: ", (tmp.min(axis=1)>=cds_range[0]))
        # print("atoms crop: ", (tmp.max(axis=1)<cds_range[1]))
        # print("atoms crop: ", atom_msk[:,1].astype(np.int64)==1)
        cropped_ind = np.where(
            (tmp.min(axis=1)>=cds_range[0]) & 
            (tmp.max(axis=1)<cds_range[1]) &
            atom_msk[:,1].astype(np.int64)==1 
        )[0]
        # import pdb;pdb.set_trace()
        # cropped_ind = np.where(
        #     (tmp.min(axis=1)>=cds_range[0]) & 
        #     (tmp.max(axis=1)<cds_range[1]) & 
        #     atom_msk[:,1].astype(np.int64)==1 
        # )[0]
        return cropped_ind

def random_cropping(den_dict, cube_width, label=None, no_crop=False):
    """
    random cropping density 
    args:
        density: np.float (W, H, L)
        cube_width: int 
    """
    den = den_dict['densityinfo']
    density = den_dict['den_pad']
    mask =  den_dict['den_mask']
    shape = density.shape
    if no_crop:
        r = (0,0,0)
    else:
        r = generate_random_crop_position(shape, cube_width)
    # print(shape, r)
    k = cube_width
    den_crop = density[r[0]:r[0]+k, r[1]:r[1]+k, r[2]:r[2]+k]
    msk_crop = mask[r[0]:r[0]+k, r[1]:r[1]+k, r[2]:r[2]+k]

    cropidx = np.array(r)
    if isinstance(den.apix, float):
        density_apix = np.array(den.apix)
    else:
        density_apix = den.apix
    density_whl = np.array(den_crop.shape)
    box_0 = den.offset + cropidx
    box_1 = den.offset + cropidx + density_whl
    density_box = np.stack([box_0, box_1]).astype(int) 
    crop_dict = {
        'density' : den_crop,
        'density_offset': den.offset,
        'density_cropidx': cropidx,
        'density_apix': density_apix,
        'density_mask': msk_crop,
        'density_whl': density_whl,
        'density_box': density_box,
        'density_num': np.array(1),
    }
    if label is not None:
        lab_crop = label[r[0]:r[0]+k, r[1]:r[1]+k, r[2]:r[2]+k]
        crop_dict.update({"density_aa": lab_crop})
    return crop_dict



def slice_cropping(den_dict, cube_width, label=None):
    """
    random cropping density 
    args:
        density: np.float (W, H, L)
        cube_width: int 
    """
    k = cube_width

    den = den_dict['densityinfo']
    density = den_dict['den_pad']
    mask =  den_dict['den_mask']
    # density = padding_density(density, target_shape=(k,k,k), value=density.min())
    # print("padding_density: ", density.shape)
    s = density.shape

    l0 = generate_slice_position(s[0], cube_width)
    l1 = generate_slice_position(s[1], cube_width)
    l2 = generate_slice_position(s[2], cube_width)

    # thr = density.mean()+density.std()
    # print(l0)
    # print(l1)
    # print(l2)
    den_crops = []
    cropidxes = []
    den_boxes = []
    msk_crops = []
    den_num = 0
    for x in l0:
        for y in l1:
            for z in l2:
                r = [x, y, z]

                den_crop = density[r[0]:r[0]+k, r[1]:r[1]+k, r[2]:r[2]+k]
                msk_crop = mask[r[0]:r[0]+k, r[1]:r[1]+k, r[2]:r[2]+k]
                if den_crop.max() == den_crop.min() or np.isclose(den_crop,0.0).all():
                    continue
                cropidx = np.array(r)
                density_whl = np.array(den_crop.shape)
                box_0 = den.offset + cropidx
                box_1 = den.offset + cropidx + density_whl
                density_box = np.stack([box_0, box_1]).astype(int) 

                den_crops.append(den_crop)
                msk_crops.append(msk_crop)
                cropidxes.append(cropidx)
                den_boxes.append(density_box)
                den_num +=1

    if isinstance(den.apix, float):
        density_apix = np.array(den.apix)
    else:
        density_apix = den.apix
    
    crop_dict = {
        'density' : np.stack(den_crops),
        'density_offset': den.offset,
        'density_cropidx': np.stack(cropidxes),
        'density_apix': density_apix,
        'density_mask': np.stack(msk_crops),
        'density_whl': density_whl,
        'density_box': np.stack(den_boxes),
        'density_num': np.array(den_num),

    }
    if label is not None:
        lab_crop = label[r[0]:r[0]+k, r[1]:r[1]+k, r[2]:r[2]+k]
        crop_dict.update({"density_aa": lab_crop})
    return crop_dict


def segmentation_labels(dens, targets, infer=False):

    density  = dens['cryoem_density']
    den_apix = dens['cryoem_apix']
    den_off  = dens['cryoem_offset']
    cropidx  = dens['cryoem_cropidx']
    den_off  = den_off + cropidx
    # if infer:
    #     return np.zeros_like(density) + 23
    seq_mask = targets['seq_mask']
    aatype   = targets['aatype']
    coords14 = targets['atom14_gt_positions']
    mask14   = targets['atom14_gt_exists']
    if len(targets['seq_mask'].shape) ==2:
        seq_mask = seq_mask[:,0]
        aatype = aatype[:,0]
        coords14 = coords14[...,0]
        mask14   = mask14[...,0]
    aatype   = aatype[seq_mask>0]
    coords14 = coords14[seq_mask>0]
    mask14   = mask14[seq_mask>0]
    if isinstance(aatype, torch.Tensor):
        aatype = aatype.numpy()
    if seq_mask.sum()==0:
        print("warning: zero seq_mask ")
        return None
    if targets['atom14_gt_exists'][:,1].sum()==0:
        print("warning: zero gt_exists ")
        return None
    # import pdb;pdb.set_trace()
    
    if isinstance(coords14, torch.Tensor):
        coords14 = coords14.numpy()
    # print("atom14_gt_exists:", batchY['atom14_gt_exists'].shape)
    masks = np.tile(mask14[:,:,None], (1,1,3))
    if isinstance(masks, torch.Tensor):
        masks = masks.numpy()
    N_res,N_A,_ = coords14.shape
    coords14 = coords14 / den_apix - np.tile(den_off.reshape(1,1,3), (N_res,N_A,1))
    coords14 = coords14*masks
    
    xx,yy,zz = np.meshgrid(np.arange(-1,2),np.arange(-1,2),np.arange(-1,2))
    grid = np.concatenate((xx.reshape(-1,1), yy.reshape(-1,1), zz.reshape(-1,1)),1)
    atom_index = []
    atom_class = []
    for i in range(N_res):
        for j in range(14):
            if not iszero(coords14[i,j]):
                atm_ind = np.round(coords14[i,j]).astype(int)
                atom_index.extend(atm_ind+grid)
                
                # import pdb;pdb.set_trace()
                # atom_class.extend([aatype[i],aatype[i],aatype[i]])
                atom_class.extend([aatype[i]]*len(grid))
    atom_index = np.array(atom_index)
    atom_class = np.array(atom_class)
    if atom_index.shape[0]==0:
        print("warning: zero atom_index ")
        return None

    map_labeled = np.zeros_like(density) +23
    sz = map_labeled.shape
    # pos_mask = ((forward_cumsum > 5) & (backward_cumsum > 5))
    keep = np.where((atom_index[:,0]<sz[0]) & (atom_index[:,1]<sz[1]) & (atom_index[:,2]<sz[2]))[0]
    atom_index = atom_index[keep]
    atom_class = atom_class[keep]
    # print("atom_index:",atom_index.shape)

    try:
        map_labeled[atom_index[:,0],atom_index[:,1],atom_index[:,2]] = atom_class
    except Exception:
        import pdb;pdb.set_trace()
    # import pdb;pdb.set_trace()

    return map_labeled

def label_density_with_aa(dens, targets, infer=False):

    density  = dens['density']
    den_apix = dens['density_apix']
    den_off  = dens['density_offset']
    cropidx  = dens['density_cropidx']
    den_off  = den_off + cropidx
    # if infer:
    #     return np.zeros_like(density) + 23
    seq_mask = targets['seq_mask']
    aatype   = targets['aatype']
    coords14 = targets['atom14_gt_positions']
    mask14   = targets['atom14_gt_exists']
    if len(targets['seq_mask'].shape) ==2:
        seq_mask = seq_mask[:,0]
        aatype = aatype[:,0]
        coords14 = coords14[...,0]
        mask14   = mask14[...,0]
    aatype   = aatype[seq_mask>0]
    coords14 = coords14[seq_mask>0]
    mask14   = mask14[seq_mask>0]
    if isinstance(aatype, torch.Tensor):
        aatype = aatype.numpy()
    if seq_mask.sum()==0:
        print("warning: zero seq_mask ")
        return None
    if targets['atom14_gt_exists'][:,1].sum()==0:
        print("warning: zero gt_exists ")
        return None
    # import pdb;pdb.set_trace()
    
    if isinstance(coords14, torch.Tensor):
        coords14 = coords14.numpy()
    # print("atom14_gt_exists:", batchY['atom14_gt_exists'].shape)
    masks = np.tile(mask14[:,:,None], (1,1,3))
    if isinstance(masks, torch.Tensor):
        masks = masks.numpy()
    N_res,N_A,_ = coords14.shape
    coords14 = coords14 / den_apix - np.tile(den_off.reshape(1,1,3), (N_res,N_A,1))
    coords14 = coords14*masks
    
    xx,yy,zz = np.meshgrid(np.arange(-1,2),np.arange(-1,2),np.arange(-1,2))
    grid = np.concatenate((xx.reshape(-1,1), yy.reshape(-1,1), zz.reshape(-1,1)),1)
    atom_index = []
    atom_class = []
    for i in range(N_res):
        for j in range(14):
            if not iszero(coords14[i,j]):
                atm_ind = np.round(coords14[i,j]).astype(int)
                atom_index.extend(atm_ind+grid)
                
                # import pdb;pdb.set_trace()
                # atom_class.extend([aatype[i],aatype[i],aatype[i]])
                atom_class.extend([aatype[i]]*len(grid))
    atom_index = np.array(atom_index)
    atom_class = np.array(atom_class)
    if atom_index.shape[0]==0:
        print("warning: zero atom_index ")
        return None

    map_labeled = np.zeros_like(density) +23
    sz = map_labeled.shape
    # pos_mask = ((forward_cumsum > 5) & (backward_cumsum > 5))
    keep = np.where((atom_index[:,0]<sz[0]) & (atom_index[:,1]<sz[1]) & (atom_index[:,2]<sz[2]))[0]
    atom_index = atom_index[keep]
    atom_class = atom_class[keep]
    # print("atom_index:",atom_index.shape)

    try:
        map_labeled[atom_index[:,0],atom_index[:,1],atom_index[:,2]] = atom_class
    except Exception:
        import pdb;pdb.set_trace()
    # import pdb;pdb.set_trace()

    return map_labeled

def label_density_with_aa0(density, den_off, den_apix, batchY, infer=False):
    if infer:
        return np.zeros_like(density) + 23
    seq_mask = batchY['seq_mask']
    if seq_mask.sum()==0:
        print("warning: zero seq_mask ")
        return None
    if batchY['atom14_gt_exists'][:,1].sum()==0:
        print("warning: zero gt_exists ")
        return None
    coords14 = batchY['atom14_gt_positions'][seq_mask>0]
    # print("atom14_gt_exists:", batchY['atom14_gt_exists'].shape)
    masks = np.tile(batchY['atom14_gt_exists'][seq_mask>0][:,:,None], (1,1,3))

    N_res,N_A,_ = coords14.shape
    coords14 = coords14 / den_apix - np.tile(den_off.reshape(1,1,3), (N_res,N_A,1))
    coords14 = coords14*masks
    aatype = batchY['aatype'][seq_mask>0]
    atom_index = []
    atom_class = []
    for i in range(N_res):
        for j in range(14):
            if not iszero(coords14[i,j]):
                atm_ind = np.round(coords14[i,j]).astype(int)
                atom_index.append(atm_ind)
                atom_index.append(atm_ind+1)
                atom_index.append(atm_ind-1)
                atom_class.extend([aatype[i],aatype[i],aatype[i]])
    atom_index = np.array(atom_index)
    atom_class = np.array(atom_class)
    if atom_index.shape[0]==0:
        print("warning: zero atom_index ")
        return None

    map_labeled = np.zeros_like(density) + 23
    
    # print("atom_index:",atom_index.shape)
    map_labeled[atom_index[:,0],atom_index[:,1],atom_index[:,2]] = atom_class

    return map_labeled

def fragmentation(resid, chain, num_res=300, return_link=False):
    posid  = np.ones_like(resid)
    fragid = np.ones_like(resid)
    if return_link:
        link   = np.zeros((len(resid), len(resid)))
    else:
        link = None
    i_frag = 1
    i_pos  = 0
    for i in range(len(resid)):
        
        if i ==0:
            p_res = resid[0]
            p_cha = chain[0]
        else:
            p_res = i_res
            p_cha = i_cha
        i_res = resid[i]
        i_cha = chain[i]

        if i_cha != p_cha and i_res - p_res != 1:
            i_frag += 1
            i_pos = 1
        else:
            i_pos += 1
            if return_link:
                if i -1 >=0:
                    link[i,i-1] = 1
                    link[i-1,i] = 1

        fragid[i] = i_frag
        posid[i]  = i_pos
    posid_final  = np.ones(num_res)
    fragid_final = np.ones(num_res)
    used_num_res = min(num_res, len(resid))
    posid_final[:used_num_res]  = posid[:used_num_res]
    fragid_final[:used_num_res] = fragid[:used_num_res]
    return posid_final, fragid_final, link
    
        
