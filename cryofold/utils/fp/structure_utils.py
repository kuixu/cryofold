import warnings
import sys,os
import math
import random
from pathlib import Path
import numpy as np

# from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_cluster import radius_graph, knn_graph



from cryofold.libs import PDB
from cryofold.libs.PDB import StructureBuilder
from cryofold.libs.PDB.PDBIO import PDBIO

# from cryofold.libs import PDB

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import PDBxFile, get_structure
from cryofold.utils.fp.constants import three_to_one_letter, letter_to_num, max_num_heavy_atoms, \
    restype_to_heavyatom_names, heavyatom_to_label, chi_alt_truths, num_to_letter, \
    chi_true_indices, chi_mask, atom_types, atom_type_num, ATOM14_MASK_NP, ATOM37_MASK_NP,\
    one_to_three_letter

from cryofold.utils.fp.sidechain_utils import get_bb_dihedral, get_chi_angles
# from tqdm.auto import tqdm
# import pandas as pd


def create_structure_from_crds(aa,crds,atom_mask,chain_id=None,outPath="test.pdb", save_traj=False):
    warnings.filterwarnings("ignore", ".*Used element.*")
    structure_builder = StructureBuilder.StructureBuilder()
    structure_builder.init_structure(0)
    if save_traj:
        assert len(crds.shape) == 4
        for model_idx in range(crds.shape[0]):
            structure_builder.init_model(model_idx)
            structure_builder.init_chain("A")
            structure_builder.init_seg(' ')

            for res_idx, res in enumerate(aa):
                aa_str = one_to_three_letter[res]
                structure_builder.init_residue(aa_str, " ", res_idx + 1, " ")
                for i, atom_name in enumerate(restype_to_heavyatom_names[one_to_three_letter[res]]):
                    if atom_name == '': continue
                    if not atom_mask[res_idx, i]: continue
                    if len(atom_name) == 1:
                        fullname = f' {atom_name}  '
                    elif len(atom_name) == 2:
                        fullname = f' {atom_name} '
                    elif len(atom_name) == 3:
                        fullname = f' {atom_name}'
                    else:
                        fullname = atom_name  # len == 4
                    structure_builder.init_atom(name=atom_name, coord=crds[model_idx, res_idx, i], b_factor=res_idx + 1.0,
                                                occupancy=1.0, altloc=" ", fullname=fullname)
    else:
        structure_builder.init_model(0)
        if chain_id is not None:
            unique_chains = np.unique(chain_id).tolist()
            if len(unique_chains) == 1 and unique_chains[0] == '':
                unique_chains = ['A']
                chain_id = np.array(['A'] * len(aa))
        else:
            chain_id = np.array(['A'] * len(aa))
            unique_chains = ['A']

        for i in unique_chains:
            structure_builder.init_chain(i)
            structure_builder.init_seg(' ')

            aa_chain = np.array(list(aa))[chain_id==i]
            crds_chain = crds[chain_id==i]
            atom_mask_chain = atom_mask[chain_id==i]

            for res_idx, res in enumerate(aa_chain):
                # print("one_to_three_letter: ",type(one_to_three_letter), type(res), res, aa_chain)
                # import pdb; pdb.set_trace()
                aa_str = one_to_three_letter[res]
                structure_builder.init_residue(aa_str," ",res_idx+1," ")
                for i,atom_name in enumerate(restype_to_heavyatom_names[one_to_three_letter[res]]):
                    if atom_name == '': continue
                    if not atom_mask_chain[res_idx, i]: continue
                    if len(atom_name) == 1:
                        fullname = f' {atom_name}  '
                    elif len(atom_name) == 2:
                        fullname = f' {atom_name} '
                    elif len(atom_name) == 3:
                        fullname = f' {atom_name}'
                    else:
                        fullname = atom_name  # len == 4
                    structure_builder.init_atom(name=atom_name,coord=crds_chain[res_idx,i],b_factor=res_idx+1.0,occupancy=1.0,altloc=" ",fullname=fullname)

    st = structure_builder.get_structure()
    io = PDBIO()
    io.set_structure(st)
    io.save(outPath)

# Atomic radii for various atom types.
# You can comment out the ones you don't care about or add new ones
atom_radii = {
#    "H": 1.20,  # Who cares about hydrogen??
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    # "F": 1.47,
    # "P": 1.80,
    # "CL": 1.75,
    # "MG": 1.73,
}

def count_clashes(path, clash_cutoff=0.4):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(0, path)

    # Set what we count as a clash for each pair of atoms
    clash_cutoffs = {i + "_" + j: (clash_cutoff * (atom_radii[i] + atom_radii[j])) for i in atom_radii for j in atom_radii}
    # Extract atoms for which we have a radii
    atoms = [x for x in structure.get_atoms() if x.element in atom_radii]
    coords = np.array([a.coord for a in atoms], dtype="d")
    # Build a KDTree (speedy!!!)
    kdt = PDB.kdtrees.KDTree(coords)
    # Initialize a list to hold clashes
    clashes = []
    # Iterate through all atoms
    for atom_1 in atoms:
        # Find atoms that could be clashing
        kdt_search = kdt.search(np.array(atom_1.coord, dtype="d"), max(clash_cutoffs.values()))
        # Get index and distance of potential clashes
        potential_clash = [(a.index, a.radius) for a in kdt_search]
        for ix, atom_distance in potential_clash:
            atom_2 = atoms[ix]
            # Exclude clashes from atoms in the same residue
            if atom_1.parent.id == atom_2.parent.id:
                continue
            # Exclude clashes from peptide bonds
            elif (atom_2.name == "C" and atom_1.name == "N") or (atom_2.name == "N" and atom_1.name == "C"):
                continue
            # Exclude clashes from disulphide bridges
            elif (atom_2.name == "SG" and atom_1.name == "SG") and atom_distance > 1.88:
                continue
            if atom_distance < clash_cutoffs[atom_2.element + "_" + atom_1.element]:
                clashes.append((atom_1, atom_2))
    return len(clashes) // 2




min_length=40
max_length=512
edge_type='radius'
max_radius=8.0
max_num_neighbors=30
scale_coords=1.0
filter_length=True

def to_tensor(d, exclude=[]):
    feat_dtypes = {
        "coord": torch.float32,
        "atom_type": torch.long,
        "aa": None,
        "mask": torch.long,
        "atom_mask": torch.long,
        "atom_mask_gt": torch.long,
        "chain_id": None,
    }

    for x in exclude:
        del d[x]
    # breakpoint()
    for k,v in d.items():
        if feat_dtypes[k] is not None:
            d[k] = torch.tensor(v).to(dtype=feat_dtypes[k])

    return d

def get_features(path):
    path = Path(path)
    # try:
    if path.suffix == ".cif":
        with open(path, "r") as f:
            structure = PDBxFile.read(f)
            structure = get_structure(structure)
    else:
        with open(path, "r") as f:
            structure = PDBFile.read(f)
            structure = structure.get_structure()
            if len(structure) > 1: 
                structure = structure[0]
    # except:
    #     print("Error reading file: ", path)
    #     return None

    # if struc.get_chain_count(structure) > 1: return None # only single chains

    # breakpoint()
    _, aa = struc.get_residues(structure)
    # Replace nonstandard amino acids with X
    for idx, a in enumerate(aa):
        if a not in three_to_one_letter.keys():
            aa[idx] = 'UNK'

    aa_str = [three_to_one_letter.get(i,'X') for i in aa]
    aa_num = [letter_to_num[i] for i in aa_str]
    atom14_mask_gt = np.array([ ATOM14_MASK_NP[i] for i in aa_num])
    atom37_mask_gt = np.array([ ATOM37_MASK_NP[i] for i in aa_num])

    # aa_ids = 
    # if len(aa_str) > self.max_length or len(aa_str) < self.min_length:
    #     return None
    # if len(aa_str) < self.min_length: return None

    aa_mask = np.ones(len(aa))
    atom14_mask = np.zeros((len(aa), max_num_heavy_atoms))
    atom37_mask = np.zeros((len(aa), atom_type_num))
    # Iterate through all residues
    coords, coords37, atom_type, chain_ids = [], [], [], []
    for res_idx, res in enumerate(struc.residue_iter(structure)):
        res_coords = res.coord[0]
        res_name = aa[res_idx]
        chain_ids.append(res.chain_id[0])

        if res_name == "UNK":
            aa_mask[res_idx] = 0

        # Append true coords
        res_crd14 = np.zeros((max_num_heavy_atoms, 3))
        res_crd37 = np.zeros((atom_type_num, 3))
        res_atom_type = []
        for atom14_idx, r in enumerate(restype_to_heavyatom_names[res_name]):
            if r == '':
                res_atom_type.append(4)
                continue
            atom37_idx = atom_types.index(r)
            res_atom_type.append(heavyatom_to_label[r[0]])
            i = np.where(res.atom_name == r)[0]
            if i.size == 0:
                res_crd14[atom14_idx] = 0
                res_crd37[atom37_idx] = 0

            else:
                res_crd14[atom14_idx] = res_coords[i[0]]
                atom14_mask[res_idx, atom14_idx] = 1
                res_crd37[atom37_idx] = res_coords[i[0]]
                atom37_mask[res_idx, atom37_idx] = 1
        coords.append(res_crd14)
        coords37.append(res_crd37)
        atom_type.append(res_atom_type)

    coords = np.array(coords)
    atom_type = np.array(atom_type)
    aa_num = np.array(aa_num)
    chain_ids = np.array(chain_ids)

    assert len(coords) == len(aa_num)
    # breakpoint()
    return {
        "coord": coords,
        "atom_type": atom_type,
        "aa": aa_str,
        "mask": aa_mask,
        "atom_mask": atom14_mask,
        "atom_mask_gt": atom14_mask_gt,
        "chain_id": chain_ids
    }


def get_feature_pdb(path):
    pdb_id = os.path.basename(path).split('.')[0]
    structure = to_tensor(get_features(path))
    # elif self.data_path.joinpath(f'{pdb_id}.cif').exists():
    # structure = to_tensor(self.get_features(self.data_path.joinpath(f'{pdb_id}.cif')))
    # else:
    #     raise FileNotFoundError(f'{pdb_id} pdb file not found')

    coords = structure['coord']
    aa_str = structure['aa']
    atom_mask = structure['atom_mask']
    atom_mask_gt = structure['atom_mask_gt']
    aa_mask = structure['mask']
    aa_num = torch.LongTensor([letter_to_num.get(i, 20) for i in aa_str])
    atom_type = structure['atom_type']
    chain_id = structure['chain_id']

    # There seems to be an issue with some pdbs where missing coordinates are just duplicated as previous residue's
    # coordinates - remove them since they dont work with equiformerv2 when computing edge_vec
    pairwise_dist = torch.cdist(coords[:,1], coords[:,1])
    x,y = torch.triu_indices(len(coords),len(coords))
    pairwise_dist[x,y] = 9999
    x, y = torch.nonzero(pairwise_dist < 0.01, as_tuple=True)
    if (x != y).sum() > 0:
        duplicate_mask = torch.ones(coords.shape[0], dtype=bool)
        duplicate_mask[x[x != y]] = 0
        coords = coords[duplicate_mask]
        aa_num = aa_num[duplicate_mask]
        atom_mask = atom_mask[duplicate_mask]
        # atom_mask_gt = atom_mask_gt[duplicate_mask]
        aa_str = ''.join([num_to_letter[i.item()] for i in aa_num])
        aa_mask = aa_mask[duplicate_mask]
        atom_type = atom_type[duplicate_mask]
        chain_id = chain_id[duplicate_mask]

    origin = coords[:,:4].reshape(-1, 3).mean(0) # CoM of backbone atoms
    coords = (coords - origin.unsqueeze(0)) * atom_mask.unsqueeze(-1)

    # aa to tensor
    seq_onehot = F.one_hot(aa_num,num_classes=21).float()

    if filter_length:
        if len(coords) > max_length:
            rand_start = random.randint(0, len(coords) - max_length)
            rand_end = rand_start + max_length
            coords = coords[rand_start:rand_end]
            aa_num = aa_num[rand_start:rand_end]
            aa_str = aa_str[rand_start:rand_end]
            atom_mask = atom_mask[rand_start:rand_end]
            # atom_mask_gt = atom_mask_gt[rand_start:rand_end]
            aa_mask = aa_mask[rand_start:rand_end]
            seq_onehot = seq_onehot[rand_start:rand_end]
            atom_type = atom_type[rand_start:rand_end]
            chain_id = chain_id[rand_start:rand_end]

    bb_dihedral = get_bb_dihedral(coords[:,0], coords[:,1], coords[:,2])
    chi_angles, chi_mask = get_chi_angles(aa_num, coords, atom_mask)
    # chi_angles, chi_mask = get_chi_angles(aa_num, coords, atom_mask_gt)
    chi_alt_mask = chi_alt_truths[aa_num] == 1
    chi_angles[chi_alt_mask] = ((chi_angles[chi_alt_mask] + math.pi) % math.pi) - math.pi
    chi_alt_angles = chi_angles.clone()
    # first move to [0, 2pi] and then add pi and then back to [-pi, pi] - this seems unnecessarily convoluted
    chi_alt_angles[chi_alt_mask] = ((chi_angles[chi_alt_mask] + (2*math.pi)) % (2*math.pi)) - math.pi

    # mask unknown residues
    chi_mask = chi_mask * aa_mask.unsqueeze(-1)
    chi_angles = chi_angles * chi_mask
    chi_alt_angles = chi_alt_angles * chi_mask

    # edge index
    ca = coords[:,1]
    if edge_type == 'radius':
        edge_index = radius_graph(ca, r=max_radius, max_num_neighbors=max_num_neighbors)
    elif edge_type == 'knn':
        edge_index = knn_graph(ca, k=max_num_neighbors)
    else: raise NotImplementedError('wrong edge type')

    edge_feat = None

    data = Data(edge_index=edge_index, aa_str=aa_str, aa_num=aa_num, aa_onehot=seq_onehot, id=pdb_id,
                pos=coords, edge_attr=edge_feat, aa_mask=aa_mask, bb_dihedral=bb_dihedral, chi=chi_angles,
                chi_alt=chi_alt_angles, chi_mask=chi_mask, atom_mask=atom_mask, atom_mask_gt=atom_mask_gt, chi_alt_mask=chi_alt_mask,
                atom_type=atom_type, chain_id=chain_id)

    return data



def get_edge_features(X, edge_index, atom_mask=None, all_atoms=False, chain_index=None):
    edge_src, edge_dst = edge_index
    edge_feat = []
    relpos = torch.clamp(edge_src - edge_dst, min=-32, max=32) + 32

    # if chain_index is not None:



    relpos = F.one_hot(relpos, num_classes=65).float()
    edge_feat.append(relpos)

    if all_atoms and atom_mask is not None:
        X_src = X[edge_src]
        X_dst = X[edge_dst]
        mask_src = atom_mask[edge_src]
        mask_dst = atom_mask[edge_dst]

        dist = torch.cdist(X_src, X_dst) * mask_src.unsqueeze(-1) * mask_dst.unsqueeze(-2)
        dist = dist.view(-1, 196).clamp(max=12.)

        edge_feat.append(dist)

    edge_feat = torch.cat(edge_feat, dim=-1)
    return edge_feat



if __name__ == "__main__":
    path = sys.argv[1]
    clashes = count_clashes(path)
    print(f"Number of clashes: {clashes}")