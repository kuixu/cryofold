# Copyright (c) CryoNet Team, and its affiliates. All Rights Reserved

"""A class for creating PDB files/strings given a protein's sequence and coordinates."""

import numpy as np
import time,os,json,string
from scipy.spatial import distance
import torch
import pathlib
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
from cryofold.libs.structure.hy36 import encode_atom_id, encode_chain_id
from cryofold.libs.PDB import PDBParser
from cryofold.libs.PDB.SSE import get_sse_records_by_class
from cryofold.utils.cn.boxes3d import Boxes
# from cryonet.model.constants import restypes_with_x, restype_name_to_atom14_names, restype_1to3, restype_3to1
from cryofold.np.residue_constants import restypes_with_x, restype_name_to_atom14_names, restype_1to3, restype_3to1

ATOM_MAP_14 = {}
restype_1to3['X'] = 'UNK'
for one_letter in restype_1to3.keys():
    # sc_atoms = [a for a in sc_atoms_all if a!='' ]
    ATOM_MAP_14[one_letter] = restype_name_to_atom14_names[restype_1to3[one_letter]]


def seqids_to_seq(seqids):
    if isinstance(seqids, np.ndarray):
        seq = "".join([restypes_with_x[id] for id in seqids.astype(int) ])
    else:
        seq = "".join([restypes_with_x[id] for id in seqids.int() ])

    return seq

class PdbBuilder(object):
    """Creates a PDB file given a protein's atomic coordinates and sequence.

    The general idea is that if any model is capable of predicting a set of coordinates
    and mapping between those coordinates and residue/atom names, then this object can
    be use to transform that output into a PDB file.

    The Python format string was taken from http://cupnet.net/pdb-format/.
    """

    def __init__(self, seqids, coords, atoms_per_res=14, ext_info=None, ter_thr=4.5, use_exters=False):
        """Initializes a PdbBuilder.

        Args:
            coords: A numpy matrix of shape (L x N) x 3, where L is the protein sequence
                length and N is the number of atoms per residue in the coordinate set.
            seq: A length L string representing the protein sequence with  one character
                per amino acid.
            atoms_per_res: The number of atoms recorded per residue. This must be the
                same for every residue.
        """
        if len(seqids) != coords.shape[0] / atoms_per_res:
            raise ValueError(
                "The sequence length must match the coordinate length and contain 1 "
                "letter AA codes." + str(coords.shape[0] / atoms_per_res) + " " +
                str(len(seqids)))
        if coords.shape[0] % atoms_per_res != 0:
            raise AssertionError(f"Coords is not divisible by {atoms_per_res}. "
                                 f"{coords.shape}")
        if atoms_per_res != 14:
            raise ValueError(
                "Values for atoms_per_res other than 14 are currently not supported.")

        self.coords = coords
        self.seqids = seqids
        self.seq = seqids_to_seq(seqids)
        self.mapping = self._make_mapping_from_seq()
        self.atoms_per_res = atoms_per_res
        self.ext_info = ext_info
        self.ter_thr = ter_thr
        self.use_exters = use_exters
        self._set_ext_info()
        

        # PDB Formatting Information
        # self.format_str       = ("{:6s}{:5d}  {:3s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}"
        #                          "{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}")
        self.format_str       = ("{:6s}{:5d}  {:3s}{:1s}{:3s}{:>2s}{:4d}{:1s}   {:8.3f}{:8.3f}"
                                 "{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}")
        # self.format_str_large = ("{:6s}{:5s}  {:3s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}"
        #                          "{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}")
        self.format_str_large = ("{:6s}{:5s}  {:3s}{:1s}{:3s}{:>2s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}")
        self.defaults = {
            "alt_loc": "",
            "chain_id": "A",
            "insertion_code": "",
            "occupancy": 3,
            "temp_factor": 0,
            "element_sym": "",
            "charge": ""
        }
        self.title = "Untitled"
        self.atom_nbr = 1
        self.res_nbr = 1
        self.res_name = 'ALA'
        self.occupancy  = self.defaults["occupancy"]
        self.bfactor    = self.defaults["temp_factor"]
        self.chain_name = self.defaults["chain_id"]
        self.chain_id_cur = 1
        self._pdb_str = ""
        self._pdb_body_lines = []
        self._pdb_lines = []

        self.res_dict   = {}
        self.res_dict['id']  = []
        self.res_dict['cha'] = []
        self.res_dict['res'] = []
        self.res_dict['sse'] = []

    def _coord_generator(self):
        """A generator that iteratively yields self.atoms_per_res atoms at a time."""
        coord_idx = 0
        while coord_idx < self.coords.shape[0]:
            yield self.coords[coord_idx:coord_idx + self.atoms_per_res]
            coord_idx += self.atoms_per_res

    def _sse_generator(self):
        # res_dict   = {}
        # import pdb;pdb.set_trace()
        if isinstance(self.res_dict['id'][0], torch.Tensor) and self.res_dict['id'][0].is_cuda:
            self.res_dict['id'] = [ id.cpu().item() for id in self.res_dict['id']]
        # if isinstance(self.res_dict['id'], torch.Tensor):
        return get_sse_records_by_class(self.res_dict)
    
    def _get_line_for_atom(self, res_name, atom_name, atom_coords):
        """Returns the 'ATOM...' line in PDB format for the specified atom.

        If missing, this function should have special, but not yet determined,
        behavior.
        """
        
        atom_nbr = encode_atom_id(self.atom_nbr)
        return self.format_str_large.format(
            "ATOM", atom_nbr, atom_name, self.defaults["alt_loc"],
            restype_1to3[res_name], self.chain_name, self.res_nbr,
            self.defaults["insertion_code"], atom_coords[0], atom_coords[1],
            atom_coords[2], self.occupancy, self.bfactor, atom_name[0],
            self.defaults["charge"])

    def _get_lines_for_residue(self, res_name, atom_names, coords, ca_only=False):
        """Returns a list of PDB-formatted lines for all atoms in a single residue.

        Calls get_line_for_atom.
        """
        residue_lines = []
        for atom_name, atom_coord in zip(atom_names, coords):
            if ca_only and atom_name != "CA":
                continue
            if (atom_name == "PAD" or atom_name == "" or np.isnan(atom_coord).sum() > 0 or
                    atom_coord.sum() == 0):
                continue
            residue_lines.append(self._get_line_for_atom(res_name, atom_name, atom_coord))
            self.atom_nbr += 1
        return residue_lines

    def _get_lines_for_protein(self, ca_only=False):
        """Returns a list of PDB-formatted lines for all residues in this protein.

        Calls get_lines_for_residue.
        """
        self._pdb_body_lines = []
        self.res_nbr = 1
        self.atom_nbr = 1
        mapping_coords = zip(self.mapping, self._coord_generator())
        i = 0
        pre_res_coords = None
        for (res_name, atom_names), res_coords in mapping_coords:
            self.res_name = res_name
            if self.ext_info is not None and 'chainames' in self.ext_info : 
                self.chain_name = self.ext_info['chainames'][i]
            if not self.use_exters and self._is_ter(pre_res_coords, res_coords):
                self._pdb_body_lines.extend(["TER      1"])
            self._set_ext_info_for_residue(i)
            self._pdb_body_lines.extend(
                self._get_lines_for_residue(res_name, atom_names, res_coords, ca_only)
            )
            if self.use_exters: 
                if self.ext_info['ters'][i] :
                    self._pdb_body_lines.extend(["TER      0"])
                # elif self._is_ter(pre_res_coords, res_coords):
                #     self._pdb_body_lines.extend(["TER      01"])
            pre_res_coords = res_coords
            i+=1
            
        return self._pdb_body_lines

    def _is_ter(self, pre_res_coords, res_coords):
        ca_ind = 1
        if res_coords.shape[0]==1:
            ca_ind = 0

        if pre_res_coords is None:
            return False
        else:
            c0 = pre_res_coords[ca_ind]
            c1 = res_coords[ca_ind]
            dist = np.sqrt(np.sum((c0-c1)**2))
            # dist = torch.sqrt(torch.sum((c0-c1)**2))
            if dist > self.ter_thr:
                return True
            else:
                return False


    def _set_ext_info(self):
        if self.ext_info is None:
            self.chain_ids_uniq = [1]
        else:
            chain = self.ext_info["chain"]
            if isinstance(chain, torch.Tensor):
                if chain.is_cuda:
                    chain = chain.cpu()
                chain = chain.numpy()
            self.chain_ids_uniq = np.unique(np.round(chain)).astype(np.int32)
            self.ext_info["chain"] = chain.astype(np.int32)

            scores = self.ext_info["scores"]
            if isinstance(scores, torch.Tensor):
                if scores.is_cuda:
                    scores = scores.cpu()
                scores = scores.numpy()
            self.ext_info["scores"] = scores

            # if 'ters' not in self.ext_info.keys():
            #     self.use_ters = True
            # else:
            #     self.use_ters = False

    def _set_ext_info_for_residue(self, i):
        if self.ext_info is None:
            self.res_nbr += 1
        else:
            if 'sses' in self.ext_info.keys():
                self.occupancy  = self.ext_info["sses"][i]
            if 'scores' in self.ext_info.keys() and self.ext_info["scores"] is not None:
                self.bfactor    = self.ext_info["scores"][i]
            if 'chain' in self.ext_info.keys():
                self.chain_name = encode_chain_id(self.ext_info["chain"][i])
                
                self.chain_id_cur = self.ext_info["chain"][i]
            if 'chainames' in self.ext_info.keys():
                self.chain_name = self.ext_info["chainames"][i]
                if "chain" in self.ext_info:
                    key = "chain"
                if "chainids" in self.ext_info:
                    key = "chainids"
                self.chain_id_cur = self.ext_info[key][i]
            if 'resnums' in self.ext_info.keys():
                self.res_nbr = self.ext_info["resnums"][i]
                if self.res_nbr > 9999:
                    self.res_nbr = self.ext_info["resnums"][i] - 9999
                    self.chain_name = encode_chain_id(len(self.chain_ids_uniq)+self.ext_info["chain"][i])
            # else:
            #     raise "resnums not "

        self.res_dict['id'].append(self.res_nbr)
        self.res_dict['cha'].append(self.chain_name)
        self.res_dict['res'].append(restype_1to3[self.res_name])
        self.res_dict['sse'].append(int(self.occupancy))

    def _make_header(self, title):
        """Return a string representing the PDB header."""
        header_str = (f"REMARK  {title}\n"
                       "REMARK  CryoNet\n"
                    #    "REMARK  Author: K.X. \n"
                      f"REMARK  Created: {time.ctime()}\n"
                       "REMARK  \n"
                      f"REMARK  Model version: {self.ext_info['version']}\n"
                       "REMARK  \n"
                       )
        return header_str

    
    @staticmethod
    def _make_footer():
        """Return a string representing the PDB footer."""
        return "TER\nEND          \n"

    def _make_mapping_from_seq(self):
        """Given a protein sequence, this returns a mapping that assumes coords are
        generated in groups of 14, i.e. the output is L x 14 x 3."""
        mapping = []
        for residue in self.seq:
            mapping.append((residue, ATOM_MAP_14[residue]))
        return mapping

    def get_pdb_string(self, title=None, ca_only=False):
        if not title:
            title = self.title

        if self._pdb_str:
            return self._pdb_str
        self._get_lines_for_protein(ca_only)
        self._pdb_lines = [self._make_header(title)] + \
                          [self._sse_generator()] + \
                          self._pdb_body_lines + \
                          [self._make_footer()]
        self._pdb_str = "\n".join(self._pdb_lines)
        return self._pdb_str

    def save_pdb(self, path, title="UntitledProtein", ca_only=False):
        """Writes out the generated PDB file as a string to the specified path."""
        with open(path, "w") as outfile:
            outfile.write(self.get_pdb_string(title, ca_only))

    def save_gltf(self, path, title="test", create_pdb=False):
        """First creates a PDB file, then converts it to GLTF and saves it to disk.

        Used for visualizing with Weights and Biases.
        """
        import pymol
        assert ".gltf" in path, "requested filepath must end with '.gtlf'."
        if create_pdb:
            self.save_pdb(path.replace(".gltf", ".pdb"), title)
        pymol.cmd.load(path.replace(".gltf", ".pdb"), title)
        pymol.cmd.color("oxygen", title)
        pymol.cmd.save(path, quiet=True)
        pymol.cmd.delete("all")


def parsePDB2(path, calc_angle=False):
    """Extracts protein sequence, coordinates, and angles from a ProDy chain.

    Args:
        chain: ProDy chain object

    Returns:
        Returns a tuple (angles, coords, sequence) for the protein chain.
        Returns None if the data fails to parse.
        Example angles returned:
            [[phi, psi, omega, ncac, cacn, cnca, chi1, chi2,...chi12],[...] ...]
    """
    parser  = PDBParser()
    protein = parser.get_structure(path, path)
    info = protein.get_info_dict()
    return info



def parsePDB3(path, q_seq):
    """
    Parse PDB template with query sequence
    """
    from cryonet.libs.structure import protein
    from cryonet.model import all_atom
    prot = protein.from_pdb_string(open(path).read())

    prot2 = parsePDB2(path)

    N = len(q_seq)
    atom_positions = np.zeros([N, 37, 3])
    atom_mask = np.zeros([N, 37])
    # b_factors = np.zeros([N, 37])
    b_factors = np.zeros([N])
    chainids = np.zeros([N])
    sses = np.zeros([N])
    ters = np.zeros([N])
    aatype = np.ones([N], dtype=np.int32) * restypes_with_x.index('X')

    for idx in range(len(prot.residue_index)):
        q_res_idx = prot.residue_index[idx]-1
        p_restype = restypes_with_x[prot.aatype[idx]]
        if p_restype != q_seq[q_res_idx]:
            print(f"Warning: {p_restype} {q_seq[q_res_idx]}")

        atom_positions[ prot.residue_index[idx]-1 ] = prot.atom_positions[idx]
        atom_mask[ prot.residue_index[idx]-1 ] = prot.atom_mask[idx]
        # b_factors[ prot.residue_index[idx]-1 ] = prot.b_factors[idx]
        aatype[ prot.residue_index[idx]-1 ] = prot.aatype[idx]
        chainids[ prot.residue_index[idx]-1 ] = prot.chain_index[idx]
        b_factors[ prot.residue_index[idx]-1 ] = prot2['scores'][idx]
        sses[ prot.residue_index[idx]-1 ] = prot2['sses'][idx]
        ters[ prot.residue_index[idx]-1 ] = prot2['ters'][idx]
        # import pdb;pdb.set_trace()

    batch = {
        'all_atom_positions': atom_positions,
        'all_atom_mask': atom_mask,
        'b_factors': b_factors,
        'aatype': aatype,
        'seq_mask': aatype<20
    }
    batch = all_atom.make_atom14_positions(batch)

    info = {
        'coords':   batch['all_atom_positions_14'],
        'masks':    batch['all_atom_mask_14'],
        'chainids': chainids,
        # 'chainames': chainames_np,
        'resnums':  np.arange(len(aatype))+1,
        'seqids':   aatype,
        'scores':   b_factors,
        'ters':     ters,
        'sses':     sses,
        'seqs':     q_seq,
        # 'atom_coords': ,
        # 'atom_elem': atom_elem_np
    }
    return info



class PdbReader(object):
    """Creates a PDB file given a protein's atomic coordinates and sequence.

    The general idea is that if any model is capable of predicting a set of coordinates
    and mapping between those coordinates and residue/atom names, then this object can
    be use to transform that output into a PDB file.

    The Python format string was taken from http://cupnet.net/pdb-format/.
    """

    def __init__(self, path, fast=True, rm_redun=False, sequence=""):
        """Initializes a PdbBuilder.

        Args:
            coords: A numpy matrix of shape (L x N) x 3, where L is the protein sequence
                length and N is the number of atoms per residue in the coordinate set.
            seq: A length L string representing the protein sequence with  one character
                per amino acid.
            atoms_per_res: The number of atoms recorded per residue. This must be the
                same for every residue.
        """
        self.path = path
        self.rm_redun = rm_redun
        if sequence=="":
            if fast:
                self.parser = parsePDB2
            else:
                self.parser = parsePDB
        
            if isinstance(self.path, list):
                self.info = self.parsePDBList(self.path)
            else:
                self.info = self.parser(path)
        else:
            self.parser = parsePDB3
            self.info = self.parser(path, sequence)

        if rm_redun:
            self.remove_redundant()

    def parsePDBList(self, paths):
        info_all = {}
        i = 0
        for path in paths:
            print("parse pdb file({}/{}): {}".format(i+1,len(paths), path))
            info = self.parser(path)
            if i == 0:
                for k in info.keys():
                    if info[k] is not None:
                        info_all[k] = info[k]
            else:
                for k in info.keys():
                    # import pdb;pdb.set_trace()
                    if info[k] is not None:
                        info_all[k] = np.concatenate([info_all[k], info[k]])
            i+=1
       
        return info_all

    def remove_redundant(self):
        assert 'seqids' in self.info.keys(), "no seqids in pdb info."
        print("Checking redundant amino acids...")
        irr = 0
        po_list = {}
        for iclass in range(1, 22):
            po_list[iclass] = []
        def is_redundant(po, po_list, thr=2.0):
            dis = distance.cdist(po.reshape(1, 3), np.array(po_list))
            return dis.min() < thr
        keep = []
        seqids = self.get("seqids")
        coords = self.get("coords")
        for i in range(len(seqids)):
            res_cls = int(seqids[i])
            if len(po_list[res_cls]) > 0 and is_redundant(coords[i,1], po_list[res_cls]):
                irr += 1
                continue
            po_list[res_cls].append(coords[i, 1])
            keep.append(i)
        print("remove redundant: {}/{}.".format(irr, len(seqids)))
        keep = np.array(keep)
        for k in self.info.keys():
            if k == 'seqs':
                continue
            if self.info[k] is not None:
                self.info[k] = self.info[k][keep]
        # print("remain: ", )




    def get(self, key):
        if key in self.info.keys():
            return self.info[key]
        else:
            raise KeyError(
                "Key {} not in pdb info.".format(key))
    def to_boxes(self):
        boxes = Boxes(self.get("boxes"))
        boxes.set_classes(self.get("seqids"))
        boxes.set_sses(self.get("sses"))
        boxes.set_poses(self.get("coords"))
        boxes.set_scores(self.get("scores"))
        boxes.set_path(self.get("resnums"))
        boxes.set_ters(self.get("ters"))
        boxes.set_res_nums(self.get("resnums"))
        boxes.set_chain_ids(self.get('chainids'))
        return boxes

    def to_structure_builder(self, device=torch.device("cpu")):
        from cryonet.libs.structure import StructureBuilder
        ext_info = {
            'chain':     self.get('chainids'),
            'resnums':   self.get('resnums'),
            'sses':      self.get('sses'),
            'scores':    self.get("scores"),
            'seqids':    self.get("seqids"),
            'ters':      self.get("ters"),
            'masks':     self.get("masks"),
            # 'seqs':      self.get("seqs"),
        }

        crd = torch.from_numpy(self.get('coords')).to(device)
        sb  = StructureBuilder(self.get('seqids'), crd=crd, ext_info=ext_info, device=device)
        sb.build()
        return sb
    
    def get_info(self):
        return self.info


# def load_temp_residues():
#     '''
#     Output: residue_dict = {'ALA' : <residue object of ALA>}
#     '''
#     residue_coords = {'coords':{}, 'masks':{}}

#     for key in RESIDUE_NAMES:
#         path = 'datasets/amino_acid/{}.pdb'.format(key)
#         pr = PdbReader(path, fast=True)
        
#         coords = pr.get("coords")[0]
#         # if key == 'ILE':
#             # import pdb; pdb.set_trace()
#         order= np.array(SC_BUILD_INFO[key]['order'])
#         coords = coords[order]
            
#         masks  = pr.get("masks")[0]
        
#         residue_coords['coords'][pr.get("seqids")[0]] = (coords - coords[1]) * masks
#         residue_coords['masks'][pr.get("seqids")[0]] =  masks

#     return residue_coords


    

def parse_to_residue(resname, path):
    '''
    Input: residue name & path
    Output: a residue object
    '''
    parser = PDBParser()
    protein = parser.get_structure(resname, path)
    for model in protein:
        for chain in model:
            for residue in chain:
                return residue

# def load_20_residues():
#     '''
#     Output: residue_dict = {'ALA' : <residue object of ALA>}
#     '''
#     residue_dict = {}
#     residue_coords = {}

#     bb_atom_list = ["N", "CA", "C","O"]
#     for key in RESIDUE_NAMES:
#         path = 'datasets/amino_acid/{}.pdb'.format(key)
#         try:
#         # import pdb; pdb.set_trace()
#             residue_dict[key] = parse_to_residue(key, path)
#         except:
#             key  = "ALA"
#             path = 'datasets/amino_acid/{}.pdb'.format(key)
#             residue_dict[key] = parse_to_residue(key, path)
#         kid = RESIDUE_CLASS_DICT[key]
#         residue_coords[key] = {}
#         residue_coords[kid] = {}
#         CA_coord = residue_dict[key]["CA"].copy().get_coord()
#         for atom in residue_dict[key].copy().get_iterator():
#             atom_id = atom.get_id()
#             if atom_id in bb_atom_list:
#                 residue_coords[key][atom_id] = atom.get_coord() - CA_coord
#                 residue_coords[kid][atom_id] = atom.get_coord() - CA_coord
#     residue_coords['UNK'] = residue_coords['ALA'] 
#     return residue_dict, residue_coords




