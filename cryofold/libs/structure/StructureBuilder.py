# Copyright (c) CryoNet Team, and its affiliates. All Rights Reserved

"""A class for generating/visualizing protein atomic coordinates from measured angles."""

import numpy as np
import torch
from copy import deepcopy
import os
import time, datetime
from .density import DensityInfo, mol_atom_density
from cryofold.libs.PDB import PDBParser, MMCIFParser
# from libs.structure.sequence import ProteinVocabulary
from cryofold.libs.transforms.rotations import quaternion_to_matrix, vec_angle_rotation
from cryofold.libs.structure.pdbfile import PdbBuilder, seqids_to_seq
# from cryofold.np.constants import restypes_with_x, restype_name_to_atom14_names, restype_1to3, restype_3to1
from cryofold.np.residue_constants import restypes_with_x, restype_name_to_atom14_names, restype_1to3, restype_3to1


global VOCAB
VOCAB = None

_EPS = np.finfo(float).eps * 4.0
RAD2DEG = 180 / np.pi

NUM_COORDS_PER_RES = 14


def get_structure(pdbfile, pred_sse=False):
    if pdbfile.endswith(".pdb"):
        parser = PDBParser()
        return parser.get_structure(pdbfile, pdbfile, pred_sse)
    elif pdbfile.endswith(".cif"):
        parser = MMCIFParser()
        return parser.get_structure(pdbfile, pdbfile)
    else:
        raise "Wrong PDB file (not .pdb/.cif format)?"


def pdb_to_fasta(pdb_path, out_path):
    if out_path == "":
        out_path = pdb_path.replace(".pdb", ".fasta")
    pdbname = os.path.basename(out_path).replace(".fasta", "")
    protein = get_structure(pdb_path)
    sequences = ""
    ifrag = 0
    for model in protein:
        for chain in model:
            res_id0 = chain.child_list[0].get_id()[1]
            label = "{}|Chain={}|{}".format(
                pdbname, chain.get_id().strip(), res_id0)
            seq = ""
            start = res_id0
            pre_res_id = None
            for r in chain:
                res_ty, res_id,_ = r.get_id()
                if res_ty.strip() != "":
                    continue
                    # import pdb; pdb.set_trace()

                if seq != "" and (pre_res_id is not None and res_id - pre_res_id != 1):
                    sequences += ">{}|{}\n{}\n".format(label + "-" + str(start + len(seq) - 1), len(seq), seq)
                    ifrag += 1
                    seq = ""
                    start = res_id

                if seq == "":
                    label = "{}|Chain={}|{}".format(
                        pdbname, chain.get_id().strip(), res_id)

                seq = seq + restype_3to1.get(r.get_resname(), 'X')
                if r.is_ter == 1:
                   sequences += ">{}|{}\n{}\n".format(label + "-" + str(start + len(seq) - 1), len(seq), seq)
                   ifrag += 1
                   seq = ""
                   start = res_id

                pre_res_id = res_id

            if len(seq) > 0:
                sequences += ">{}|{}\n{}\n".format(label + "-" + str(start + len(seq) - 1), len(seq), seq)
                ifrag += 1
    f = open(out_path, "w")
    f.writelines(sequences)
    f.close()
    print('===Done! {} sequence fragment(s) in {}'.format(ifrag, out_path))


class StructureBuilder(object):
    """Reconstruct a protein's structure given its sequence and angles or coordinates.

    The hydroxyl-oxygen of terminal residues is not placed because this would
    mean that the number of coordinates per residue would not be constant, or
    cause other complications (i.e. what if the last atom of a structure is not
    really a terminal atom because it's tail is masked out?).
    """

    def __init__(self, seq, crd, qua=None, tran=None, off_set=None, ext_info={}, ter_thr=5.0, sc_var=False, device=torch.device("cpu")):
        """Initialize a StructureBuilder for a single protein. Does not build coordinates.

        To generate coordinates after initialization, see build().
        To create PDB/GLTF files or to generate a py3Dmol visualization, see
        to_{pdb,gltf,3Dmol}.

        Args:
            seq: An integer tensor or a string of length L that represents the protein's
                amino acid sequence.
            ang: A float tensor (L X NUM_PREDICTED_ANGLES) that contains all of the
                protein's interior angles.
            crd: A float tensor ((L X NUM_COORDS_PER_RES) X 3) that contains all of the
                protein's atomic coordinates. Each residue must contain the same number
                of coordinates, with empty coordinate entries padded with 0-vectors.
            device: An optional torch device on which to build the structure.
        """
        # TODO support one-hot sequences
        # Perhaps the user mistakenly passed coordinates for the angle arguments
        global VOCAB
        # if VOCAB is None:
        #     VOCAB = ProteinVocabulary(device=device)
        
        self.device   = device
        self.qua      = qua
        self.tran     = tran
        self.off_set  = off_set
        self.ext_info = ext_info
        self.ter_thr  = ter_thr
        self.sc_var   = sc_var

        
        
        # Validate input data
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq).to(torch.int32)
        
        # build coords
        if isinstance(seq, np.ndarray) or isinstance(seq, torch.Tensor):
            seq[seq>=21]=1
        if not isinstance(crd, torch.Tensor):
            crd = torch.from_numpy(crd)
        if len(crd.shape) == 3:
            crd = crd.view(-1,3)
        self.coords = crd.to(torch.float32).to(device)

        self.set_seq(seq)
        self.set_coords_mask()
        self.set_ext_info()

        self.data_type = "torch" if isinstance(self.coords, torch.Tensor) else "numpy"
        # import pdb;pdb.set_trace()
        if (self.coords is not None and
            (self.coords.shape[0] // NUM_COORDS_PER_RES) != len(self.seqids)):
            raise ValueError(
                f"The length of the coordinate matrix must match the sequence length "
                f"times {NUM_COORDS_PER_RES}. You have provided {self.coords.shape[0]} //"
                f" {NUM_COORDS_PER_RES} = {self.coords.shape[0] // NUM_COORDS_PER_RES}.")

        self.prev_ang = None
        self.prev_bb = None
        self.next_bb = None
        self.residues = []

        self.pdb_creator = None

    def set_seq(self, seqids):
    #     self.seq_as_str = seqids_to_seq(seqids)
    #     # self.seq_as_str = seq if type(seq) == str else self._convert_seq_to_str(seq)
    #     # self.seqs = self.seq_as_str
    #     # self.seq_as_ints = np.asarray([VOCAB._char2int[s] for s in self.seq_as_str])
        self.seqids = seqids
    # def set_seq(self, seq):
    #     self.seq_as_str = seq if type(seq) == str else self._convert_seq_to_str(seq)
    #     self.seqs = self.seq_as_str
    #     self.seq_as_ints = np.asarray([VOCAB._char2int[s] for s in self.seq_as_str])
    #     # import pdb;pdb.set_trace()
    #     # print("seq", type(seq))
    #     if isinstance(seq, torch.Tensor):
    #         self.seqids = seq
    #     else:
    #         self.seqids = self.seq_as_ints

    def set_coords_mask(self):
        assert self.ext_info is not None and 'masks' in self.ext_info.keys()
        masks = self.ext_info['masks']
        if isinstance(masks, torch.Tensor):
            self.coords_mask = masks
        else:
            self.coords_mask = torch.from_numpy(masks)
    # def set_coords_mask(self):
    #     if self.ext_info is not None and 'masks' in self.ext_info.keys():
    #         self.coords_mask = torch.from_numpy(self.ext_info['masks'])
    #     else:
    #         self.coords_mask = torch.cat([VOCAB._int2coordsmask[s] for s in self.seq_as_ints]) # [N*14,3]

    def has(self, key):
        if self.ext_info is None:
            return False
        else:
            return key in self.ext_info.keys()


    def set_ext_info(self):
        nr = self.num_res()
        if self.has('resnums'):
            self.resnums   = self.ext_info['resnums'] if isinstance(self.ext_info['resnums'], torch.Tensor) else torch.from_numpy(self.ext_info['resnums']).to(self.device)
        else:
            n0 = (nr//10000) 
            n1 = nr%10000
            resnums = []
            chainids = []
            for i in range(n0):
                resnums.append(torch.arange(10000))
                chainids.append(torch.ones(10000)*i)
            if n1>0:
                resnums.append(torch.arange(n1))
                chainids.append(torch.ones(n1)*n0)
            self.resnums   = torch.cat(resnums).to(self.device)
            self.ext_info['resnums'] = self.resnums

            self.chainids   = torch.cat(chainids).to(self.device)
            self.ext_info['chain'] = self.chainids

        if self.has('version'):
            self.version = self.ext_info['version'] 
        else:
            self.version = "CryoNet_V2022-03-02"
            self.ext_info['version'] = self.version

        if self.has('name'):
            self.name = self.ext_info['name'] 
        else:
            self.name = "XXXX"
            self.ext_info['name'] = self.name

        if self.has('chain'):
            self.chainids   = self.ext_info['chain'] if isinstance(self.ext_info['chain'], torch.Tensor) else torch.from_numpy(self.ext_info['chain']).to(torch.int32).to(self.device)
        else:
            self.chainids   = torch.zeros(nr, dtype=torch.int32).to(self.device)
            self.ext_info['chain'] = self.chainids

        
        if self.has('scores'):
            self.scores   = self.ext_info['scores'] if isinstance(self.ext_info['scores'], torch.Tensor) else torch.from_numpy(self.ext_info['scores']).to(torch.float32).to(self.device)
        else:
            self.scores   = torch.zeros(nr, dtype=torch.float32).to(self.device)
            self.ext_info['scores'] = self.scores

        if self.has('sses'):
            self.sses   = self.ext_info['sses'] if isinstance(self.ext_info['sses'], torch.Tensor) else torch.from_numpy(self.ext_info['sses']).to(torch.float32).to(self.device)
        else:
            self.sses   = torch.zeros(nr, dtype=torch.float32).to(self.device) + 3
            self.ext_info['sses'] = self.sses

        if self.has('ters'):
            self.ters   = self.ext_info['ters'] if isinstance(self.ext_info['ters'], torch.Tensor) else torch.from_numpy(self.ext_info['ters']).to(torch.bool).to(self.device)
        else:
            self.ters   = torch.tensor([False]*nr).to(self.device)
            self.ext_info['ters'] = self.ters
        
        
        self.resinds   = torch.arange(nr).to(self.device)


    def generate_fragments(self):
        r_keys = ['ters', 'chain', 'seqids']
        assert self.ext_info is not None and set(r_keys).issubset(set(self.ext_info.keys())), "not all keys({}) in ext_info.".format(r_keys)
        self.ters   = self.ext_info['ters']
        self.chainids  = self.ext_info['chain']
        self.seqids  = self.ext_info['seqids']

    def isindex(self, ind):
        return ind>=0 and ind < self.num_res()

    def get_resnum(self, resnum, resnums):
        ind = (resnums==resnum).nonzero()
        if len(ind)==0:
            return None
        else:
            return ind[0, 0]

    def get_coords(self, ind=None):
        if ind is None:
            return self.coords.view(self.num_res(), -1, 3)[:,:,:]
        else:
            if type(ind) ==int and ind>=0 and ind < self.num_res():
                return self.coords.view(self.num_res(), -1, 3)[ind,:,:]
            elif (type(ind) == list or isinstance(ind, np.ndarray) or isinstance(ind, torch.Tensor)):
                return self.coords.view(self.num_res(), -1, 3)[ind,:,:]
            else:
                raise "index error,  {} is not in (0,{})".format(ind, self.num_res())

    def get_ca_coords(self, ind=None):
        return self.get_coords(ind)[:,1,:]

    

    # @staticmethod
    def _convert_seq_to_str(self, seq):
        """Assuming seq is an int list or int tensor, returns its str representation."""
        seq_as_str = ""
        # if isinstance(seq, torch.Tensor):
        #     seq = seq.numpy()
        seq = seq.flatten().int()
        # import pdb;pdb.set_trace()
        if len(seq.shape) == 1:
            # The seq is represented as an integer sequence
            if len(seq.shape) == 1:
                seq_as_str = VOCAB.ints2str(seq)
            elif len(seq.shape) == 2:
                if seq.shape[0] != 1:
                    raise ValueError(f"Seq shape {seq.shape} is not supported.")
                else:
                    # import pdb;pdb.set_trace()
                    seq_as_str = VOCAB.ints2str(seq[0])
        else:
            raise UnsupportedOperation(f"Seq shape {seq.shape} is not supported.")

        return seq_as_str

    def __len__(self):
        """Return length of the protein sequence.

        Returns:
            int: Integer sequence length.
        """
        return len(self.seqids)
    
    def num_res(self):
        """Return length of the protein sequence.

        Returns:
            int: Integer sequence length.
        """
        return len(self.seqids)


    def build(self, by_residue=False):
        """Construct all of the atoms for a residue.

        Special care must be taken for the first residue in the sequence in
        order to place its CB, if present.

        Returns:
            (numpy.ndarray, torch.Tensor): An array or tensor of the generated coordinates
            with shape ((L X NUM_COORDS_PER_RES) X 3).
        """
        # If a StructureBuilder does not have angles, build returns its coordinates
        return self.coords
        
    def _insert_(self, index, main, var):
        return torch.cat((main[:index], var, main[index:]))

    def insert(self, chainid, resind, seq, coords, sses=None):

        ne_l = self.get_neighbors(resind, chainid, t='l')
        # import pdb;pdb.set_trace()
        index = ne_l[-1]+1
        g_n = len(seq)
        merg_seq = self.seqs[:index] + seq + self.seqs[index:]
        self.set_seq(merg_seq)

        self.coords = self._insert_(index, self.get_3dtensor(), coords)
        g_cha  = torch.zeros(g_n, dtype=torch.int32).to(self.device) + chainid
        self.chainids = self._insert_(index, self.chainids, g_cha)

        g_resnum       = torch.arange(g_n).to(self.device) + self.resnums[index-1] + 1
        self.resnums   = self._insert_(index, self.resnums, g_resnum) 

        self.ters      = self._insert_(index, self.ters, torch.tensor( [False] * g_n, dtype=torch.bool).to(self.device)) 

        g_scores       = torch.ones(g_n, dtype=torch.float32).to(self.device) + 0.2
        self.scores    = self._insert_(index, self.scores, g_scores) 

        if sses is None:
            sses     = torch.zeros(g_n, dtype=torch.float32).to(self.device) + self.sses[index-1]
        # else:

        self.sses  = self._insert_(index, self.sses, sses) 

        # merg = torch.cat((main[:index], coords, main[index+1:]))

        self.resinds   = torch.arange(self.num_res()).to(self.device)

        self.ext_info = {
            'chain':   self.chainids,
            'resnums': self.resnums,
            'sses':    self.sses,
            'scores':  self.scores,
            'seqids':  self.seqids,
            'ters':    self.ters,
            'version': self.version,
            'name':    self.name,
            # 'seqs':      self.get("seqs"),
        }

        self.pdb_creator = None
        # print(self.coords.shape)

    def sub(self,s,e):
        coords = self.get_3dtensor()[s:e]
        ext_info = {k:v[s:e] for k,v in self.ext_info.items()}    
        # import pdb;pdb.set_trace()

        return StructureBuilder(ext_info['seqids'], coords, ext_info=ext_info)

    def to_3dtensor(self):
        self.coords = self.coords.view(-1, 14, 3)
    
    def get_3dtensor(self):
        return self.coords.view(-1, 14, 3)

    def to_batch_tensor(self):
        self.coords = self.coords.view(1, -1, 14, 3)
    
    def get_batch_tensor(self):
        return self.coords.view(1, -1, 14, 3)

    def get_gap_index(self):
        gap = torch.zeros(self.num_res())
        chain_pre = self.chainids[0]
        resnum_pre = self.resnums[0]
        for i in range(1, self.num_res()):
            if self.resnums[i] - resnum_pre !=1 or self.chainids[i]!=chain_pre:
                gap[i] = 1
            chain_pre = self.chainids[i]
            resnum_pre = self.resnums[i]
        return gap

    def to_list(self):
        #TODO: to complete
        assert self.coords.shape[1:] == (14,3), "wrong shape."
        coords = self.coords.view(-1,3) * self.coords_mask
        atom_ind = coords.sum(dim=1)!=0
        return coords[atom_ind]

    def to_boxes(self):
        from util.boxes3d import Boxes
        ca = self.get_ca_coords()
        bb=torch.cat([ca,ca],dim=1)
        # import pdb ; pdb.set_trace()
        boxes = Boxes(bb)
        boxes.set_classes(self.seqids)
        boxes.set_sses(self.sses)
        boxes.set_poses(self.get_3dtensor())
        boxes.set_scores(self.scores)
        boxes.set_path(self.resnums)
        boxes.set_ters(self.ters)
        boxes.set_res_nums(self.resnums)
        boxes.set_chain_ids(self.chainids)
        return boxes

    def estimate_mask(self):
        from util.metrics import AAMetrics
        boxes = self.to_boxes()
        self.met = AAMetrics(boxes, boxes)
        masks,_ = self.met.estimate_mask()
        self.masks = torch.tensor(masks).to(self.device)
        # import pdb ; pdb.set_trace()

    def _initialize_coordinates_and_PdbCreator(self, use_exters=False):
        if self.coords is None or len(self.coords) == 0:
            self.build()
        # import pdb ; pdb.set_trace()
        if not self.pdb_creator:
            if len(self.coords.shape) == 3:
                coords = self.coords.reshape(-1, 3)
            else:
                coords = self.coords
            if self.data_type == 'numpy':
                self.pdb_creator = PdbBuilder(self.seqids, coords, ext_info=self.ext_info, ter_thr=self.ter_thr,use_exters=use_exters)
            else:
                if coords.is_cuda:
                    coords =coords.cpu()
                self.pdb_creator = PdbBuilder(self.seqids, 
                                              coords.detach().numpy(), ext_info=self.ext_info, ter_thr=self.ter_thr,use_exters=use_exters)
    
    def translation(self, tran, first_ca=False):
        if first_ca:
            self.coords = self.coords - self.coords[1,:] 
        self.coords = self.coords + tran 

    def rotation(self):
        rot = quaternion_to_matrix(self.qua)
        if rot.shape[0] == 4:
            tmp_coords = torch.zeros([self.coords.shape[0], rot.shape[1]], device=self.device)
            tmp_coords[:,:3] = self.coords
            self.coords = torch.matmul(tmp_coords, rot)[:,:3] 
        else:
            self.coords = torch.matmul(self.coords, rot) 
        # import pdb;pdb.set_trace()
        self.coords = self.coords * self.coords_mask

    def transform_in_density(self):
        scale = self.off_set[0]
        self.rotation()
        self.coords = self.coords / scale  + self.tran 

    def transform_in_physics(self):
        self.rotation()
        self.coords = self.coords + self.tran 
    
    def transform(self, rot, tran):
        # import pdb;pdb.set_trace()
        self.coords = (torch.matmul(self.get_3dtensor(), rot) + tran) * self.coords_mask.float()

    def deoffset(self, datatype='exp'):
        voxelsize = self.ext_info['voxelsize']
        offset = self.ext_info['offset']
        coords = self.coords 
        # coords = de_offset_coords(self.coords, voxelsize, offset, datatype='exp')
        if datatype == "syn":
            coords = coords * voxelsize + offset
        else:
            coords = (coords + offset) * voxelsize 
        self.coords = coords

    def density_to_physics(self):
        # unused
        # to physics space
        scale = self.off_set[0]
        tran_in_den = self.tran / scale
        self.coords = self.coords *  scale - tran_in_den
        self.rotation()
        
    def physics_to_density(self):
        # unused
        # to density space
        scale = self.off_set[0]
        tran_in_phy = self.tran * scale
        self.rotation()
        self.coords = (self.coords + tran_in_phy)/ scale 

    def coords_in_density_space(self):
        self.transform(self.qua, self.tran, self.off_set)
        return self.coords

    def coords_in_physical_space(self):
        return self.coords

    def to_pdb(self, path, title="CryoNet", use_exters=False, ca_only=False):
        """Save protein structure as a PDB file to given path.

        Args:
            path (str): Path to save PDB file.
            title (str, optional): Title of structure for PDB file. Defaults to "pred".
        """
        self._initialize_coordinates_and_PdbCreator(use_exters=use_exters)
        self.pdb_creator.save_pdb(path, title, ca_only)

    def to_fasta(self, path, title="CryoNet"):
        """Save protein structure as a PDB file to given path.

        Args:
            path (str): Path to save PDB file.
            title (str, optional): Title of structure for PDB file. Defaults to "pred".
        """
        # not 
        # self._initialize_coordinates_and_PdbCreator()
        # self.pdb_creator.save_pdb(path, title)
        import pdb;pdb.set_trace()

    def to_gltf(self, path, title="pred"):
        """Save protein structure as a GLTF (3D-object) file to given path.

        Args:
            path (str): Path to save GLTF file.
            title (str, optional): Title of structure for GLTF file. Defaults to "pred".
        """
        self._initialize_coordinates_and_PdbCreator()
        self.pdb_creator.save_gltf(path, title)

    def to_3Dmol(self, style=None, **kwargs):
        """Generate protein structure & return interactive py3Dmol.view for visualization.

        Args:
            style (str, optional): Style string to be passed to py3Dmol for
                visualization. Defaults to None.

        Returns:
            py3Dmol.view object: A view object that is interactive in iPython notebook
                settings.
        """
        import py3Dmol
        if not style:
            style = {'cartoon': {'color': 'spectrum'}, 'stick': {'radius': .15}}
        self._initialize_coordinates_and_PdbCreator()

        view = py3Dmol.view(**kwargs)
        view.addModel(self.pdb_creator.get_pdb_string(), 'pdb')
        if style:
            view.setStyle(style)
        view.zoomTo()
        return view
    
    def to_density(self, atom_weight=14.0, res=3.0, voxel_size=1.0, full_atom=True, exclude=[], datatype="torch"):

        # import pdb;pdb.set_trace()
        if isinstance(voxel_size, np.ndarray) or isinstance(voxel_size, torch.Tensor) or isinstance(voxel_size, tuple) or isinstance(voxel_size, list):
            voxel_size = voxel_size[0]
        coords = self.coords.view(-1,3)
        num_res = coords.shape[0]//14
        
        if len(exclude) >0:
            # np.arange(num_res)
            coords = coords.view(num_res,14,3)
            bb=np.delete(np.arange(num_res), exclude)
            ind=torch.from_numpy(bb)
            coords = coords[ind]
            print("exclude:", exclude)
        # num_res = len(coords)
        if full_atom:
            coords = coords.reshape(num_res*14,3)
            mask = self.coords_mask.reshape(num_res*14,3)
            coords = coords[mask[:,0]>0]
        else:
            coords = coords[:,:4,:].reshape(num_res*4,3)
            
        density, offset = mol_atom_density(coords, atom_weight, res, voxel_size, datatype=datatype)

        den = DensityInfo(density=density, offset=offset, apix=voxel_size, datatype=datatype, device=self.device)
        return den

    def to_mask(self, atom_weight=14.0, res=4.0, voxel_size=1.0, backbone=True):
        den = self.to_density(atom_weight=atom_weight, res=res, apix=voxel_size, backbone=backbone)
        den.density[den.density>0] = 1.0
        
        return den

    def to_cpu(self):
        device = torch.device("cpu")
        self.device = device
        self.coords = self.coords.to(self.device)
        self.sses = self.sses.to(self.device)


def batch_building_coords(seq_ids, angles, rots, off_sets, device=torch.device("cpu")):
    D, B, L, _ = angles.shape
    coords = torch.zeros((D, B, L, 14, 3), device=device)
    for ib in range(B):
        seq_id = seq_ids[ib]
        for idec in range(D):
            ang = angles[idec, ib]
            qtr = rots[idec, ib]
            ose = off_sets[idec, ib]
            tra = qtr[1,:3] + qtr[1,3] * 0
            sb  = StructureBuilder(seq_id, ang=ang, qua=qtr[0], tran=tra, off_set=ose, device=device)
            sb.build(by_residue=True)
            sb.transform_in_density()
            sb.to_3dtensor()
            coords[idec, ib] = sb.coords
    return coords

def generate_coords(seq, angles, device=torch.device("cpu")):

    angles_full = torch.zeros((len(angles), 12), device=device)
    # import pdb;pdb.set_trace()
    if isinstance(angles, torch.Tensor):
        angles_full[:,:2] = angles * np.pi
    else:
        angles_full[:,:2] = torch.from_numpy(angles).to(device) * np.pi
    ang = torch.tensor([180, 110, 116.642992978143, 121.382215820277], device=device) * np.pi/180
    angles_full[:,2:6] = ang
    sb  = StructureBuilder(seq, ang=angles_full, sc_var=True, device=device)
    sb.build()
    return sb
