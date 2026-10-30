# Copyright (C) 2002, Thomas Hamelryck (thamelry@binf.ku.dk)
#
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.

"""The structure class, representing a macromolecular structure."""
import numpy as np
from .Entity import Entity
from .internal_coords import IC_Chain
from .Residue import AAMAP, seq_coords_mask, seq_to_ids
from .SSE import get_sse_records_by_class

def getSequence(resnames):
    """Returns polypeptide sequence as from list of *resnames* (residue
    name abbreviations)."""

    get = AAMAP.get
    return ''.join([get(rn, 'X') for rn in resnames])

class Structure(Entity):
    """The Structure class contains a collection of Model instances."""

    def __init__(self, id):
        """Initialize the class."""
        self.level = "S"
        Entity.__init__(self, id)

    def __repr__(self):
        """Return the structure identifier."""
        return "<Structure id=%s>" % self.get_id()

    def get_models(self):
        """Return models."""
        yield from self

    def get_chains(self):
        """Return chains from models."""
        for m in self.get_models():
            yield from m

    def get_chain_ids(self):
        """Return chains ids from models."""
        c_list = []
        for c in self.get_chains():
            c_list.append(c.get_id())
        return c_list 

    def get_residues(self):
        """Return residues from chains."""
        for c in self.get_chains():
            yield from c

    def get_residues_list(self):
        """Return residues from chains."""
        r_list = []
        for res in self.get_residues():
            if not res.has_id('CA'):
                continue
            r_list.append(res)
        return r_list

    def get_atoms(self):
        """Return atoms from residue."""
        for r in self.get_residues():
            yield from r

    def get_coords(self, name='all'):
        if name =='ca':
            name_list = ['CA']
        elif name =='backbone':
            name_list = ['N','CA','C','O']
        elif name =='all':
            name_list = []
        else:
            raise 'name error, {} is not in [all, ca, backbone]'
        coords = []
        for res in self.get_residues():
            r_atom_list=[]
            for atom in res.child_list:
                if name =='all':
                    r_atom_list.append(atom.get_coord())
                elif atom.name in name_list:
                    r_atom_list.append(atom.get_coord())
            if name =='all' and len(r_atom_list)<14:
                    r_atom_list.extend([(0.0,0.0,0.0)] * (14-len(r_atom_list)))
            coords.extend(r_atom_list)
        return np.asarray(coords)
    
    def get_occupancy(self):
        name_list = ['CA']
        occupancy = []
        for res in self.get_residues():
            for atom in res.child_list:
                if atom.name in name_list:
                    occupancy.append(atom.get_occupancy())
        return np.asarray(occupancy)

    def generate_sse(self):
        sse_dict   = {}
        sse_dict['id']  = []
        sse_dict['cha'] = []
        sse_dict['res'] = []
        sse_dict['sse'] = []
        for c in self.get_chains():
            for r in c.child_list:
                sse_dict['id'] .append(r.get_id()[1])
                sse_dict['cha'].append(c.get_id())
                sse_dict['res'].append(r.get_resname())
                sse_dict['sse'].append(r.child_list[0].get_occupancy())
        
        self.sse_list =  get_sse_records_by_class(sse_dict)
    
    # def set_sse_to_c

    def get_sse(self):
        sses = []
        for res in self.get_residues():
            sses.append(res.get_sse())
        return np.asarray(sses)

    def get_chainname_by_id(self, cid):
        return self.chainid_list[cid]
    
    def get_info_dict(self):
        self.chainid_list = []
        coords = []
        sses = []
        atom_elem = []
        atom_coords = []
        boxes = []
        angles = []
        resnums = []
        chainids  = []
        chainames = []
        scores = []
        sses = []
        ters = []
        seqs = ""
        # import pdb;pdb.set_trace()
        for res in self.get_residues():
            res_name = res.get_resname()
            hetflag, resnum, icode = res.get_id()
            # ignore HETATM
            if hetflag.strip() != "":
                continue
            
            resnums.append(resnum)
            if len(self.chainid_list )==0 or res.get_chainid() not in self.chainid_list:
                self.chainid_list.append(res.get_chainid())
            chainames.append(res.get_chainid())
            chainids.append(len(self.chainid_list)-1)
            
            sses.append(res.get_sse())
            scores.append(res.get_bfactor())
            ters.append(res.is_ter)
            seqs += res.get_resname1c()
            coords_i = res.get_coords()
            if len(coords_i)<14:
                coords_i.extend([(0.0,0.0,0.0)] * (14-len(coords_i)))
            coords.append(coords_i)

            boxes.append(res.get_box())
            atom_coords.extend(res.get_coords_list())
            atom_elem.extend(res.get_atomelem_list())
            # masks.append()

            # r_atom_list=[]
            # for atom in res.child_list:
            #     r_atom_list.append(atom.get_coord())
            
            # coords.extend(r_atom_list)
            # import pdb;pdb.set_trace()


        # angles_np   = np.asarray(angles)
        # try:
        if len(coords)==0:
            raise ValueError('get_info_dict need at least one array to stack')
            return None
        coords_np   = np.stack(coords)
        # except:
        #     import pdb;pdb.set_trace()
        masks_np    = np.stack(seq_coords_mask(seqs))
        boxes_np    = np.array(boxes)
        resnums_np  = np.array(resnums)
        chainids_np = np.array(chainids)
        chainames_np = np.array(chainames)
        scores_np   = np.array(scores)
        ters_np     = np.array(ters)
        sses_np     = np.array(sses)
        seqids_np   = np.array(seq_to_ids(seqs))
        atom_coords_np   = np.array(atom_coords)
        atom_elem_np   = np.array(atom_elem)
        # if calc_angle:
        #     qua, tran = compute_rot_trans(observed_sequence, angles_np, coords_np)
        # else:
        qua, tran = None, None
        angles_np = None
            
        info = {
            'angles':   angles_np,
            'qua':      qua,
            'tra':      tran,
            'coords':   coords_np,
            'masks':    masks_np,
            'boxes':    boxes_np,
            'chainids': chainids_np,
            'chainames': chainames_np,
            'resnums':  resnums_np,
            'seqids':   seqids_np,
            'scores':   scores_np,
            'ters':     ters_np,
            'sses':     sses_np,
            'seqs':     seqs,
            'atom_coords': atom_coords_np,
            'atom_elem': atom_elem_np
        }
        # import pdb;pdb.set_trace()
        return info
    
    def sort_residue(self, rank, ter_list):
        # chain_list = []
        new_child_list = []
        resid = 1 
        i = 0
        r_list = self.get_residues_list()
        for r in rank:
            res = r_list[r]
            # import pdb;pdb.set_trace()
            res.set_resid(resid)
            res.is_ter = ter_list[i]
            new_child_list.append(res)
            resid += 1
            i += 1
        self.child_list[0].child_list[0].child_list = new_child_list

        for id in self.get_chain_ids()[1:]:
            self.child_list[0].detach_child(id)

    
    def atom_to_internal_coordinates(self, verbose: bool = False) -> None:
        """Create/update internal coordinates from Atom X,Y,Z coordinates.

        Internal coordinates are bond length, angle and dihedral angles.

        :param verbose bool: default False
            describe runtime problems

        """
        for chn in self.get_chains():
            chn.atom_to_internal_coordinates(verbose)

    def internal_to_atom_coordinates(self, verbose: bool = False) -> None:
        """Create/update atom coordinates from internal coordinates.

        :param verbose bool: default False
            describe runtime problems

        :raises Exception: if any chain does not have .pic attribute
        """
        for chn in self.get_chains():
            chn.internal_to_atom_coordinates(verbose)
