# Copyright (C) 2002, Thomas Hamelryck (thamelry@binf.ku.dk)
#
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.

"""Residue class, used by Structure objects."""

# My Stuff
import warnings
from .PDBExceptions import PDBConstructionException
from .Entity import Entity, DisorderedEntityWrapper
from cryofold.libs.structure.sequence import AA_MAP,id_residue_dict, RESTYPE_NAME_TO_ATOM14_NAMES
from copy import deepcopy


import numpy as np
_atom_name_dict = {}
_atom_name_dict["N"] = 1
_atom_name_dict["CA"] = 2
_atom_name_dict["C"] = 3
_atom_name_dict["O"] = 4

# 0: BG
# cryonet version
ATOM_COUNTS_cn = [4, 5, 4, 8, 8, 7, 7, 11, 14, 12, 8, 9, 11, 10, 9, 6, 7, 6, 8, 8, 9, 4]
# af2 version
ATOM_COUNTS = [5, 11, 8, 8, 6, 9, 9, 4, 10, 8, 8, 9, 8, 11, 7, 6, 7, 14, 12, 7, 4]

 
REPLACEMENTS = {
        "ASX": "ASP",
        "GLX": "GLU",
        "MEQ": "GLN",
        "CSO": "CYS",
        "HIP": "HIS",
        "HSD": "HIS",
        "HSE": "HIS",
        "HSP": "HIS",
        "HSO": "HIS",
        "MSE": "MET",
        "FME": "MET",
        "SEC": "CYS",
        "SEP": "SER",
        "TPO": "THR",
        "PTR": "TYR",
        "XLE": "LEU",
        "4FB": "PRO",
        "MLY": "LYS"  # N-dimethyl-lysine
    }

# for i in range(21):
#     ATOM_COUNTS.append( ATOM_COUNTS_cn[ RESIDUE_ID_DICT[id_residue_dict[i]] ])
# print(ATOM_COUNTS)
# import pdb;pdb.set_trace()

AAMAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'UNK': 'X',
    'ASX': 'B', 'GLX': 'Z', 'SEC': 'U', 'PYL': 'O', 'XLE': 'J', '': '-'
}

# add bases
# AAMAP.update({'ADE': 'a', 'THY': 't', 'CYT': 'c', 'GUA': 'g', 'URA': 'u'})
AAMAP.update({'  A': 'a', '  T': 't', '  C': 'c', '  G': 'g', '  U': 'u',
              'A  ': 'a', 'T  ': 't', 'C  ': 'c', 'G  ': 'g', 'U  ': 'u',
              'A  ': 'a', 'T  ': 't', 'C  ': 'c', 'G  ': 'g', 'U  ': 'u',
              'DA ': 'a', 'DT ': 't', 'DC ': 'c', 'DG ': 'g',
              ' DA': 'a', ' DT': 't', ' DC': 'c', ' DG': 'g',
              })

# add reversed mapping
_ = {}
for aaa, a in AAMAP.items():
    _[a] = aaa
AAMAP.update(_)

re = {}
for aaa, a in REPLACEMENTS.items():
    re[aaa] = AAMAP[a]
AAMAP.update(_)

int2coordsmask = np.zeros((36,14,3))
int2coordsmask[22:]=np.ones((14,14,3))
for i in range(21):
    na = ATOM_COUNTS[i]
    int2coordsmask[i,:na,:] = np.ones((na,3))

# cryonet
# def seq_coords_mask(seq):
#     return [int2coordsmask[RESIDUE_ID_DICT[ AAMAP[aa]]] for aa in seq]

# def seq_to_ids(seq):
#     return [RESIDUE_ID_DICT[ AAMAP[aa]] for aa in seq]

def seq_coords_mask(seq):
    return [int2coordsmask[AA_MAP.get(aa, 0)] for aa in seq]

def seq_to_ids(seq):
    return [AA_MAP.get(aa, 0) for aa in seq]


class Residue(Entity):
    """Represents a residue. A Residue object stores atoms."""

    def __init__(self, id, resname, segid):
        """Initialize the class."""
        self.level = "R"
        self.disordered = 0
        self.resname = resname
        self.segid = segid
        self.is_ter = 0
        self.is_het = 0
        self.ssType = 3
        self.internal_coord = None
        Entity.__init__(self, id)
        hetflag, resseq, icode = self.get_id()
        if hetflag.strip() !="":
            self.is_het = 1
        

    def __repr__(self):
        """Return the residue full id."""
        resname = self.get_resname()
        hetflag, resseq, icode = self.get_id()
        full_id = (resname, hetflag, resseq, icode)
        return "<Residue %s het=%s resseq=%s icode=%s>" % full_id

    def add(self, atom):
        """Add an Atom object.

        Checks for adding duplicate atoms, and raises a
        PDBConstructionException if so.
        """
        atom_id = atom.get_id()
        if self.has_id(atom_id):
            raise PDBConstructionException(
                "Atom %s defined twice in residue %s" % (atom_id, self)
            )
        Entity.add(self, atom)
    
    def set_resid(self, resid):
        hetflag, resseq, icode = self.get_id()
        self._id = (hetflag, resid, icode)


    def sort(self):
        """Sort child atoms.

        Atoms N, CA, C, O always come first, thereafter alphabetically
        by name, with any alternative location specifier for disordered
        atoms (altloc) as a tie-breaker.
        """
        # warnings.warn(
        #     "The custom sort() method will be removed in the future in favour of rich "
        #     "comparison methods. Use the built-in sorted() function instead.",
        #     BiopythonDeprecationWarning,
        # )
        self.child_list.sort()

    def flag_disordered(self):
        """Set the disordered flag."""
        self.disordered = 1

    def is_disordered(self):
        """Return 1 if the residue contains disordered atoms."""
        return self.disordered

    def set_disordered(self, flag):
        """Return 1 if the residue contains disordered atoms."""
        for a in self.child_list:
            a.disordered_flag = flag

    def het_to_atom(self):
        hetflag, resseq, icode = self.get_id()
        self.id = (" ", resseq, icode)
        self.is_het = 0

    def clear_atloc(self):
        """set the residue name."""
        for a in self.child_list:
            a.set_altloc(" ")

    def set_resname(self, resname):
        """set the residue name."""
        self.resname = resname

    def get_resname(self):
        """Return the residue name."""
        return self.resname

    def get_resname1c(self):
        """Return the residue name."""
        return AAMAP.get(self.resname, 'A')
    
    def get_resnum(self):
        return self.get_id()[1]
    
    def get_resint(self):
        # return RESIDUE_ID_DICT[self.get_resname()]
        return AA_MAP[self.get_resname()]

    def get_chainid(self):
        """Return the residue name."""
        return self.parent.id

    def get_unpacked_list(self):
        """Return the list of all atoms, unpack DisorderedAtoms."""
        atom_list = self.get_list()
        undisordered_atom_list = []
        for atom in atom_list:
            if atom.is_disordered():
                undisordered_atom_list += atom.disordered_get_list()
            else:
                undisordered_atom_list.append(atom)
        return undisordered_atom_list

    def get_segid(self):
        """Return the segment identifier."""
        return self.segid

    def get_atoms(self):
        """Return atoms."""
        yield from self

    def get_atom_by_name(self, name):
        """Return atoms."""
        for a in self.get_atoms():
            if a.get_id() == name:
                return a
    
    def set_sse(self, sse):
        sse = int(sse)
        assert sse in [1,2,3], "sse type ({}) error!".format(sse)
        self.ssType = sse

    def get_sse(self):
        return int(self.ssType)

    def get_bfactor(self):
        return self.child_list[0].get_bfactor()

    def get_occupancy(self):
        return self.child_list[0].get_occupancy()

    def get_coords_cn(self):
        """ padding
        """
        coords = []
        aelems = []
        
        if len(self.child_list)==1:
            coords = [(0.0,0.0,0.0)] * 4
            coords[1] = self.child_list[0].get_coord()
            aelems = ['N','C','C','O']
        else:
            for atom in self.child_list:
                coords.append(atom.get_coord())
                aelems.append(atom.element)
        self.coords_list = deepcopy(coords)
        self.aelems_list = deepcopy(aelems)

        if len(coords)<14:
            coords.extend([(0.0,0.0,0.0)] * (14-len(coords)))
        if len(coords)>14:
            coords = coords[:14]
        return coords

    def get_coords(self):
        """ padding
        """
        coords = [(0.0,0.0,0.0)] * 14
        aelems = []
        
        if len(self.child_list)==1:
            coords = [(0.0,0.0,0.0)] * 4
            coords[1] = self.child_list[0].get_coord()
            aelems = ['N','C','C','O']
        else:
            for atom in self.child_list:
                if self.resname not in RESTYPE_NAME_TO_ATOM14_NAMES.keys() or atom.name not in RESTYPE_NAME_TO_ATOM14_NAMES[self.resname]:
                    continue
                ind = RESTYPE_NAME_TO_ATOM14_NAMES[self.resname].index(atom.name)
                coords[ind] = atom.get_coord()
                aelems.append(atom.element)
        self.coords_list = deepcopy(coords)
        self.aelems_list = deepcopy(aelems)
        return coords

    def get_coords_list(self):
        return self.coords_list

    def get_atomelem_list(self):
        return self.aelems_list
    
    def get_box(self, radius=1):
        points = np.array(self.coords_list)
        box = [
                np.min(points[:, 0]) - radius,
                np.min(points[:, 1]) - radius,
                np.min(points[:, 2]) - radius,
                np.max(points[:, 0]) + radius,
                np.max(points[:, 1]) + radius,
                np.max(points[:, 2]) + radius
            ]
        return box
    
    # def get_atom(self):
    #     """Return atom."""
    #     warnings.warn(
    #         "`get_atom` has been deprecated and we intend to remove it"
    #         " in a future release of Biopython. Please use `get_atoms` instead.",
    #         BiopythonDeprecationWarning,
    #     )
    #     yield from self

    def remove_nne_children(self):
        rm_list = []
        for a in self.child_list:
            # if a.get_name() not in  RESIDUE_ATOM_NAME_DICT[self.resname.upper()]:
            if a.get_name() not in  RESTYPE_NAME_TO_ATOM14_NAMES[self.resname.upper()]:
                rm_list.append(a.get_id())
        for i in rm_list:
            self.detach_child(i)

    def mutate(self, res_name):
        # 1.mutate 
        self.set_resname(res_name)
        # 2.replace het
        self.het_to_atom()
        # 3.remove extra atoms
        self.remove_nne_children()


class DisorderedResidue(DisorderedEntityWrapper):
    """DisorderedResidue is a wrapper around two or more Residue objects.

    It is used to represent point mutations (e.g. there is a Ser 60 and a Cys 60
    residue, each with 50 % occupancy).
    """

    def __init__(self, id):
        """Initialize the class."""
        DisorderedEntityWrapper.__init__(self, id)

    def __repr__(self):
        """Return disordered residue full identifier."""
        resname = self.get_resname()
        hetflag, resseq, icode = self.get_id()
        full_id = (resname, hetflag, resseq, icode)
        return "<DisorderedResidue %s het=%s resseq=%i icode=%s>" % full_id

    def add(self, atom):
        """Add atom to residue."""
        residue = self.disordered_get()
        if not atom.is_disordered() == 2:
            # Atoms in disordered residues should have non-blank
            # altlocs, and are thus represented by DisorderedAtom objects.
            resname = residue.get_resname()
            het, resseq, icode = residue.get_id()
            # add atom anyway, if PDBParser ignores exception the atom will be part of the residue
            residue.add(atom)
            raise PDBConstructionException(
                "Blank altlocs in duplicate residue %s (%s, %i, %s)"
                % (resname, het, resseq, icode)
            )
        residue.add(atom)

    def sort(self):
        """Sort the atoms in the child Residue objects."""
        for residue in self.disordered_get_list():
            residue.sort()

    def disordered_add(self, residue):
        """Add a residue object and use its resname as key.

        Arguments:
         - residue - Residue object

        """
        resname = residue.get_resname()
        # add chain parent to residue
        chain = self.get_parent()
        residue.set_parent(chain)
        assert not self.disordered_has_id(resname)
        self[resname] = residue
        self.disordered_select(resname)
