# Copyright (C) 2002, Thomas Hamelryck (thamelry@binf.ku.dk)
#
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.

"""Chain class, used in Structure objects."""

from .Entity import Entity
from .internal_coords import IC_Chain

from typing import Optional


class Chain(Entity):
    """Define Chain class.

    Chain is an object of type Entity, stores residues and includes a method to
    access atoms from residues.
    """

    def __init__(self, id):
        """Initialize the class."""
        self.level = "C"
        self.internal_coord = None
        Entity.__init__(self, id)

    # Sorting methods: empty chain IDs come last.
    def __gt__(self, other):
        """Validate if id is greater than other.id."""
        if isinstance(other, Chain):
            if self.id == " " and other.id != " ":
                return 0
            elif self.id != " " and other.id == " ":
                return 1
            else:
                return self.id > other.id
        else:
            return NotImplemented

    def __ge__(self, other):
        """Validate if id is greater or equal than other.id."""
        if isinstance(other, Chain):
            if self.id == " " and other.id != " ":
                return 0
            elif self.id != " " and other.id == " ":
                return 1
            else:
                return self.id >= other.id
        else:
            return NotImplemented

    def __lt__(self, other):
        """Validate if id is less than other.id."""
        if isinstance(other, Chain):
            if self.id == " " and other.id != " ":
                return 0
            elif self.id != " " and other.id == " ":
                return 1
            else:
                return self.id < other.id
        else:
            return NotImplemented

    def __le__(self, other):
        """Validate if id is less or equal than other id."""
        if isinstance(other, Chain):
            if self.id == " " and other.id != " ":
                return 0
            elif self.id != " " and other.id == " ":
                return 1
            else:
                return self.id <= other.id
        else:
            return NotImplemented

    def _translate_id(self, id):
        """Translate sequence identifier to tuple form (PRIVATE).

        A residue id is normally a tuple (hetero flag, sequence identifier,
        insertion code). Since for most residues the hetero flag and the
        insertion code are blank (i.e. " "), you can just use the sequence
        identifier to index a residue in a chain. The _translate_id method
        translates the sequence identifier to the (" ", sequence identifier,
        " ") tuple.

        Arguments:
         - id - int, residue resseq

        """
        if isinstance(id, int):
            id = (" ", id, " ")
        return id

    def __getitem__(self, id):
        """Return the residue with given id.

        The id of a residue is (hetero flag, sequence identifier, insertion code).
        If id is an int, it is translated to (" ", id, " ") by the _translate_id
        method.

        Arguments:
         - id - (string, int, string) or int

        """
        id = self._translate_id(id)
        return Entity.__getitem__(self, id)

    def __contains__(self, id):
        """Check if a residue with given id is present in this chain.

        Arguments:
         - id - (string, int, string) or int

        """
        id = self._translate_id(id)
        return Entity.__contains__(self, id)

    def __delitem__(self, id):
        """Delete item.

        Arguments:
         - id - (string, int, string) or int

        """
        id = self._translate_id(id)
        return Entity.__delitem__(self, id)

    def __repr__(self):
        """Return the chain identifier."""
        return "<Chain id=%s>" % self.get_id()

    # Public methods

    def get_unpacked_list(self):
        """Return a list of undisordered residues.

        Some Residue objects hide several disordered residues
        (DisorderedResidue objects). This method unpacks them,
        ie. it returns a list of simple Residue objects.
        """
        unpacked_list = []
        for residue in self.get_list():
            if residue.is_disordered() == 2:
                for dresidue in residue.disordered_get_list():
                    unpacked_list.append(dresidue)
            else:
                unpacked_list.append(residue)
        return unpacked_list

    def has_id(self, id):
        """Return 1 if a residue with given id is present.

        The id of a residue is (hetero flag, sequence identifier, insertion code).

        If id is an int, it is translated to (" ", id, " ") by the _translate_id
        method.

        Arguments:
         - id - (string, int, string) or int

        """
        id = self._translate_id(id)
        return Entity.has_id(self, id)

    # Public
    def get_sequence(self, gap="-"):
        """Return a list of sequences (one for each disordered region)."""
        sequence = ""
        resid = - 99999
        for residue in self.get_unpacked_list():
            resid_ = residue.get_resnum()
            if resid != - 99999:
                if resid_ != resid + 1:
                    sequence = sequence + gap * (resid_ - resid - 1)
            resid = resid_

            # import pdb;pdb.set_trace()
            sequence = sequence+residue.get_resname1c()
        return sequence

    def get_residues(self):
        """Return residues."""
        yield from self

    def get_atoms(self):
        """Return atoms from residues."""
        for r in self.get_residues():
            yield from r

    def atom_to_internal_coordinates(self, verbose: bool = False) -> None:
        """Create/update internal coordinates from Atom X,Y,Z coordinates.

        Internal coordinates are bond length, angle and dihedral angles.

        :param verbose bool: default False
            describe runtime problems
        """
        if not self.internal_coord:
            self.internal_coord = IC_Chain(self, verbose)
        self.internal_coord.atom_to_internal_coordinates(verbose=verbose)

    def internal_to_atom_coordinates(
        self,
        verbose: bool = False,
        start: Optional[int] = None,
        fin: Optional[int] = None,
    ):
        """Create/update atom coordinates from internal coordinates.

        :param verbose bool: default False
            describe runtime problems
        :param: start, fin lists
            sequence position, insert code for begin, end of subregion to
            process
        :raises Exception: if any chain does not have .pic attribute
        """
        if self.internal_coord:
            self.internal_coord.internal_to_atom_coordinates(
                verbose=verbose, start=start, fin=fin
            )
        else:
            raise Exception(
                "Structure %s Chain %s does not have internal coordinates set"
                % (self.parent.parent, self)
            )

    def sort_residue(self, rank, ter_list):
        new_child_list = []
        resid = 1 
        i = 0
        for r in rank:
            res = self.child_list[r]
            # import pdb;pdb.set_trace()
            res.set_resid(resid)
            res.is_ter = ter_list[i]
            new_child_list.append(res)
            resid += 1
            i += 1
        self.child_list = new_child_list