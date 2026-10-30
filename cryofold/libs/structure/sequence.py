"""Implement utilities for handling protein sequences and vocabularies."""

import numpy as np
import torch
from .build_info import NUM_COORDS_PER_RES, NUM_ANGLES, ATOM_COUNTS


def trim_mask_and_true_seqs(mask_seq, true_seq):
    """Given an equal-length mask and sequence, removes gaps from the ends of both."""
    mask_seq_no_left = mask_seq.lstrip('-')
    mask_seq_no_right = mask_seq.rstrip('-')
    n_removed_left = len(mask_seq) - len(mask_seq_no_left)
    n_removed_right = len(mask_seq) - len(mask_seq_no_right)
    n_removed_right = None if n_removed_right == 0 else -n_removed_right
    true_seq = true_seq[n_removed_left:n_removed_right]
    mask_seq = mask_seq.strip("-")
    return mask_seq, true_seq


def empty_coord():
    from sidechainnet.utils.measure import GLOBAL_PAD_CHAR
    """Return an empty coordinate tensor representing 1 residue-level pad character."""
    coord_padding = np.zeros((NUM_COORDS_PER_RES, 3))
    coord_padding[:] = GLOBAL_PAD_CHAR
    return coord_padding


def empty_ang():
    from sidechainnet.utils.measure import GLOBAL_PAD_CHAR
    """Return an empty angle tensor representing 1 residue-level pad character."""
    dihe_padding = np.zeros(NUM_ANGLES)
    dihe_padding[:] = GLOBAL_PAD_CHAR
    return dihe_padding


def use_mask_to_pad_coords_dihedrals(mask_seq, coords, dihedrals):
    """Given a mask sequence ('-' for gap, '+' for present), and python lists of
    coordinates and dihedrals, this function places gaps in the relevant locations for
    each before returning.

    At the end, both should have the same length as the mask_seq.
    """
    new_coords = []
    new_angs = []
    coords = iter(coords)
    dihedrals = iter(dihedrals)
    for m in mask_seq:
        if m == "+":
            new_coords.append(next(coords))
            new_angs.append(next(dihedrals))
        else:
            new_coords.append(empty_coord())
            new_angs.append(empty_ang())
    return new_coords, new_angs


def bin_sequence_data(seqs, maxlen):
    """Given a list of sequences and a maximum training length, this function bins the
    sequences by their lengths (using numpy's 'auto' parameter), and then records the
    histogram information, as well as some statistics. This information is returned as a
    dictionary.

    This function allows the user to avoid computing this information at the
    start of each training run.
    """
    lens = list(map(lambda x: len(x) if len(x) <= maxlen else maxlen, seqs))
    hist_counts, hist_bins = np.histogram(lens, bins="auto")
    hist_bins = hist_bins[
        1:]  # make each bin define the rightmost value in each bin, ie '( , ]'.
    bin_probs = hist_counts / hist_counts.sum()
    bin_map = {}

    # Compute a mapping from bin number to index in dataset
    seq_i = 0
    bin_j = 0
    while seq_i < len(seqs):
        if lens[seq_i] <= hist_bins[bin_j]:
            try:
                bin_map[bin_j].append(seq_i)
            except KeyError:
                bin_map[bin_j] = [seq_i]
            seq_i += 1
        else:
            bin_j += 1

    return {
        "hist_counts": hist_counts,
        "hist_bins": hist_bins,
        "bin_probs": bin_probs,
        "bin_map": bin_map,
        "bin_max_len": maxlen
    }


class ProteinVocabulary(object):
    """Represents the 'vocabulary' of amino acids for encoding a protein sequence.

    Includes pad, sos, eos, and unknown characters as well as the 20 standard
    amino acids.
    """

    def __init__(self,
                 add_sos_eos=False,
                 include_unknown_char=False,
                 include_pad_char=True,
                 device=torch.device("cpu")):
        self.include_unknown_char = include_unknown_char
        self.include_pad_char = include_pad_char
        self.pad_char = "_"  # Pad character
        self.unk_char = "?"  # unknown character
        self.sos_char = "<"  # SOS character
        self.eos_char = ">"  # EOS character
        self.device = device

        self._char2int = dict()
        self._int2char = dict()
        # import pdb; pdb.set_trace()
        self._int2coordsmask = torch.zeros((21,14,3), device=self.device)
        for i in range(21):
            na = ATOM_COUNTS[i]
            self._int2coordsmask[i,:na,:] = torch.ones((na,3), device=self.device)

        # Extract the ordered list of 1-letter amino acid codes from the project-level
        # AA_MAP.
        self.stdaas = map(lambda x: x[0], sorted(list(AA_MAP.items()),
                                                 key=lambda x: x[1]))
        self.stdaas = "".join(filter(lambda x: len(x) == 1, self.stdaas))
        for aa in self.stdaas:
            self.add(aa)
        # import pdb; pdb.set_trace()

        if include_pad_char:
            self.add(self.pad_char)
            self.pad_id = self[self.pad_char]
        else:
            self.pad_id = 0  # Implicit padding with all-zeros
        if include_unknown_char:
            self.add(self.unk_char)
        if add_sos_eos:
            self.add(self.sos_char)
            self.add(self.eos_char)
            self.sos_id = self[self.sos_char]
            self.eos_id = self[self.eos_char]

    def __getitem__(self, aa):
        if self.include_unknown_char:
            return self._char2int.get(aa, self._char2int[self.unk_char])
        else:
            return self._char2int.get(aa, self._char2int[self.pad_char])

    def __contains__(self, aa):
        return aa in self._char2int

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self._char2int)

    def __repr__(self):
        return f"ProteinVocabulary[size={len(self)}]"

    def int2char(self, idx):
        return self._int2char[idx]

    def int2chars(self, idx):
        return ONE_TO_THREE_LETTER_MAP[self._int2char[idx]]

    def int2coordsmask(self, idx):
        return self._int2coordsmask[idx]

    def char2coordsmask(self, aa):
        return self._int2coordsmask[self[aa]]

    def add(self, aa):
        if aa not in self:
            aaid = self._char2int[aa] = len(self)
            self._int2char[aaid] = aa
            return aaid
        else:
            return self[aa]

    def str2ints(self, seq, add_sos_eos=True):
        if add_sos_eos:
            return [self["<"]] + [self[aa] for aa in seq] + [self[">"]]
        else:
            return [self[aa] for aa in seq]

    def ints2str(self, ints, include_sos_eos=False, exclude_pad=False):
        seq = ""
        for i in ints:
            c = self.int2char(i.item())
            if exclude_pad and c == self.pad_char:
                continue
            if include_sos_eos or (c not in [self.sos_char, self.eos_char, self.pad_char
                                            ]):
                seq += c
        return seq

    def str2coordsmask(self, seq):
        return [self.int2coordsmask(self[aa]) for aa in seq]

    def str2aa(self, seq):
        return "".join([aa if aa in ONE_TO_THREE_LETTER_MAP.keys() else "A" for aa in seq])

    def reformat_fasta(self, fastafile, outfile, aa_only=False):
        f = open(fastafile, 'r')
        lines = f.readlines()
        seq_list = []
        sequence = ""
        seq_name = ""
        seq_new = ""
        i=1
        for line in lines:
            if (line[0] == ">"):
                if (len(sequence) != 0):
                    # seq_list.append(sequence)
                    if aa_only:
                        seq = self.str2aa(sequence)
                    else:
                        seq = sequence
                    seq_new +=">{}|{}\n{}\n".format(seq_name,i, seq)
                    i+=1
                    sequence = ""
                seq_name=line[1:].rstrip('\n')
                continue
            seq_line = line.rstrip('\n')
            sequence += seq_line
        if (len(sequence) != 0):
            if aa_only:
                seq = self.str2aa(sequence)
            else:
                seq = sequence
            seq_new +=">{}|{}\n{}\n".format(seq_name,i, seq)
        f = open(outfile, "w")
        f.writelines(seq_new)
        f.close()
        return seq_list

    
    def read_fasta(self, fastafile):
        f = open(fastafile, 'r')
        lines = f.readlines()
        seq_list = []
        sequence=""
        for line in lines:
            if (line[0] == ">"):
                if (len(sequence) != 0):
                    seq_list.append(sequence)
                    sequence = ""
                continue
            name = line.rstrip('\n')
            sequence += name
        if (len(sequence) != 0):
            seq_list.append(sequence)
        return seq_list


    def create_new_fasta(self, outfile,seq_name,new_seq_list):
        sequences = ""
        for i in range(len(new_seq_list)):
            if len(new_seq_list[i])==0:
                continue
            sequences += ">{}|{}|seq_name={}\n{}\n".format("sp",i,seq_name,new_seq_list[i],)
        f = open(outfile, "w")
        f.writelines(sequences)
        f.close()
        print(outfile)


    def sse_str2ints(self, sse_str):
        sse = np.array(list(sse_str))
        classes = np.zeros((len(sse)))
        classes[sse=='H']=1
        classes[sse=='G']=1
        classes[sse=='I']=1
        classes[sse=='E']=2
        classes[sse=='C']=3
        return classes


ONE_TO_THREE_LETTER_MAP = {
    "R": "ARG",
    "H": "HIS",
    "K": "LYS",
    "D": "ASP",
    "E": "GLU",
    "S": "SER",
    "T": "THR",
    "N": "ASN",
    "Q": "GLN",
    "C": "CYS",
    "G": "GLY",
    "P": "PRO",
    "A": "ALA",
    "V": "VAL",
    "I": "ILE",
    "L": "LEU",
    "M": "MET",
    "F": "PHE",
    "Y": "TYR",
    "W": "TRP",
    "X": "UNK",
    "B": "BG_",
}
THREE_TO_ONE_LETTER_MAP = {v: k for k, v in ONE_TO_THREE_LETTER_MAP.items()}

# CryoNet rank
# restypes = [
#     "B", "A", "G", "I", "L", "P", "V", "F", "W", "Y", "D", "E", "R", "H", "K", 
#     "S", "T", "C", "M", "N", "Q",
# ]

# Alphafold2 rank
restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = restypes + ['X']
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}

AA_MAP = restype_order_with_x

id_residue_dict = {i:ONE_TO_THREE_LETTER_MAP[restypes_with_x[i]]  for i, restype in enumerate(restypes_with_x)}


AA_MAP_INV = {v: k for k, v in AA_MAP.items()}

for one_letter_code in list(AA_MAP.keys()):
    AA_MAP[ONE_TO_THREE_LETTER_MAP[one_letter_code]] = AA_MAP[one_letter_code]

# TODO: create VOCAB object only when needed.
# VOCAB = ProteinVocabulary()

RESTYPE_NAME_TO_ATOM14_NAMES = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    'GLY': ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'UNK': ['N', 'CA', 'C', 'O', 'CB',  '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'BG_': ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],

}
