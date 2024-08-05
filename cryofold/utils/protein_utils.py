
import dataclasses
from copy import deepcopy
import logging
import torch

import os, sys, time, re, random, pickle, gzip, io, configparser, shutil, pathlib, tempfile, hashlib, argparse, json, inspect
from typing import Dict, Union, Optional
import numpy as np

cur_path = str(pathlib.Path(__file__).parent.resolve())

from cryofold.np import protein, residue_constants

# _e = lambda x: Colors.f(x, 'red')
# _w = lambda x: Colors.f(x, 'cyan')
restype_1to3 = residue_constants.restype_1to3
restype_3to1 = residue_constants.restype_3to1
restypes_with_x = residue_constants.restypes_with_x
PDB_CHAIN_IDS = protein.PDB_CHAIN_IDS
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)



@dataclasses.dataclass(frozen=True)
class Protein:
  """Protein structure representation."""

  # Cartesian coordinates of atoms in angstroms. The atom types correspond to
  # residue_constants.atom_types, i.e. the first three are N, CA, CB.
  atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

  # Amino-acid type for each residue represented as an integer between 0 and
  # 20, where 20 is 'X'.
  aatype: np.ndarray  # [num_res]

  # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
  # is present and 0.0 if not. This should be used for loss masking.
  atom_mask: np.ndarray  # [num_res, num_atom_type]

  # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
  residue_index: np.ndarray  # [num_res]

  # 0-indexed number corresponding to the chain in the protein that this residue
  # belongs to.
  chain_index: np.ndarray  # [num_res]

  # B-factors, or temperature factors, of each residue (in sq. angstroms units),
  # representing the displacement of the residue from its ground truth mean
  # value.
  b_factors: np.ndarray  # [num_res, num_atom_type]

  # Secondary structure. 1: Loop, 2: Helix, 3: Sheet
  occupancies: np.ndarray = None  # [num_res, num_atom_type]


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')
          
def to_pdb(prot: Protein, gap_ter_threshold=600.0) -> str:
  """Converts a `Protein` instance to a PDB string.

  Args:
    prot: The protein to convert to PDB.

  Returns:
    PDB string.
  """
  restypes = residue_constants.restypes + ['X']
  res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], 'UNK')
  atom_types = residue_constants.atom_types

  pdb_lines = []

  atom_mask = prot.atom_mask
  aatype = prot.aatype
  atom_positions = prot.atom_positions
  residue_index = prot.residue_index.astype(np.int32)
  chain_index = prot.chain_index.astype(np.int32)
  b_factors = prot.b_factors
  occupancies = prot.occupancies
  if occupancies is None:
    occupancies = np.ones_like(b_factors)

  if np.any(aatype > residue_constants.restype_num):
    raise ValueError('Invalid aatypes.')

  # Construct a mapping from chain integer indices to chain ID strings.
  chain_ids = {}
  for i in np.unique(chain_index):  # np.unique gives sorted output.
    if i >= PDB_MAX_CHAINS:
      raise ValueError(
          f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
    chain_ids[i] = PDB_CHAIN_IDS[i]

  pdb_lines.append('MODEL     1')
  atom_index = 1
  last_chain_index = chain_index[0]
  last_res_ca_xyz = atom_positions[0, 1] if atom_mask[0, 1] == 1.0 else None
  # Add all atom sites.
  for i in range(aatype.shape[0]):
    # Close the previous chain if in a multichain PDB.
    if last_chain_index != chain_index[i]:
      pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]], residue_index[i - 1]))
      last_chain_index = chain_index[i]
      atom_index += 1  # Atom index increases at the TER symbol.
    elif last_res_ca_xyz is not None and atom_mask[i, 1] == 1.0 and \
        np.sqrt(np.sum((last_res_ca_xyz - atom_positions[i, 1])**2)) >= gap_ter_threshold:
      pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]], residue_index[i - 1]))
      last_chain_index = chain_index[i]
      atom_index += 1  # Atom index increases at the TER symbol.
    last_res_ca_xyz = atom_positions[i, 1] if atom_mask[i, 1] == 1.0 else None

    res_name_3 = res_1to3(aatype[i])
    # import pdb;pdb.set_trace()
    for atom_name, pos, mask, b_factor, occupancy in zip(
        atom_types, atom_positions[i], atom_mask[i], b_factors[i], occupancies[i]):
      if mask < 0.5:
        continue

      record_type = 'ATOM'
      name = atom_name if len(atom_name) == 4 else f' {atom_name}'
      alt_loc = ''
      insertion_code = ''
      occupancy = occupancy[0]
      element = atom_name[0]  # Protein supports only C, N, O, S, this works.
      charge = ''
      # PDB is a columnar format, every space matters here!
    #   import pdb;pdb.set_trace()
      atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                   f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                   f'{residue_index[i]:>4}{insertion_code:>1}   '
                   f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                   f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                   f'{element:>2}{charge:>2}')
      pdb_lines.append(atom_line)
      atom_index += 1

  # Close the final chain.
  pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[-1]), chain_ids[chain_index[-1]], residue_index[-1]))
  pdb_lines.append('ENDMDL')
  pdb_lines.append('END')

  # Pad all lines to 80 characters.
  pdb_lines = [line.ljust(80) for line in pdb_lines]
  return '\n'.join(pdb_lines) + '\n'  # Add terminating newline.



######################
## Util functions
######################

def extend_PDB_Chains(extend_to=500):
    """
    Extend the PDB chains range

    Parameter
    ---------------
    extend_to: int, Extend the chain name range to given range

    Return
    ---------------
    PDB_CHAIN_IDS: New PDB Chain ID string
    """
    from alphafold.common import protein
    protein.extend_PDB_Chains(extend_to)
    global PDB_CHAIN_IDS
    PDB_CHAIN_IDS = protein.PDB_CHAIN_IDS

def func_has_agu(func, agu):
    param_keys = list(inspect.signature(func).parameters.keys())
    return agu in param_keys

def reset_af2_default_tmpdir(def_tmp_dir):
    import alphafold.data.tools.utils
    from typing import Optional
    import tempfile, contextlib, shutil
    
    @contextlib.contextmanager
    def tmpdir_manager(base_dir: Optional[str] = def_tmp_dir):
      """Context manager that deletes a temporary directory on exit."""
      tmpdir = tempfile.mkdtemp(dir=base_dir)
      try:
        yield tmpdir
      finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    alphafold.data.tools.utils.tmpdir_manager = tmpdir_manager

def write_pdb_seq(sequence: str, chain_id: str):
    """
    Write Sequence to PDB SEQRES field

    Return
    -----------
    seqres_lines: list
        Lines to save in PDB file. e.g. SEQRES   1 A   21  GLY ILE VAL GLU GLN CYS CYS THR SER ILE CYS SER LEU
    """
    seq_len = len(sequence)
    assert seq_len <= 9999
    assert len(chain_id) == 1
    seqres_lines = []
    for row_idx, start_idx in enumerate(range(0, seq_len, 13)):
        seq_1 = sequence[start_idx:start_idx+13]
        seq_3 = [ restype_1to3.get(res, 'UNK') for res in seq_1 ]
        line = f"SEQRES {row_idx+1:3d} {chain_id} {seq_len:4d}  {' '.join(seq_3)}"
        seqres_lines.append(line)
    return seqres_lines

def read_pdb_seq(pdb_file: str):
    """
    Read SEQRES field from PDB file

    Return
    -----------
    seq_dict: dict
        Dict of chain_id to seq. e.g. {'A': 'EYTISHTGGTLGSSKVTTA'}
    """
    seq_dict = {}
    seq_len_dict = {}
    for line in open(pdb_file):
        if line.startswith('SEQRES '):
            chain_id = line[11]
            seq_len = int(line[13:13+4].strip())
            frag = "".join([ restype_3to1.get(res, 'X') for res in line[19:].strip().split() ])
            seq_dict[chain_id] = seq_dict.get(chain_id, '') + frag
            seq_len_dict[chain_id] = seq_len
    for chain_id in seq_len_dict:
        if seq_len_dict[chain_id] != len(seq_dict[chain_id]):
            print(f"Warning: expect sample length for chain {chain_id}, but got {seq_len_dict[chain_id]} and {len(seq_dict[chain_id])}")
    return seq_dict

def read_pdb_atom_line(atom_line: str):
    """
    Parse ATOM line of PDB file

    Return
    ---------------
    list: [atom_idx, atom_name, restype, chain, res_index, x, y, z, occ, temp ]
    """
    if len(atom_line) > 78:
        atom_line = atom_line.strip()
    assert atom_line.startswith('ATOM')
    atom_idx = atom_line[6:11].strip()
    atom_name = atom_line[12:16].strip()
    restype = atom_line[17:20].strip()
    chain = atom_line[21]
    res_index = atom_line[22:26].strip()
    x, y, z = float(atom_line[30:38].strip()), float(atom_line[38:46].strip()), float(atom_line[46:54].strip())
    occ = float(atom_line[54:60].strip())
    temp = float(atom_line[60:66].strip())
    return [atom_idx, atom_name, restype, chain, res_index, x, y, z, occ, temp ]

def write_pdb_atom_line(idx: int, atom_name: str, restype: str, chain: str, res_index: int,
                        x: float, y: float, z: float, occ: float = 0.0, temp: float = 0.0):
    """
    Convert to atom 
    """
    assert len(atom_name) <= 4
    assert len(restype) == 3
    assert len(chain) == 1
    assert len(str(res_index)) <= 4
    atom_mark = atom_name[0]
    if res_index < 1:
        print(f"Warning: res_index should greater than 0")
    return f"ATOM  {idx:5d} {atom_name.center(4)} {restype} {chain}{res_index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{temp:6.2f}           {atom_mark:1s}"

def write_pdb_helix_line(serial: int, identifier: str, resname1: str, chain1: str, res_index1: int,
                        resname2: str, chain2: str, res_index2: int):
    """
    HELIX    1 AA1 THR A   90  GLY A  109  1                                  20
    """
    assert len(str(serial)) <= 3
    assert len(identifier) <= 3
    assert len(resname1) == len(resname2) == 3
    assert len(chain1) == len(chain2) == 1
    assert len(str(res_index1)) <= 4
    assert len(str(res_index2)) <= 4

    length = res_index2 - res_index1 + 1
    return f"HELIX  {serial:3d} {identifier:3s} {resname1} {chain1} {res_index1:4d}  {resname2} {chain2} {res_index2:4d}  1                               {length:5d}"

def write_pdb_sheet_line(serial: int, identifier: str, resname1: str, chain1: str, res_index1: int,
                         resname2: str, chain2: str, res_index2: int):
    """
    SHEET    1 AA1 3 PHE A 125  LYS A 127  0
    """
    assert len(str(serial)) <= 3
    assert len(identifier) <= 3
    assert len(resname1) == len(resname2) == 3
    assert len(chain1) == len(chain2) == 1
    assert len(str(res_index1)) <= 4
    assert len(str(res_index2)) <= 4

    length = res_index2 - res_index1 + 1
    assert len(str(length)) <= 2
    return f"SHEET  {serial:3d} {identifier:3s}{length:2d} {resname1} {chain1}{res_index1:4d}  {resname2} {chain2}{res_index2:4d}  0"

def save_as_pdb(aatype, residue_index, atom_positions, atom_position_mask, out_file,
                b_factors=None, asym_id=None, full_seq=None, occupancies=None, ss_coding=[1, 2, 3], 
                write_ss=False, gap_ter_threshold=600):
    """
    Save PDB file from positions
    
    Warning
    -----------
    residue_index must be zero-based

    Parameters
    -----------
    aatype: [N_res]
    residue_index: [N_res]
    atom_positions: [N_res, 37, 3]
    atom_position_mask: [N_res, 37]
    out_file: str
    b_factors: [N_res, 37] or None
    occupancies: occupancies of protein
    asym_id: [N_res] or None
    full_seq: str or dict
        full sequence, write to SEQRES field
        str type: single chain
        dict type: chain_id -> str mapping
    ss_coding: 3 elements, for index of loop, helix and sheet
    write_ss: bool
        Write the SS information
    """
    
    
    if aatype.shape[0] == 1:
        aatype = aatype[0]

    if residue_index.shape[0] == 1:
        residue_index = residue_index[0]
    assert aatype.ndim == 1
    assert residue_index.ndim == 1
    assert atom_positions.ndim == 3
    assert atom_position_mask.ndim == 2
    if b_factors is not None:
        assert b_factors.ndim == 2
    else:
        b_factors = np.zeros_like(atom_position_mask)
    if asym_id is not None:
        if asym_id.shape[0] == 1:
            asym_id = asym_id[0]
        assert asym_id.ndim == 1
    else:
        asym_id = np.zeros_like(aatype)
    asym_id = asym_id.astype(np.int32)
    if write_ss:
        assert len(ss_coding) == 3
        assert occupancies is not None
    if full_seq is not None:
        assert isinstance(full_seq, (str, dict))
    
    ## Convert call from from_prediction to protein.Protein
    unrelaxed_protein = Protein(atom_positions, aatype, atom_position_mask, residue_index+1, asym_id, b_factors, occupancies)

    def check_seq(ch_full_seq, ch_aatype, ch_residx):
        for res_aatype, res_residx in zip(ch_aatype, ch_residx):
            fseq_res = ch_full_seq[res_residx]
            aatype_res = restypes_with_x[res_aatype]
            if fseq_res != aatype_res:
                print(_w(f"Warning: different sequence for full_seq and aatype: {fseq_res} -- {aatype_res}"))

    SEQRES = []
    if full_seq is not None:
        if isinstance(full_seq, str):
            assert len(set(asym_id)) == 1
            check_seq(full_seq, aatype, residue_index)
            SEQRES = write_pdb_seq(full_seq, protein.PDB_CHAIN_IDS[ asym_id[0] ])
        else:
            if len(set(asym_id) - set(full_seq.keys())) > 0:
                no_seq_chain_id = ",".join([ str(key) for key in list(set(asym_id) - set(full_seq.keys())) ])
                print(f"Warning: seq of chains have no sequence -- {no_seq_chain_id}")
            for ch_idx in sorted(full_seq.keys()):
                mask = (ch_idx == asym_id)
                if mask.sum() > 0:
                    check_seq(full_seq[ch_idx], aatype[mask], residue_index[mask])
                SEQRES += write_pdb_seq(full_seq[ch_idx], protein.PDB_CHAIN_IDS[ch_idx])

    if write_ss:
        helix_lines, sheet_lines = parse_prot_ss(unrelaxed_protein, ss_coding=ss_coding)
        ss_line = "\n".join(helix_lines) + "\n" + "\n".join(sheet_lines) + "\n"
    else:
        ss_line = ""

    if SEQRES is None or len(SEQRES) == 0:
        SEQRES = ""
    else:
        SEQRES = "\n".join(SEQRES) + "\n"

    pdb_str = SEQRES + ss_line + to_pdb(unrelaxed_protein, gap_ter_threshold=gap_ter_threshold)
    print(pdb_str, file=open(out_file, 'w'))

def parse_prot_ss(prot, ss_coding=[1, 2, 3]):

    L_TYPE = ss_coding[0]
    H_TYPE = ss_coding[1]
    S_TYPE = ss_coding[2]
    cur_ss_type = None
    start_ch = None
    start_res_idx = None
    last_res_idx = None
    start_restype = None
    last_restype = None

    helix_lines = []
    sheet_lines = []
    for ch_idx, res_idx, aatype, occ in zip(prot.chain_index, prot.residue_index, prot.aatype, prot.occupancies[:,0,0]):
        #ch_idx, res_idx, aatype, occ = ch_idx.item(), res_idx.item(), aatype.item(), occ.item()
        # import pdb;pdb.set_trace()
        if int(occ) != cur_ss_type or start_ch != ch_idx or abs(res_idx - last_res_idx) > 3:
            if cur_ss_type == H_TYPE and last_res_idx - start_res_idx >= 3:
                helix_lines.append([start_ch, start_res_idx, start_restype, last_res_idx, last_restype])
            elif cur_ss_type == S_TYPE and last_res_idx - start_res_idx >= 3:
                sheet_lines.append([start_ch, start_res_idx, start_restype, last_res_idx, last_restype])
            start_res_idx, start_restype, start_ch = res_idx, restype_1to3.get(restypes_with_x[aatype], 'UNK'), ch_idx
            cur_ss_type = int(occ)
        last_res_idx, last_restype = int(res_idx), restype_1to3.get(restypes_with_x[aatype], 'UNK')
    if cur_ss_type == H_TYPE and last_res_idx - start_res_idx >= 3:
        helix_lines.append([start_ch, start_res_idx, start_restype, last_res_idx, last_restype])
    elif cur_ss_type == S_TYPE and last_res_idx - start_res_idx >= 3:
        sheet_lines.append([start_ch, start_res_idx, start_restype, last_res_idx, last_restype])
    
    for idx in range(len(helix_lines)):
        start_ch, start_res_idx, start_restype, last_res_idx, last_restype = helix_lines[idx]
        ch_name = PDB_CHAIN_IDS[start_ch]
        helix_lines[idx] = write_pdb_helix_line(min(idx+1,999), 'AA1', start_restype, ch_name, start_res_idx, last_restype, ch_name, last_res_idx)
    for idx in range(len(sheet_lines)):
        start_ch, start_res_idx, start_restype, last_res_idx, last_restype = sheet_lines[idx]
        ch_name = PDB_CHAIN_IDS[start_ch]
        sheet_lines[idx] = write_pdb_sheet_line(min(idx+1,999), 'AA1', start_restype, ch_name, start_res_idx, last_restype, ch_name, last_res_idx)
    
    return helix_lines, sheet_lines

def save_result_dict_as_pdb(aatype, prediction_result, out_file):
    ### save result as PDB

    if aatype.ndim == 2:
        aatype = aatype.argmax(-1)
    residue_index = np.arange(aatype.shape[0])
    atom_positions = prediction_result['structure_module']['final_atom_positions']
    atom_position_mask = prediction_result['structure_module']['final_atom_mask']
    b_factors = np.repeat(prediction_result['plddt'][:, None], residue_constants.atom_type_num, axis=-1)
    save_as_pdb(aatype, residue_index, atom_positions, atom_position_mask, out_file, b_factors=b_factors)

def get_prot_md5(aa_str):
    """
    Calculate MD5 values for protein sequence
    """
    import hashlib
    assert isinstance(aa_str, str)

    aa_str = aa_str.upper()
    return hashlib.md5(aa_str.encode('utf-8')).hexdigest()

def get_complex_md5(aa_str_list):
    """
    Calculate MD5 value for protein sequences
    """
    import hashlib
    assert isinstance(aa_str_list, list)
    
    aa_str_list = [ seq.upper() for seq in aa_str_list ]
    aa_str_list.sort(key=lambda x: (len(x), x))
    aa_str = "=".join(aa_str_list)
    aa_str = aa_str.upper()
    return hashlib.md5(aa_str.encode('utf-8')).hexdigest()

def get_fasta_md5(fastafile):
    """
    Calculate MD5 value from Fasta file
    """
    seq_dict = General.load_fasta(fastafile)
    md5 = get_complex_md5(list(seq_dict.values()))
    return md5

def get_features_md5(features, get_seq=False):
    aatype = features['aatype']
    if aatype.ndim == 2:
        aatype = aatype.argmax(1)
    assert aatype.ndim == 1
    if 'asym_id' not in features:# or len(set(features['asym_id'])) == 1:
        ## AF2 features.pkl.gz
        seq = "".join([ restypes_with_x[d] for d in aatype ])
        md5 = get_prot_md5(seq)
        if get_seq:
            return md5, seq
        else:
            return md5
    else:
        seq_list = []
        for ch_idx in set(features['asym_id']):
            seq = "".join([ restypes_with_x[d] for d in aatype[features['asym_id'] == ch_idx] ])
            seq_list.append(seq)
        md5 = get_complex_md5(seq_list)
        if get_seq:
            return md5, seq_list
        else:
            return md5

def get_pklgz_md5(pklgzfile):
    """
    Calculate MD5 value from .pkl.gz file
    """
    if pklgzfile.endswith('.gz'):
        ft = pickle.load(gzip.open(pklgzfile, 'rb'))
    else:
        ft = pickle.load(open(pklgzfile, 'rb'))
    return get_features_md5(ft)

def get_pdb_md5(pdbfile):
    """
    Calculate MD5 value from PDB file
    """
    import numpy as np
    
    seq_dict = read_pdb_seq(pdbfile)
    if len(seq_dict) > 0:
        md5 = get_complex_md5(list(seq_dict.values()))
        return md5
    
    prot = protein.from_pdb_string(open(pdbfile).read())
    if set(prot.chain_index) == 1:
        ## AF2 PDB
        seq = "".join([ restypes_with_x[d] for d in prot.aatype ])
        md5 = get_prot_md5(seq)
    else:
        seq_list = []
        for ch_idx in set(prot.chain_index):
            cond = prot.chain_index == ch_idx
            seq = "".join([ restypes_with_x[d] for d in prot.aatype[cond] ])
            res_idx = prot.residue_index[cond]
            if not np.all(res_idx[1:] - res_idx[:-1] == 1):
                raise RuntimeError(f"{pdbfile}: Not compelet sequences: {res_idx.tolist()}")
            seq_list.append(seq)
        md5 = get_complex_md5(seq_list)
    return md5


def pdbid_to_pdb(pdb_id, filename, apply_symmetry=False, only_ca: Union[str, bool]='auto'):
    """
    Parse cif file and as save as PDB file

    only_ca: str or bool
        'auto' -- Automatic discriminate use or not use all atom
        True   -- Force to print Ca atoms only
        False  -- Force to print all atoms
    """
    import pdb_features
    from alphafold.common.protein import PDB_MAX_CHAINS, PDB_CHAIN_IDS
    
    assert isinstance(only_ca, (str, bool))
    if isinstance(only_ca, str):
        assert only_ca == 'auto'

    chain2seq      = {}
    atom_positions = []
    all_atom_masks = []
    aatype         = []
    residue_index  = []
    chain_index    = []
    b_factors      = []
    pdb_3d = pdb_features.get_pdb_3D_info(pdb_id)
    assert len(pdb_3d) <= PDB_MAX_CHAINS, f"Number chain exceed PDB_MAX_CHAINS: {len(pdb_3d)} > {PDB_MAX_CHAINS} use extend_PDB_Chains to extend the chain range"
    
    ch_idx = 0
    for ch_name, p in pdb_3d.items():
        num_res = p['all_atom_positions'].shape[0]
        atom_positions.append( p['all_atom_positions'] )
        aatype.append( p['aatype'].argmax(1) )
        all_atom_masks.append( p['all_atom_masks'] )
        residue_index.append( np.arange(num_res) )
        chain_index.append( np.ones(num_res, dtype=np.int32)*ch_idx )
        b_factors.append( np.zeros([num_res, 37]) )
        chain2seq[ch_idx] = p['sequence'].decode()
        ch_idx += 1
    
    if apply_symmetry:
        import pdb_data
        symmetry = pdb_data.get_cif_symmetry(pdb_id)
        for sym in symmetry[1:]:
            rot = sym.matrix
            trans = sym.vector
            for ch_name, p in pdb_3d.items():
                num_res = p['all_atom_positions'].shape[0]
                all_atom_positions = np.dot(p['all_atom_positions'], rot.T) + trans
                atom_positions.append( all_atom_positions )
                aatype.append( p['aatype'].argmax(1) )
                all_atom_masks.append( p['all_atom_masks'] )
                residue_index.append( np.arange(num_res) )
                chain_index.append( np.ones(num_res, dtype=np.int32)*ch_idx )
                b_factors.append( np.zeros([num_res, 37]) )
                chain2seq[ch_idx] = p['sequence'].decode()
                ch_idx += 1
    
    atom_positions = np.concatenate(atom_positions)
    all_atom_masks = np.concatenate(all_atom_masks)
    aatype         = np.concatenate(aatype)
    residue_index  = np.concatenate(residue_index)
    chain_index    = np.concatenate(chain_index)
    b_factors      = np.concatenate(b_factors)

    if only_ca == 'auto':
        only_ca = True if (all_atom_masks.sum() > 99999) else False

    if all_atom_masks.sum() > 99999 and not only_ca:
        print(_e(f"Expect atom num less than 99999, but got {all_atom_masks.sum()}, suggest use only_ca=True"))
    
    if only_ca:
        all_atom_masks[:, 0] = 0
        all_atom_masks[:, 2:] = 0
        if all_atom_masks.sum() > 99999:
            print(_e(f"TOO MANY RESIDUES: Expect atom num less than 99999, but got {all_atom_masks.sum()}"))
    
    save_as_pdb(aatype,
        residue_index,
        atom_positions,
        all_atom_masks,
        filename,
        b_factors=b_factors,
        asym_id=chain_index,
        full_seq=chain2seq)


def save_dict_as_pdb(features, result, output_dir, output_name, multimer_ri_gap=200):
    protein_final_output_path = os.path.join(
        output_dir, f'{output_name}.pdb'
    )

    protein_bf_output_path = os.path.join(
        output_dir, f'{output_name}_bf.pdb'
    )

    aatype              = features["aatype"]
    residue_index       = features["residue_index"] #+ 1
    atom_positions      = result["final_atom_positions"]
    atom_position_mask  = result["final_atom_mask"]
    if 'all_atom_mask' in features:
        atom_position_mask  = features["all_atom_mask"]
    if atom_positions.ndim ==4:
        atom_positions = atom_positions[-1]
    if 'secondary_structure_logits' in result:
        occupancies     = result["secondary_structure_logits"][-1,0,:,:3].argmax(-1)+1
    else:
        occupancies     = None
    if 'backbone_frame' in result:
        backbone_frame  = result['backbone_frame']
    else:
        backbone_frame  = None
    if len(aatype.shape) ==3:
        has_batch = True
    else:
        has_batch = False
    if 'plddt' in result:
        plddt = result["plddt"]
    if isinstance(aatype, torch.Tensor):
        aatype          = deepcopy(aatype).cpu().numpy()#[:,:,-1]
        residue_index   = deepcopy(residue_index).cpu().numpy()#[:,:,-1]
        # aatype          = deepcopy(aatype).cpu().numpy()[:,:,-1]
        # residue_index   = deepcopy(residue_index).cpu().numpy()[:,:,-1]
    if len(aatype.shape)>=2 and aatype.shape[-1]<aatype.shape[-2]:
        aatype        = aatype[...,-1]
        residue_index = residue_index[...,-1]
    if isinstance(atom_positions, torch.Tensor):
        atom_positions      = deepcopy(atom_positions.detach()).cpu().numpy()
        atom_position_mask  = deepcopy(atom_position_mask.float()).cpu().numpy()
        if 'plddt' in result:
            plddt = deepcopy(plddt.detach()).cpu().numpy()

        if occupancies is not None:    
            occupancies         = deepcopy(occupancies).cpu().numpy()
        if backbone_frame is not None:    
            backbone_frame      = deepcopy(backbone_frame.detach()).cpu().numpy()

        
    if occupancies is not None:    
        if has_batch:
            occupancies = np.tile(occupancies.reshape(1, -1, 1, 1), [1, 1, 37, 1])
        else:
            occupancies = np.tile(occupancies.reshape(-1, 1, 1), [1, 37, 1])

    if 'plddt' in result:
        b_factors = np.repeat(
            plddt[..., None], residue_constants.atom_type_num, axis=-1
        )
    else:
        if occupancies is not None:    
            b_factors = np.zeros_like(occupancies)[...,-1] 
        else:
            b_factors = None

    # import pdb;pdb.set_trace()

    # For multi-chain FASTAs
    chain_index = (residue_index - np.arange(residue_index.shape[0])) / multimer_ri_gap
    chain_index = chain_index.astype(np.int64)
    if len(aatype.shape) == 2:
        aatype = aatype[0]
        atom_positions = atom_positions[0]
        atom_position_mask = atom_position_mask[0]
        residue_index = residue_index[0]
        chain_index = chain_index[0]
        if occupancies is not None:
            occupancies = occupancies[0]
        if b_factors is not None:
            b_factors = b_factors[0]
    # occupancies = np.zeros([atom_positions.shape[0], 37, 1])

    # import pdb;pdb.set_trace()
    

    # import pdb;pdb.set_trace()

    save_as_pdb(
        aatype,
        residue_index,
        atom_positions,
        atom_position_mask,
        out_file=protein_final_output_path,
        b_factors=b_factors,
        asym_id=chain_index,
        occupancies=occupancies,
        # full_seq={ PDB_CHAIN_IDS.index(k):v for k,v in name2seq.items() },
        ss_coding=[3,1,2],
        write_ss=occupancies is not None,
        gap_ter_threshold=9.0
    )


    # save_type = "backbone_frame"
    save_type = ""
    if save_type == "backbone_frame":
        # import pdb;pdb.set_trace()
        atom_positions[:,1,:] = backbone_frame[-1,:,4:]
        atom_position_mask = np.zeros_like(atom_position_mask)
        atom_position_mask[:,1]=1
        save_as_pdb(
            aatype,
            residue_index,
            atom_positions,
            atom_position_mask,
            out_file=protein_bf_output_path,
            b_factors=b_factors,
            asym_id=chain_index,
            occupancies=occupancies,
            # full_seq={ PDB_CHAIN_IDS.index(k):v for k,v in name2seq.items() },
            ss_coding=[3,1,2],
            write_ss=True,
            gap_ter_threshold=9.0
        )
    



    # with open(protein_final_output_path, 'w') as fp:
    #     fp.write(protein.to_pdb(protein_final))

    # with open(protein_bf_output_path, 'w') as fp:
    #     fp.write(protein.to_pdb(protein_bf))

    # logger.info(f"Output written to {protein_final_output_path}...")
