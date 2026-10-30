import os,sys
from typing import List
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_types_set=set(atom_types)
restype_name_to_atom14_names = {
    "ALA": ["N", "CA", "C", "O", "CB",],
    "ARG": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
    ],
    "ASN": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OD1",
        "ND2",
    ],
    "ASP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OD1",
        "OD2",
    ],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "NE2",
    ],
    "GLU": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "OE2",
    ],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "ND1",
        "CD2",
        "CE1",
        "NE2",
    ],
    "ILE": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG1",
        "CG2",
        "CD1",
    ],
    "LEU": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
    ],
    "LYS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "CE",
        "NZ",
    ],
    "MET": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "SD",
        "CE",
    ],
    "PHE": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
    ],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD",],
    "SER": ["N", "CA", "C", "O", "CB", "OG", ],
    "THR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "OG1",
        "CG2",
    ],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
    ],
    "VAL": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG1",
        "CG2",
    ],
    "UNK": atom_types,
}
res_types_set=set(restype_name_to_atom14_names.keys())
res_atom=dict()
for k,v in restype_name_to_atom14_names.items():
    res_atom[k]=set(v)

def get_atom(s:str):
    if len(s)<17: return ""
    if len(s)>=78 and s[77]=='H': return ""
    return s[13:16].strip()

def get_res(s:str):
    if len(s)<21: return ""
    return s[17:20]

def get_resid(s:str):
    if len(s)<28: return ""
    return s[21:27].strip()

def check_file(file:str):
    with open(file,'r') as rf:
        lines=rf.readlines()
    lines=[x.strip() for x in lines if x.startswith("ATOM")]
    cur_part=list()
    cur_res=""
    cur_resid=""
    for i,l in enumerate(lines):
        if get_res(l)==cur_res and get_resid(l)==cur_resid:
            cur_part.append(l)
        elif len(cur_part)==0:
            cur_part.append(l)
            cur_res=get_res(l)
            cur_resid=get_resid(l)
        else:
            flag=check_atom(cur_part)
            if not flag:
                # err_out=[x[13:31].strip() for x in cur_part]
                print(cur_part[0][17:31].strip()+"\n")
                # for e in err_out: print(e)
            cur_part=[l]
            cur_res=get_res(l)
            cur_resid=get_resid(l)


def check_atom(lines:List[str]):
    cur_res=get_res(lines[0])
    if cur_res not in res_types_set: return False
    cur_atoms=list()
    for l in lines:
        atom_name = get_atom(l)
        if atom_name!="":
            cur_atoms.append(atom_name)
    if len(cur_atoms)!=len(res_atom[cur_res]) or set(cur_atoms)!=res_atom[cur_res]:
        # false atoms in files
        display(cur_atoms)
        # standard atoms for the current residue
        display(res_atom[cur_res])
        # breakpoint()
        return False
    return True
    
def display(xs:Iterable):
    print(",".join(x for x in xs))

if __name__=='__main__':
    file_path=sys.argv[1]
    file_name=os.path.basename(file_path)
    print(f"check {file_name} ...")
    check_file(file_path)