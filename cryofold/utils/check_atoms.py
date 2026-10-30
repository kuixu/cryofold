#!/usr/bin/env python3
import re, os, sys, collections, Bio.SeqIO, subprocess, numpy as np
three2one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
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
res_types_set=set(three2one.keys())
restype_name_to_atom14_names = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
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
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
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
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
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
dir="/data/share/batch"
# dict.list: k=date, value=pdb id
# dict_path="/data/share/1211rf/dict.list"
# emdb_date_dict=dict()
# with open(dict_path,'r') as rf:
#     lines=rf.readlines()
#     for l in lines:
#         l=l.strip().split()
#         d,e=l[0],l[1]
#         emdb_date_dict[e]=d

if __name__ != '__main__': sys.exit()
# parser = argparse.ArgumentParser(description="extract CDS to .fa file")
# parser.add_argument('-i', required=True, help = 'path to input .pdb file')
# parser.add_argument('-o', help = 'path to output .pdb file')

# emdb2CC = {'7607':0.4, '8800':0.8}  # special CC threshold for a few maps
# args = parser.parse_args()
in_path=sys.argv[1]
ou_path=sys.argv[2]
emdb,date,in_path=str(),str(),str()
# if in_path.endswith(".pdb"):
#     # in_path=args.i
#     emdb = in_path.split('/')[-1].split('.')[0]
# else:
#     emdb=in_path
#     # date=emdb_date_dict.get(emdb)
#     in_path=os.path.join(dir,date,emdb+".pdb")
# date=emdb_date_dict.get(emdb)

if ou_path is None: 
    ou_path = os.path.join(os.getcwd(),emdb+"-rm.pdb")

# read pdb
# CaLD = collections.defaultdict(list)
# for line in open(args.i).read().rstrip('\n').split('\n'): 
#     if not line.startswith('ATOM') or line[12:16] != ' CA ': continue
#     cname = line[20:22].strip()
#     CaLD[cname].append(line)
#     assert int(line[ 22:26 ]) == len(CaLD[cname]), f"{args.i} {cname} residue numbering error"
# # read CC per residue
# ccvD = {k:np.zeros(len(v)) for k,v in CaLD.items() }
# for l in open(args.c).read().rstrip('\n').split('\n'):
#     cname, _, pos, cc = re.split('\s+', l)[:4]
#     ccvD[cname][int(pos)-1] = float(cc)
# read fasta
# seq_headL, cname2seq = [(str(x.seq),x.id) for x in Bio.SeqIO.parse(args.s, 'fasta')], {}

# 当序列一样时，假设seq1与chainA删两段，seq2与chainB删一段，那么seq1匹配chainA,B分别为(0.2+0.2)/2和(0.2+0.8)/2，但seq2匹配chainA,B都是0.2
# for seq, head in sorted(seq_headL, key = lambda x:x[0].count('-') , reverse = True): 
#     #from IPython import embed; embed()
#     cname_mccL = []
#     for cname, CaL in CaLD.items(): 
#         if len(seq) != len(CaL) or cname in cname2seq: continue
#         modelling = ''.join(three2one[l[17:20]] for c,l in zip(seq, CaL) if c!='-')
#         if modelling != ''.join(c for c in seq if c!='-'): continue
#         mcc = np.mean([x for c,x in zip(seq, ccvD[cname]) if c=='-']) if '-' in seq else 1
#         cname_mccL.append([cname, mcc])
#     if cname_mccL == []: continue
#     assert len(cname_mccL) > 0, f"{args.s} {head} can not find matched chain"
#     cname2seq[ sorted(cname_mccL, key = lambda x:x[1])[0][0] ] = seq

outL, delS = [], set()
# seq = seq[:10] + '-'*20 + seq[30:]
# for cname, seq in cname2seq.items(): 
#     for region in re.finditer('-+', seq): 
#         i, j = region.start(0), region.end(0)
#         if np.mean(ccvD[cname][i:j]) >= args.d: continue
#         for l in CaLD[cname][i:j]: delS.add( l[20:26] )

for l in open(in_path).readlines():
    if not l.startswith("ATOM"): continue
    cur_s=l[20:26]
    l=l.strip().split()
    atom,residue=l[2],l[3]
    # print(atom,residue)
    if atom not in atom_types_set or residue not in res_types_set  \
        or atom not in restype_name_to_atom14_names.get(residue):
        delS.add(cur_s)

print(delS)

for l in open(in_path).readlines(): 
    if not l.startswith('ATOM'): 
        continue
    if l[20:26] not in delS: 
        outL.append(l)
    # if l[20:26] in delS and outL!=[] and outL[-1]!='TER\n': 
        # outL.append('TER\n')
open(args.o, 'w').write(''.join(outL))
