#!/usr/bin/env python3
import os, sys, io, argparse, collections, numpy as np, scipy
from cryonet.libs import PDB
# Bio.PDB
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# import metrics


# read PDB coordinates
def parse_pdb(file): #各种软件都不符合BioPDB，mm把cName重复使用，ma用两个字符表示cName
    # lineL, model, parser = [], Bio.PDB.Model.Model(0), Bio.PDB.PDBParser(QUIET = True)
    lineL, model, parser = [], PDB.Model.Model(0), PDB.PDBParser(QUIET = True)
    processed = False
    for line in open(file).read().rstrip('\n').split('\n'):
        if line.startswith('HETATM'): 
            lineL.append(line)
        if line[:3] in ('TER', 'END') and len(lineL)>0: 
            struct = parser.get_structure('', io.StringIO('\n'.join(lineL)))
            chain = struct.child_list[0].child_list[0]
            chain.id, lineL = lineL[0][20:22].strip(), []
            model.add(chain)
            processed = True
    if not processed:
        struct = parser.get_structure('', io.StringIO('\n'.join(lineL)))
        chain = struct.child_list[0].child_list[0]
        chain.id, lineL = lineL[0][20:22].strip(), []
        model.add(chain)
    return model

def check_res(x): 
    return x.id[0]==' ' and x.id[2]==' ' and 'C1' in x 
def check_res_gt(x): 
    return x.id[0]==' ' and x.id[2]==' ' and 'C1' in x and x.resname!='UNK'  # ignore HETATM & insertion code


# cacluate various matrics
def get_dist(x): 
    if len(x)==0: 
        return None
    return np.linalg.norm(x[:, :3] - x[:, 3:], ord = 2, axis = 1)
def getRMSD(coord12A):
    if len(coord12A)==0: 
        return -1
    else:
        return np.mean(get_dist(coord12A)**2)**0.5
def getTM(coord12A, d0): 
    if len(coord12A)==0: 
        return 0
    else:
        return np.sum(1/(1 + get_dist(coord12A)**2/d0**2))
def get_LDDT(coord12A, missA): 
    if len(coord12A)==0: 
        return 0
   
    true_coord, N = coord12A[:, 3:] if len(missA) ==0 else np.vstack([coord12A[:, 3:], missA]), len(coord12A)+len(missA)
    n, pred_map = len(coord12A), np.full((N, N), 15+1)  # LDDT fill value
    true_map = scipy.spatial.distance.cdist(true_coord, true_coord)
    pred_map[:n, :n] = scipy.spatial.distance.cdist(coord12A[:, :3], coord12A[:, :3])
    return metrics.calculate_LDDT(true_map, pred_map, R = 15)
def align(coord12A): # center and kabsch rotate
    x = coord12A - coord12A.mean(axis = 0)
    rot = scipy.spatial.transform.Rotation.align_vectors(x[:, :3], x[:, 3:])[0]
    return np.hstack([ x[:, :3], rot.apply(x[:, 3:]) ])

def main(inpath, truepath, return_results=False):

    # if args.wrap_BioPDB:
    model1 = parse_pdb(inpath)
    # import pdb;pdb.set_trace()

    model2 = parse_pdb(truepath)
    # else: 
    # model1 = PDB.PDBParser(QUIET = True).get_structure('', inpath).child_list[0]
    # model2 = PDB.PDBParser(QUIET = True).get_structure('', truepath).child_list[0]

    

    # resL1 = [res for res in model1.get_residues() if check_res(res)]
    # resL2 = [res for res in model2.get_residues() if check_res_gt(res)]

    resL1 = [res for res in model1.get_residues() ]
    resL2 = [res for res in model2.get_residues() ]

    raw_coord1 = np.array([res['C1'].get_coord() for res in resL1])
    raw_coord2 = np.array([res['C1'].get_coord() for res in resL2])
    # import pdb;pdb.set_trace()
    # print(raw_coord1)
    # print(raw_coord2)
    distM = scipy.spatial.distance.cdist(raw_coord1, raw_coord2)
    iA, jA = distM.argmin(axis = 0), distM.argmin(axis = 1)
    iA[ (distM>3).all(axis = 0) ] = -1 # 只考虑3A以内的
    closeD = collections.defaultdict(list)  # 一个predRes可能被多个trueRes视为最近，先存进list、再取最近的trueRes
    for j, i in enumerate(iA): closeD[i].append(j)
    bestD = { i:jL[ distM[i][jL].argmin() ] for i, jL in closeD.items() if i!=-1 }
    pairL = [ (resL1[i], resL2[j]) for i,j in bestD.items() ]

    # accumlate coordinates of matched residues & atoms
    Res12L, Atom12L, res12L, atom12L, Miss2L, miss2L = [], [], [], [], [0]*len(resL2), [0]*len(resL2)
    for i, res in enumerate(resL2): Miss2L[i] = miss2L[i] = res['C1'].get_coord().tolist()
    for res1, res2 in pairL: 
        if True:                       Miss2L.remove(res2['C1'].get_coord().tolist())
        if res1.resname==res2.resname: miss2L.remove(res2['C1'].get_coord().tolist())
        for k in res2.child_dict.keys(): 
            if k not in res1.child_dict: continue
            coord1 = res1.child_dict[k].get_coord()
            coord2 = res2.child_dict[k].get_coord()
            coord12 = np.hstack([ coord1, coord2 ])
            # print(res1.resname, res2.resname)
            if                                k=='C1':  Res12L.append(coord12)
            if                                True:    Atom12L.append(coord12)
            if res1.resname==res2.resname and k=='C1':  res12L.append(coord12)
            if res1.resname==res2.resname and True:    atom12L.append(coord12)
    Res12A, Atom12A, res12A, atom12A, Miss2A, miss2A = [np.array(x) for x in (Res12L, Atom12L, res12L, atom12L, Miss2L, miss2L)]
    assert len(Res12A)+len(Miss2A)==len(resL2) and len(res12A)+len(miss2A)==len(resL2), "some CA in true model not seen in alignment"

    # generate output
    # N1, N2, USd0 = len(resL1), len(resL2), max(0.5, (len(resL2)-15)**(1/3)*1.24-1.8)
    N1, N2 = len(resL1), len(resL2)
    # resultL = [USd0, N1, N2, len(Res12A), len(res12A)]
    resultL = [N1, N2, len(Res12A), len(res12A)]
    found = len(Res12A) / N2
    
    if return_results:
        return {"found": found}
    else:
        print('\t'.join(str(round(x, 4)) for x in resultL))


if __name__ == '__main__': 
    # sys.exit()
    parser = argparse.ArgumentParser(description = "Cacluate RMSD between two .pdb based on USalign specified correspondence")
    parser.add_argument('-i', '--input', required = True, help = "path to input .pdb")
    parser.add_argument('-t', '--true',  required = True, help = "path to ground truth .pdb")
    parser.add_argument('-o', '--output', help = "path to residue RMSD .txt, default to input filename + .txt")
    parser.add_argument('--wrap-BioPDB', action = 'store_true', help = 'use BioPDB to parse each chain instead of entire file, allowing two-letter chain name, but只识别ATOM、无法处理多model')
    args = parser.parse_args()
    main(args.input, args.true )

# print("Res12A:", Res12A.shape) 
# print("Atom12A:", Atom12A.shape) 
# print("res12A:", res12A.shape) 
# print("atom12A:", atom12A.shape) 

# resultL += [getRMSD(x) for x in (Res12A, Atom12A, res12A, atom12A)]
# resultL += [getTM(x, USd0)/N2 for x in (Res12A, res12A)]
# resultL += [get_LDDT(Res12A, Miss2A), get_LDDT(res12A, miss2A)]
# GDT与USalign不一致，不知道USalign用的是什么matrix
# Ca_dist_bb, Ca_dist_seq = get_dist(Res12A), get_dist(res12A)

# def get_GDT(dist, is_high_accuracy = False):
#     if dist is None: 
#         return 0
#     if is_high_accuracy: 
#         Ps = (0.5, 1, 2, 4)
#     else:
#         Ps = (1, 2, 4, 8)
#     return np.array([dist<=P for P in Ps]).mean()


# resultL.append( get_GDT(Ca_dist_bb , False) ) # GDT_TS_bb
# resultL.append( get_GDT(Ca_dist_bb , True)  ) # GDT_HA_bb
# resultL.append( get_GDT(Ca_dist_seq, False) ) # GDT_TS_seq
# resultL.append( get_GDT(Ca_dist_seq, True)  ) # GDT_HA_seq

# resultL.append( np.array([Ca_dist_bb <=P for P in (  1, 2, 4, 8)]).mean() ) # GDT_TS_bb
# resultL.append( np.array([Ca_dist_bb <=P for P in (0.5, 1, 2, 4)]).mean() ) # GDT_HA_bb
# resultL.append( np.array([Ca_dist_seq<=P for P in (  1, 2, 4, 8)]).mean() ) # GDT_TS_seq
# resultL.append( np.array([Ca_dist_seq<=P for P in (0.5, 1, 2, 4)]).mean() ) # GDT_HA_seq
# def get_GDT(coord12A, Ps): return np.array([get_dist(coord12A)<=P for P in Ps]).mean()
# resultL += [get_GDT(align(x), Ps) for Ps in [(1, 2, 4, 8), (0.5, 1, 2, 4)] for x in (Res12A, res12A)]
# print('\t'.join(str(round(x, 4)) for x in resultL))

    
# def add_rmsd(pdb_path, distD, out_path=""):
#     cc_dict = {}
#     dist_max = 0
#     for res in model1.get_residues(): 
#         coord = None if 'CA' not in res else tuple(res['CA'].get_coord())
#         if coord and coord in distD:
#             dist = distD[coord] 
#             dist_max = max(dist_max, dist)
#         else:
#             dist = -1
#         outL.append(f"{res.parent.id} {res.resname} {res.id[1]} {dist} 0 0\n")

#         k = f"{res.parent.id}_{res.resname}_{res.id[1]}".strip()
#         cc_dict[k] = "{:.2f}".format(dist).rjust(6)
#     # with open(cc_path,'r') as f:
#     #     for line in f.readlines():
#     #         fields = line.strip().split()
#     #         k = f"{fields[0]}_{fields[1]}_{fields[2]}"
#     #         cc_dict[k] = "{:.2f}".format(float(fields[3])*100).rjust(5)
#     # print(dist_max)
#     cc_pdb_lines = ""
#     with open(pdb_path,'r') as f:
#         for line in f.readlines():
            
#             if line[0:4]=="ATOM" :
#                 k = f"{line[20:22].strip()}_{line[17:20]}_{line[22:26].strip()}".strip()
#                 # import pdb;pdb.set_trace()
#                 if k in cc_dict:
#                     # if float(cc_dict[k].strip())==-1.0:
#                     #     rmsd_str = "{:.2f}".format(dist_max).rjust(6)
#                     # else:
#                     rmsd_str = cc_dict[k]
#                     new_line = f"{line[0:60]}{rmsd_str}{line[66:]}"
#                     cc_pdb_lines += new_line
#                 else:
#                     cc_pdb_lines+=line
#             else:
#                 cc_pdb_lines+=line
#     with open(out_path,'w') as f:
#         f.write(cc_pdb_lines)
#     # print('========== add_cc Done! =================')

# # output residue dist
# outL, distD = [], dict(zip( [tuple(x[:3]) for x in res12A], get_dist(res12A) ))
# # outL, distD = [], dict(zip( [tuple(x[:3]) for x in Res12A], get_dist(Res12A) ))
# # for res in model1.get_residues(): 
# #     coord = None if 'CA' not in res else tuple(res['CA'].get_coord())
# #     dist = distD[coord] if coord and coord in distD else -1
# #     dist_str = "{:.2f}".format(dist).rjust(5)
# #     outL.append(f"{res.parent.id} {res.resname} {str(res.id[1]).rjust(4)} {dist_str} 0 0\n")
# # open(args.input.replace('.pdb', '_rmsd.txt'), 'w').write(''.join(outL))

# add_rmsd(args.input, distD, args.output)