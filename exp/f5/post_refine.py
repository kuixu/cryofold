import os, sys
import logging

# from pyrosetta import logging_support
import time
import subprocess

# import argparse
from pathlib import Path
import numpy as np
import re
import socket
import json

# Geometry goal


def rama_outliers_goal(rama_out: float, max=1.0, min=0.0):
    return float(rama_out) >= min and rama_out <= max


def rama_favored_goal(rama_fav: float, max=100.0, min=95.0):
    return float(rama_fav) >= min and rama_fav <= max


def rotamer_outliers_goal(rot_out: float, max=1.0, min=0.0):
    return float(rot_out) >= min and rot_out <= max


def rotamer_favored_goal(rot_fav: float, max=100.0, min=95.0):
    return float(rot_fav) >= min and rot_fav <= max


def cbeta_deviations_goal(cbeta_dev: float, max=1.0, min=0.0):
    return float(cbeta_dev) >= min and cbeta_dev <= max


def Bond_goal(rms_bond: float, max=0.02, min=0.0):
    return float(rms_bond) <= max and rms_bond >= min


def Angle_goal(rms_angle: float, max=2.0, min=0.0):
    return float(rms_angle) >= min and rms_angle <= max


def rama_z_goal(rama_z: float, max=2.0, min=-2.0):
    return float(rama_z) >= min and rama_z <= max


metrics = [
    "rama_outliers",
    "rama_favored",
    "rotamer_outliers",
    "rotamer_favored",
    "cbeta_deviations",
    "Bond",
    "Angle",
    "rama_z",
]
cc_metrics = ["CC_mask", "CC_box", "CC_mc", "CC_sc"]

goal_funcs = [m + "_goal" for m in metrics]

# M_SUF=".mrc"
# P_SUF="_cf1.pdb"
RESOLU = 5.0  # default resolution
BFACTOR_DEFAULT = "100.00 "  # default b-factor

resolu_idx = "/share01/hpc/zhangxiaoxing/data/emdb_list_info.csv"
phenix_env = "/share01/hpc/zhangxiaoxing/usr/phenix-1.21.1-5286/phenix_env.sh"
thread_num = 10
hostname = socket.gethostname()
# print(hostname)
# if hostname == "smtl01":
if hostname == "smtl01" or hostname == "hpc.smart.org.cn":
    thread_num = 40
    resolu_idx = "/share01/hpc/zhangxiaoxing/data/emdb_list_info.csv"
    phenix_env = "/share01/hpc/zhangxiaoxing/usr/phenix-1.21.1-5286/phenix_env.sh"
elif hostname == "worker201" or hostname == "worker202":
    resolu_idx = "/data/share/emdb_list_info.csv"
    phenix_env = "/home/mccxadmin/phenix-1.21.1-5286/phenix_env.sh"
    thread_num = 30
elif hostname == "localhost":
    resolu_idx = "/home/jsptgpu/data/db/emdb/emdb_list_info.csv"
    phenix_env = "/opt/phenix-1.21.1-5286/phenix_env.sh"
    thread_num = 30
elif hostname=='cryonet' : # s22
    resolu_idx="/home/jsptgpu/data/db/emdb/emdb_list_info.csv"
    phenix_env="/home/jsptgpu/usr/phenix-1.21.1-5286/phenix_env.sh"
    thread_num = 30
elif hostname=="bnode" or hostname.startswith("node5"):
    resolu_idx="/data1/casual/yvxiaozhu/emdb_list_info.csv"
    phenix_env="/home/casual/phenix-1.21.1-5286/phenix_env.sh"
    thread_num=30


clash_pattern = r"Angstrom:\n(.*?)clash"


logger = logging.getLogger(__file__)
consoleHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s",
    datefmt="%m-%d %H:%M:%S",
)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
# logger.setLevel(logging.ERROR)
logger.setLevel(logging.INFO)

# create an emdb-resolution dictionary

resolu_dict = dict()
with open(resolu_idx, "r") as rf:
    lines = rf.readlines()
for l in lines:
    l = l.strip().split()
    if len(l) < 4:
        continue
    if l[3] == "resolution":
        continue
    try:
        value = float(l[3])
        resolu_dict[l[1]] = value
        resolu_dict[l[1]] = RESOLU
        continue
    except:
        pass

# phenix.real_space_refine parameters
phenix_params = {
    "nproc": "12",
    "refinement.macro_cycles": 2,
    "ncs_constraints": "False",
    "secondary_structure.enabled": "True",
    "remove_outliers": "False",
    # "run":"minimization_global+simulated_annealing+local_grid_search+adp+occupancy+nqh_flips+morphing+rigid_body",
    # "run":"minimization_global+simulated_annealing+local_grid_search+adp+occupancy+nqh_flips+morphing",
    # "simulated_annealing":"every_macro_cycle",
    "run": "minimization_global+local_grid_search+adp+occupancy+nqh_flips+morphing",
    "morphing": "once",
    "weight": None,
    "max_reasonable_bond_distance": "150",
    "output.suffix": "_rsr",
    "model_format": "pdb",
    "rotamers.restraints.sigma": 1,
    "scattering_table": "electron",
    "pdb_interpretation.secondary_structure.protein.remove_outliers": "True",
    "write_initial_geo_file": "False",
    "skip_map_model_overlap_check": "True",
    "nonbonded_weight": 200,
    "pdb_interpretation.reference_coordinate_restraints.enabled":"True",
    "pdb_interpretation.reference_coordinate_restraints.selection":'''"(name ca or name c or name n or name o)"''',
    "pdb_interpretation.reference_coordinate_restraints.sigma":"0.05",
    "pdb_interpretation.restraints_library.omega_cdl":"True",
    "pdb_interpretation.restraints_library.cdl_svl":"True",
    "pdb_interpretation.restraints_library.rdl":"True",
    "pdb_interpretation.restraints_library.hpdl":"True",
    

    # "reference_model.enabled":"True",
    # "reference_model.file":""
}

phenix_params2 = {
    "nproc": "12",
    "refinement.macro_cycles": 2,
    "ncs_constraints": "False",
    "secondary_structure.enabled": "True",
    "remove_outliers": "False",
    "run": "minimization_global+local_grid_search+adp+occupancy+nqh_flips+morphing",
    "morphing": "once",
    "max_reasonable_bond_distance": "150",
    "output.suffix": "_rf",
    "model_format": "pdb",
    "scattering_table": "electron",
    "pdb_interpretation.secondary_structure.protein.remove_outliers": "True",
    "write_initial_geo_file": "False",
    "skip_map_model_overlap_check": "True",
}

# eff_params=[
#     "nproc=12",
#     "output.suffix=_eff ",
#     "ncs_constraints=False ",
#     "model_format=pdb",
#     "scattering_table=electron",
#     "refinement.macro_cycles=2",
#     "pdb_interpretation.secondary_structure.protein.remove_outliers=True",
#     "refinement{run=minimization_global+adp+occupancy+nqh_flips}",
#     "write_initial_geo_file=False ",
#     "skip_map_model_overlap_check=True ",
#     "weight=None ",
#     "nonbonded_weight=200 ",
#     "pdb_interpretation{max_reasonable_bond_distance=150} ",
# ]

eff_params4 = {
    "nproc": "12",
    "refinement.macro_cycles": 5,
    "ncs_constraints": "False",
    # "secondary_structure.enabled":"True",
    # "remove_outliers":"False",
    # "run":"minimization_global+simulated_annealing+local_grid_search+adp+occupancy+nqh_flips+morphing+rigid_body",
    "refinement{run": "minimization_global+local_grid_search+adp+occupancy+nqh_flips}",
    # "simulated_annealing":"every_macro_cycle",
    # "morphing":"once",
    # "weight": None,
    "pdb_interpretation{max_reasonable_bond_distance": "150}",
    "output.suffix": "_eff",
    "model_format": "pdb",
    # "rotamers.restraints.sigma":1,
    "scattering_table": "electron",
    "pdb_interpretation.secondary_structure.protein.remove_outliers": "True",
    "write_initial_geo_file": "False",
    "skip_map_model_overlap_check": "True",
    # "nonbonded_weight":200,
}


eff_params = {
    "nproc": "12",
    "refinement.macro_cycles": 5,
    "ncs_constraints": "False",
    # "secondary_structure.enabled":"True",
    # "remove_outliers":"False",
    # "run":"minimization_global+simulated_annealing+local_grid_search+adp+occupancy+nqh_flips+morphing+rigid_body",
    "refinement{run": "minimization_global+adp+occupancy+nqh_flips}",
    # "simulated_annealing":"every_macro_cycle",
    # "morphing":"once",
    # "weight": None,
    "pdb_interpretation{max_reasonable_bond_distance": "150",
    "restraints_library{omega_cdl":"True",
    "cdl_svl":"True",
    "rdl":"True",
    "hpdl":"True}",
    "reference_coordinate_restraints{enabled":"True",
    "selection":'''"(name ca or name c or name n or name o)"''',
    "sigma":"0.05}}",
    "output.suffix": "_eff",
    "model_format": "pdb",
    # "rotamers.restraints.sigma":1,
    "scattering_table": "electron",
    "pdb_interpretation.secondary_structure.protein.remove_outliers": "True",
    "write_initial_geo_file": "False",
    "skip_map_model_overlap_check": "True",
    # "nonbonded_weight":200,
}


geo_params = [
    "ramachandran_plot_restraints.enabled=True",
    "rmsd_bonds_termination_cutoff=0.01",
    "rmsd_angles_termination_cutoff=1.0",
    # "strict_rotamer_matching=True",
    # "fix_rotamer_outliers=False",
    "allow_allowed_rotamers=True",
    "write_geo_file=False",
    "restraints_library.omega_cdl=True",
    "restraints_library.cdl_svl=True",
    "restraints_library.rdl=True",
    "restraints_library.hpdl=True",
    "reference_coordinate_restraints.enabled=True",
    '''reference_coordinate_restraints.selection="(name ca and name c and name o and name n)"''',
    "reference_coordinate_restraints.sigma=0.05",

]



# pyrosetta fast_relax parameters
pyr_init_cmd = [
    "-mute all",
    "-relax:default_repeats 2",
    "-default_max_cycles 200",
    "-relax:constrain_relax_to_start_coords",
    "-relax:ramp_constraints false",
    "-relax:dualspace true",
    "-relax:minimize_bond_angles",
    "-relax:jump_move true",
    "-relax:bb_move true",
    "-relax:chi_move true",
]


def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text


def get_weight(resolu: float):
    nb_weight = 100
    weight = 20
    bins = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    nb_weights = [100, 200, 250, 300, 350, 400, 500]
    # nb_weights=np.array([100,200,250,300,350,400,500])*2
    weights = [20, 10, 10, 5, 5, 3, 1]
    # weights=np.array([20,10,10,5,5,3,1])*2
    if resolu >= bins[-1]:
        return weights[-1], nb_weights[-1]
    if resolu <= bins[0]:
        return weights[0], nb_weights[0]
    for idx, b in enumerate(bins):
        if resolu >= b:
            weight = weights[idx]
            nb_weight = nb_weights[idx]

    return weight, nb_weight


def get_constraints(cc_filename, clash_filename, bond_filename="", cc_threshold=0.7):
    with open(cc_filename, mode="r") as f:
        data = f.readlines()
    with open(clash_filename, mode="r") as f1:
        data1 = f1.read()
        match = re.search(clash_pattern, data1, re.DOTALL)
        data1 = match.group(1).strip("\n").split("\n") if match else []
    data2 = list()
    if len(bond_filename) > 0:
        with open(bond_filename, "r") as f2:
            lines = f2.readlines()
        for i, l in enumerate(lines):
            if (
                "----------Bond lengths----------" in l
                or "----------Bond angles----------" in l
            ):
                j = i + 3
                while len(lines[j]) > 5:
                    data2.append(lines[j])
                    j += 1
    return generate_constraints(data, data1, data2, cc_threshold)


def convert_to_ranges(arr):
    if not arr:
        return []
    ranges = []
    start = arr[0]
    end = arr[0]
    for i in range(1, len(arr)):
        if arr[i] == end + 1:  # 检查是否连续
            end = arr[i]
        else:
            # 当前范围结束，存储范围
            template = f"resseq {start}:{end}" if start != end else f"resid {start}"
            ranges.append(template)
            # 更新新的范围起点
            start = arr[i]
            end = arr[i]
    # 最后一个范围
    template = f"resseq {start}:{end}" if start != end else f"resid {start}"
    ranges.append(template)
    return ranges


def generate_constraints(data, data1, data2, cc_threshold=0.7):
    final_res = []
    full_res = {}
    res = {}
    complement_res = {}

    # deal with lines in cc_per_residue.log
    for i, d in enumerate(data):
        # print(d)
        chain_id = d[:2].strip()
        # residue = d[1]
        residue_id = int(d[6:11].strip())
        cc = float(d[12:19].strip())

        l = full_res.get(chain_id, [])
        if residue_id not in l:
            l.append(residue_id)
            full_res[chain_id] = l
        if cc < cc_threshold:
            # print(chain_id,residue_id,cc)
            l = res.get(chain_id, [])
            if residue_id not in l:
                l.append(residue_id)
                res[chain_id] = l

    # deal with lines in molprobity.out
    for d in data2:
        chain_id = d[:4].strip()
        residue_id = int(d[4:8].strip())
        if chain_id not in res.keys():
            res[chain_id] = [residue_id]
        elif residue_id not in res[chain_id]:
            res[chain_id].append(residue_id)

    # deal with lines in clashscore logs
    for d in data1:
        chain_id1 = d[:2].strip()
        residue_id1 = int(d[2:8].strip())
        if chain_id1 not in res:
            res[chain_id1] = [residue_id1]
        elif residue_id1 not in res[chain_id1]:
            res[chain_id1].append(residue_id1)
        # print(d)
        chain_id2 = d[17:19].strip()
        residue_id2 = int(d[19:25].strip())
        if chain_id2 not in res:
            res[chain_id2] = [residue_id2]
        elif residue_id2 not in res[chain_id2]:
            res[chain_id2].append(residue_id2)

    for chain_id in full_res.keys():
        residues = full_res[chain_id]
        if chain_id not in res.keys():
            complement_res[chain_id] = residues
            continue
        l = complement_res.get(chain_id, [])
        for r in residues:
            if r not in res[chain_id] and r not in l:
                l.append(r)
        complement_res[chain_id] = l

    # convert to ranges
    for k in complement_res:
        ranges = convert_to_ranges(complement_res[k])
        if len(ranges) == 0:
            continue
        t = "chain " + k + " and " "(" + " or ".join(ranges) + ")"
        final_res.append(t)

    if len(final_res) == 0:
        return ""
    elif len(final_res) == 1:
        return final_res[0]
    else:
        return " or ".join(["(" + r + ")" for r in final_res])


def parse_vc(ccfile, per_chain_cc=False):

    CC_mask, CC_volume, CC_peaks, CC_box = None, None, None, None
    CC_mc, CC_sc = None, None
    clashscore = None
    rama_outliers, rama_allowed, rama_favored = None, None, None

    is_rama_outliers = False
    is_rotamer_outliers = False

    rotamer_outliers, rotamer_allowed, rotamer_favored = None, None, None
    cbeta_deviations = None
    chain_cc = {}
    chain_cc_region = False
    Bond, Angle, Chirality, Planarity, Dihedral = None, None, None, None, None
    # if per_chain_cc :
    #     Per chain:
    rama_z = None
    with open(ccfile, "r") as f:
        output = f.readlines()
        # output = output.split('\n')
        CIS_PROLINE = 0
        CIS_GENERAL = 0
        TWISTED_PROLINE = 0
        TWISTED_GENERAL = 0
        for idx in range(len(output)):
            if output[idx].startswith("    All-atom Clashscore"):
                clashscore = float(output[idx].split(":")[1].strip())
            elif output[idx].startswith("    Ramachandran Plot:"):
                is_rama_outliers = True
                is_rotamer_outliers = False
            elif output[idx].startswith("    Rotamer:"):
                is_rama_outliers = False
                is_rotamer_outliers = True
            # elif output[idx].startswith('  ROTAMER OUTLIERS :'):
            # rotamer_outliers = float(output[idx].split(':')[1].strip())
            # rotamer_outliers = float(output[idx].split(':')[1].strip().split()[0].strip())

            # rotamer_allowed = round(random.uniform(0, 10),2)
            # rotamer_favored = 100 - rotamer_allowed

            elif is_rama_outliers and output[idx].startswith("      Outliers :"):
                rama_outliers = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif is_rama_outliers and output[idx].startswith("      Allowed  :"):
                rama_allowed = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif is_rama_outliers and output[idx].startswith("      Favored  :"):
                rama_favored = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )

            elif is_rotamer_outliers and output[idx].startswith("      Outliers :"):
                rotamer_outliers = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif is_rotamer_outliers and output[idx].startswith("      Allowed  :"):
                rotamer_allowed = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif is_rotamer_outliers and output[idx].startswith("      Favored  :"):
                rotamer_favored = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )

            elif output[idx].startswith("    Bond      :"):
                Bond = float(output[idx].split(":")[1].strip().split()[0].strip())
            elif output[idx].startswith("    Angle     :"):
                Angle = float(output[idx].split(":")[1].strip().split()[0].strip())
            elif output[idx].startswith("    Chirality :"):
                Chirality = float(output[idx].split(":")[1].strip().split()[0].strip())
            elif output[idx].startswith("    Planarity :"):
                Planarity = float(output[idx].split(":")[1].strip().split()[0].strip())
            elif output[idx].startswith("    Dihedral  :"):
                Dihedral = float(output[idx].split(":")[1].strip().split()[0].strip())
            elif output[idx].startswith("    whole:"):
                rama_z = float(output[idx].split(":")[1].strip().split()[0].strip())

            elif output[idx].startswith("    Cbeta Deviations :"):
                cbeta_deviations = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif output[idx].startswith("      Cis-proline     :"):
                CIS_PROLINE = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif output[idx].startswith("      Cis-general     :"):
                CIS_GENERAL = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif output[idx].startswith("      Twisted Proline :"):
                TWISTED_PROLINE = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif output[idx].startswith("      Twisted General :"):
                TWISTED_GENERAL = float(
                    output[idx].split(":")[1].strip().split()[0].strip()
                )
            elif output[idx].startswith("Map-model CC (overall)"):
                CC_mask, CC_volume, CC_peaks, CC_box = [
                    float(line.split(":")[1].strip())
                    for line in output[idx + 2 : idx + 6]
                ]
            elif output[idx].startswith("Main chain:"):
                CC_mc = float(output[idx + 2].strip().split()[0])
                chain_cc_region = False
            elif per_chain_cc and chain_cc_region:
                fields = output[idx].strip().split()
                # import pdb;pdb.set_trace()
                print(output[idx].strip())
                chain_cc[fields[0]] = fields[1]
            elif per_chain_cc and output[idx].startswith("chain ID  CC"):
                chain_cc_region = True
            elif output[idx].startswith("Side chain:"):
                try:
                    # breakpoint()
                    CC_sc = float(output[idx + 2].strip().split()[0])
                except ValueError:
                    CC_sc = 0.0
    # import pdb;pdb.set_trace()
    return {
        "CC_mask": CC_mask,
        "CC_volume": CC_volume,
        "CC_peaks": CC_peaks,
        "CC_box": CC_box,
        "CC_mc": CC_mc,
        "CC_sc": CC_sc,
        "CC_per_chain": chain_cc,
        "clashscore": clashscore,
        "rama_outliers": rama_outliers,
        "rama_allowed": rama_allowed,
        "rama_favored": rama_favored,
        "rotamer_outliers": rotamer_outliers,
        "rotamer_allowed": rotamer_allowed,
        "rotamer_favored": rotamer_favored,
        "cbeta_deviations": cbeta_deviations,
        "cis_proline": CIS_PROLINE,
        "cis_general": CIS_GENERAL,
        "twisted_proline": TWISTED_PROLINE,
        "twisted_general": TWISTED_GENERAL,
        "Bond": Bond,
        "Angle": Angle,
        "Chirality": Chirality,
        "Planarity": Planarity,
        "Dihedral": Dihedral,
        "rama_z": rama_z,
    }


def load_vc(vc_path):
    line = open(vc_path, "r").readlines()[0]
    kv = line.strip().split()[1:]
    vc_dict = {}
    for i in range(len(kv) // 2):
        vc_dict[kv[i * 2].strip(":")] = float(kv[i * 2 + 1])
    return vc_dict


def score_vc(vc: dict):
    clash10WT = 3
    clash20WT = 1
    ccWT = 5
    ccVal = 0.73

    weight_sum = 0
    # geometry metrics (no clashscore)
    for m, f in zip(metrics, goal_funcs):
        goal = globals()[f](vc[m])
        if goal:
            weight_sum += 1

    # clashscore
    if vc["clashscore"] < 10:
        weight_sum += clash10WT
    elif vc["clashscore"] < 20:
        weight_sum += clash20WT
    # cc
    cc_avg = sum([vc[cc] for cc in cc_metrics]) / 4.0
    if cc_avg > ccVal:
        weight_sum += ccWT

    return weight_sum


def select_vc(vc_path: list):
    assert len(vc_path) > 0
    json_res = dict()

    ret_vc = None
    max_wt = 0
    for p in vc_path:
        vc = load_vc(p)
        cur_wt = score_vc(vc)
        json_res[os.path.basename(p)] = cur_wt
        # print(vc_path[i])
        # print(cur_wt)
        if cur_wt > max_wt:
            ret_vc = p
            max_wt = cur_wt

    wkdir = os.path.dirname(ret_vc)
    emdb = os.path.basename(ret_vc).split("_")[0]
    json_str = json.dumps(json_res, indent=4)
    res_path=vc_path[0].replace("rsr.vcx","refine.json")
    open(res_path, "w").write(json_str)

    return ret_vc


def save_vc(inputfile, suffix=None):
    inf = Path(inputfile)

    if suffix is None:
        basename = os.path.basename(inputfile)
        emdb = basename.split(".")[0].split("_")[0]
        suffix = basename.replace(emdb, "")

    per_chain_cc = False
    cc_dict = parse_vc(inputfile, per_chain_cc)
    pdbname = inf.name.replace(suffix, "").replace(".log", "")
    names = pdbname.split("_")
    if len(names) == 2:
        pdbname = names[0]
        mapname = names[1]
    else:
        mapname = "---"
    line = f"{pdbname}"
    for k in cc_dict.keys():
        if k != "CC_per_chain":
            line += f" {k}: {cc_dict[k]}"
    # line = f"{pdbname} CC_mask: {cc_dict['CC_mask']} CC_volume: {cc_dict['CC_volume']} CC_peaks: {cc_dict['CC_peaks']} CC_box: {cc_dict['CC_box']} CC_mc: {cc_dict['CC_mc']} CC_sc: {cc_dict['CC_sc']}  "
    print(line)
    if inputfile.endswith(".vc"):
        outpath = inputfile.replace(".vc", ".vcx")
    else:
        outpath = f"{inputfile}.vcx"

    if not "None" in line:
        with open(outpath, "w") as f:
            f.write(line + "\n")


def is_none_vcx(path):
    if os.path.exists(path):

        with open(path, "r") as f:
            vcx = f.read()
            if len(vcx) > 0 and "None" in vcx:
                return True
            else:
                return False
    else:
        return True


def run_validation(map_path: str, pdb_path: str, r: float):
    wkdir = os.path.dirname(pdb_path)
    # emdb = os.path.basename(pdb_path).replace(in_suffix+".pdb", "")
    emdb = os.path.basename(pdb_path).split(".")[0].split("_")[0]

    # log_path=pdb_path.replace(".pdb", ".vc")
    log_path = pdb_path.replace(".pdb", ".vc").replace(".cif", ".vc")
    vcx_path = log_path.replace(".vc", ".vcx")

    # map_path=os.path.join(mdir, emdb+M_SUF)
    # map_path=os.path.join(wkdir,emdb+M_SUF)
    # r=resolu_dict.get(emdb,RESOLU)
    # if r==0.0:
    #     logger.warning(f"No resolution for emdb {emdb} in the index file!!!")
    #     r=RESOLU

    # breakpoint()
    # cmd=f"phenix.validation_cryoem {pdb_path} {map_path} resolution={str(r)} > {log_path}"
    # cmd1=f"phenix.map_model_cc {pdb_path} {map_path} resolution={r} >> {log_path}_cc"
    if not os.path.exists(log_path) or is_none_vcx(vcx_path):
        # cmd0=f"rm {log_path}; "
        # ret=os.system(cmd0)

        cmd1 = f"source {phenix_env};cd {wkdir};phenix.map_model_cc {pdb_path} {map_path} resolution={r} >> {log_path}"
        # cmd2=f"phenix.molprobity {pdb_path} >> {log_path}_geo"
        cmd2 = f"source {phenix_env};cd {wkdir};phenix.molprobity {pdb_path} coot=False probe_dots=False>> {log_path}"
        logger.info(cmd1)
        logger.info(cmd2)

        t = time.perf_counter()
        ret = os.system(cmd1)
        ret = os.system(cmd2)
        validation_time = time.perf_counter() - t
        logger.info("Validation time for {:s}: {:.4f}s".format(emdb, validation_time))

    if is_none_vcx(vcx_path):
        save_vc(log_path)

    if ret == 0 and os.path.exists(log_path):
        return True, log_path
    else:
        return False, log_path


def run_validation_or_not(pdb_path):
    log_path = pdb_path.replace(".pdb", ".vc").replace(".cif", ".vc")
    vcx_path = pdb_path.replace(".vc", ".vcx")
    if (
        os.path.exists(log_path)
        and os.path.getsize(log_path) > 0
        and os.path.exists(vcx_path)
    ):
        comple = False
        with open(log_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                if "All-atom Clashscore" in l:
                    comple = True
        if comple:
            return True
        else:
            return False
    else:
        return False


def run_phenix_gm(wkdir: str, pdb_path: str, r: float):
    base_path = pdb_path.replace(".pdb", "").replace(".cif", "")
    out_pathgm = f"{base_path}_gm"
    out_path = f"{base_path}_gm.pdb"

    if not os.path.exists(out_path):
        cmd = f"source {phenix_env};cd {wkdir};phenix.geometry_minimization {pdb_path} output_file_name_prefix={out_pathgm} write_geo_file=False "

        logger.info(cmd)
        print(cmd)
        t = time.perf_counter()
        ret = os.system(cmd)
        rsr_time = time.perf_counter() - t
    else:
        logger.info(
            f"Phenix real_space_refine for {pdb_path} has been done before, skip it"
        )
        rsr_time = 0.0
    if not run_validation_or_not(out_path):
        # validation of the first step
        _, log1 = run_validation(map_path, pdb_path=out_path, r=r)
        # sys.exit(0)

    return True, out_path


def run_phenix_rsr(
    map_path: str, pdb_path: str, out_suffix: str = "_rsr", r: float = 4.0
):
    wkdir = os.path.dirname(pdb_path)

    emdb = os.path.basename(pdb_path).split(".")[0].split("_")[0]
    base_path = pdb_path.replace(".pdb", "").replace(".cif", "")

    weight, nw = get_weight(r)
    phenix_params["weight"] = weight
    phenix_params["nonbonded_weight"] = nw
    phenix_params["output.suffix"] = out_suffix
    # phenix_params['reference_model.file']=os.path.join(wkdir,emdb+".cif")

    out_suffix = phenix_params["output.suffix"]
    # map_path=os.path.join(wkdir,emdb+M_SUF)

    # out_pathgm = f"{base_path}{out_suffix}_gm"
    # out_pathgmf = f"{base_path}{out_suffix}_gm.pdb"

    log_path = f"{base_path}{out_suffix}.log"
    log0_path = f"{base_path}{out_suffix}_000.log"
    out_path = f"{base_path}{out_suffix}_000.pdb"
    out_pathf = f"{base_path}{out_suffix}.pdb"
    out_pathx = f"{base_path}{out_suffix}_00*.*"

    if not (os.path.exists(out_path) or os.path.exists(out_pathf)):
        cmd = (
            f"source {phenix_env};cd {wkdir};phenix.real_space_refine {pdb_path} {map_path} resolution={r} "
        )
        params = " ".join([f"{k}={v}" for k, v in phenix_params.items()])
        params += f" output_dir={wkdir}"
        # params+=f" >/dev/null 2>&1"
        # params+=f" >{log_path}"
        logger.info(cmd + params)
        print(cmd + params)

        t = time.perf_counter()
        os.system(f"rm {out_pathx}")
        ret = os.system(cmd + params)
        if os.path.exists(log0_path):
            os.rename(log0_path, log_path)
        rsr_time = time.perf_counter() - t
        logger.info(
            "Phenix real_space_refine time for {:s}: {:.4f}s".format(emdb, rsr_time)
        )
    else:
        logger.info(
            f"Phenix real_space_refine for {pdb_path} has been done before, skip it"
        )
        rsr_time = 0.0
    if os.path.exists(out_path) and not os.path.exists(out_pathf):
        os.rename(out_path, out_pathf)
    # out_path=os.path.join(wkdir,emdb+out_suffix+"_000"+P_SUF)
    # out_path=execCmd(f"ls {pdb_path}").split()[-1].strip()
    # if ret==0 and out_path!=None: return True,out_path
    # else: return False, out_path
    if not run_validation_or_not(out_pathf):
        # validation of the first step
        _, log1 = run_validation(map_path, pdb_path=out_pathf, r=r)
        # sys.exit(0)

    return True, out_pathf


# def run_eff(wkdir:str,pdb_path:str,out_suffix=""):
#     emdb=os.path.basename(pdb_path).split(".")[0].split("_")[0].split("-")[0]
#     map_path=os.path.join(wkdir,emdb+M_SUF)
#     r=resolu_dict.get(emdb,RESOLU)
#     cmd1=f"source {phenix_env};source {phenix_env};cd {wkdir};phenix.map_model_cc {pdb_path} {map_path} resolution={r}"
#     cmd2=f"source {phenix_env};source {phenix_env};cd {wkdir};phenix.clashscore {pdb_path} > clash.log"
#     os.system(cmd1)
#     os.system(cmd2)
#     reference=get_constraints(os.path.join(wkdir,"cc_per_residue.log"),os.path.join(wkdir,"clash.log"))
#     w,nbw=get_weight(resolu=r)
#     with open(os.path.join(wkdir,emdb+".eff"),'w') as wf:
#         wf.write("\n".join(eff_params))
#         wf.write(f"\npdb_interpretation{{nonbonded_weight={nbw}}}")
#         wf.write(f"\nweight={w}")
#         wf.write(f'''\nreference_model{{\nenabled=True\nreference_group{{\nreference={reference}\nselection={reference}\nfile_name={pdb_path}\n}}sigma=0.05\n}}''')


def run_phenix_eff(
    map_path: str,
    pdb_path: str,
    out_suffix: str = "_eff",
    r: float = 4.0,
    cc_threshold=0.7,
):
    wkdir = os.path.dirname(pdb_path)

    emdb = os.path.basename(pdb_path).split(".")[0].split("_")[0]
    # log_path=os.path.join(wkdir,emdb+out_suffix+".log")
    log_path = pdb_path.replace(".pdb", f"{out_suffix}.rsr")

    # map_path  = os.path.join(wkdir,emdb+M_SUF)
    out_path = pdb_path.replace(".pdb", f"{out_suffix}_000.pdb")
    out_pathf = pdb_path.replace(".pdb", f"{out_suffix}.pdb")
    out_pathx = pdb_path.replace(".pdb", f"{out_suffix}_00*.*")
    cpr_path = pdb_path.replace(".pdb", f"{out_suffix}.cpr")
    cls_path = pdb_path.replace(".pdb", f"{out_suffix}.cls")
    mmc_path = pdb_path.replace(".pdb", f"{out_suffix}.mmc")
    eff_path = pdb_path.replace(".pdb", f"{out_suffix}.eff")

    # r=resolu_dict.get(emdb,0.0)
    # if r==0.0:
    #     logger.warning(f"No resolution for emdb {emdb} in the index file!!!")
    #     r=RESOLU

    if not (os.path.exists(out_path) or os.path.exists(out_pathf)):

        if not os.path.exists(cpr_path):
            cmd1 = f"source {phenix_env};cd {wkdir};phenix.map_model_cc {pdb_path} {map_path} resolution={r}>{mmc_path};mv cc_per_residue.log {cpr_path}"
            os.system(cmd1)

        if not os.path.exists(cls_path):
            cmd2 = f"source {phenix_env};cd {wkdir};phenix.clashscore {pdb_path} > {cls_path}"
            os.system(cmd2)

        reference = get_constraints(cpr_path, cls_path, cc_threshold=cc_threshold)

        w, nbw = get_weight(resolu=r)
        eff_params["output.suffix"] = out_suffix
        params = "\n".join([f"{k}={v}" for k, v in eff_params.items()])

        with open(eff_path, "w") as wf:
            # wf.write("\n".join(eff_params))
            wf.write(params)
            wf.write(f"\npdb_interpretation{{nonbonded_weight={nbw}}}")
            wf.write(f"\nweight={w}")
            if len(reference)>0:
                wf.write(
                f"""\nreference_model{{\nenabled=True\nreference_group{{\nreference={reference}\nselection={reference}\nfile_name={pdb_path}\n}}sigma=0.05\n}}"""
            )

        cmd = f"source {phenix_env};cd {wkdir};phenix.real_space_refine {pdb_path} {map_path} resolution={r} {eff_path}"

        t = time.perf_counter()
        os.system(f"rm {out_pathx}")
        ret = os.system(cmd)
        rsr_time = time.perf_counter() - t
        logger.info(
            "Phenix real_space_refine time for {:s}: {:.4f}s".format(emdb, rsr_time)
        )
        if os.path.exists(out_path) and not os.path.exists(out_pathf):
            os.rename(out_path, out_pathf)
    else:
        logger.info(
            f"Phenix real_space_refine for {pdb_path} has been done before, skip it"
        )
        rsr_time = 0.0

    # out_path=os.path.join(wkdir,emdb+out_suffix+"_000"+P_SUF)
    # out_path=execCmd(f"ls {pdb_path}").split()[-1].strip()
    # if ret==0 and out_path!=None: return True,out_path
    # else: return False, out_path
    if not run_validation_or_not(out_pathf):
        # validation of the first step
        _, log1 = run_validation(map_path, pdb_path=out_pathf, r=r)

    return True, out_pathf


def run_pyr(wkdir: str, pdb_path: str, out_suffix="_pyr"):
    import pyrosetta

    emdb = os.path.basename(pdb_path).split(".")[0].split("_")[0]
    # emdb = os.path.basename(pdb_path).replace(P_SUF, "")
    # emdb=os.path.basename(pdb_path).split(".")[0].split("_")[0]
    # emdb="".join(filter(str.isdigit,emdb))
    # out_path=os.path.join(wkdir,emdb+out_suffix+P_SUF)
    out_path = pdb_path.replace(".pdb", f"{out_suffix}.pdb")

    if not os.path.exists(out_path):

        # turn off the stdout of pyrosetta
        sys.stdout = open(os.devnull, "w")

        pyrosetta.init(" ".join(pyr_init_cmd))
        pose = pyrosetta.pose_from_pdb(pdb_path)
        scorefxn = pyrosetta.create_score_function("ref2015_cart")
        fr = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn)
        fr.constrain_relax_to_start_coords(True)
        t = time.perf_counter()
        fr.apply(pose)
        relaxation_time = time.perf_counter() - t
    else:
        logger.info(f"Pyrosetta relax for {emdb} has been done before, skip it")
        relaxation_time = 0.0

    # turn on stdout
    sys.stdout = sys.__stdout__
    logger.info("Relaxation time for {:s}: {:.4f}s".format(emdb, relaxation_time))
    pose.dump_pdb(out_path)
    if os.path.exists(out_path):
        lines_pdb = []
        with open(out_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(
                    "# All scores below are weighted scores, not raw scores."
                ):
                    break
                else:
                    lines_pdb.append(line)
        with open(out_path, "w") as f:
            f.writelines(lines_pdb)

    if os.path.getsize(out_path) > 0:
        return True, out_path
    else:
        return False, out_path


def concat(wkdir: str, bfactor: str, others: str, out_suffix="_bf"):
    # name=os.path.basename(others).split(".")[0].split("_")[0]
    # out_path=os.path.join(wkdir,name+"_cct"+P_SUF)
    # name = os.path.basename(others).replace(P_SUF, "")
    print(others)
    out_path = others.replace(".pdb", f"{out_suffix}.pdb")

    with open(bfactor, "r") as r1:
        bfactor_lines = r1.readlines()
    with open(others, "r") as r2:
        others_lines = r2.readlines()

    temp_dict = dict()
    for l in bfactor_lines:
        if not l.startswith("ATOM"):
            continue
        k = l[21:26]
        v = l[60:67]
        temp_dict[k] = v

    outLines = list()
    for l in others_lines:
        if not l.startswith("ATOM"):
            outLines.append(l)
            continue
        else:
            b_factor = temp_dict.get(l[21:26], BFACTOR_DEFAULT)
            outLines.append(l[:60] + b_factor + l[67:])

    with open(out_path, "w") as wf:
        wf.writelines(outLines)
    if len(outLines) > 0:
        return True, out_path
    else:
        return False, out_path


def run_phenix_geo(
    map_path: str, pdb_path: str, out_suffix: str = "_gmn", r: float = 4.0
):
    wkdir = os.path.dirname(pdb_path)
    emdb = os.path.basename(pdb_path).split(".")[0].split("_")[0]
    res_pdb = pdb_path.replace(".pdb", "_minimized.pdb")
    gm_pdb = pdb_path.replace(".pdb", f"{out_suffix}.pdb")
    gm_vcx = pdb_path.replace(".pdb", f"{out_suffix}.vcx")

    if not os.path.exists(gm_pdb):
        cmd = f"source {phenix_env};cd {wkdir};phenix.geometry_minimization {pdb_path} " + " ".join(
            geo_params
        )
        cmd = cmd + f";mv {res_pdb} {gm_pdb}"
        logger.info(cmd)
        t = time.perf_counter()
        os.system(cmd)
        geo_time = time.perf_counter() - t
        logger.info("Geo_minimization time for {:s}: {:.4f}s".format(emdb, geo_time))

    if not run_validation_or_not(gm_pdb):
        # validation of the first step
        _, log1 = run_validation(map_path, pdb_path=gm_pdb, r=r)

    if os.path.exists(gm_pdb) and os.path.getsize(gm_pdb) > 0:
        return True, gm_pdb
    else:
        return False, gm_pdb


def perform_refine(pdb_path, map_path, r):
    wkdir = os.path.dirname(pdb_path)
    if not run_validation_or_not(pdb_path):
        _, log0 = run_validation(map_path, pdb_path, r)

    # inp_pdb = pdb_path
    # out_suffix = "_gmn"
    # out_pdb = inp_pdb.replace(".pdb", f"{out_suffix}.pdb")
    # out_log = out_pdb.replace(".pdb", f".vc")
    # gmn_vcx = out_pdb.replace(".pdb", f".vcx")
    # if not os.path.exists(gmn_vcx):
    #     ret, out_pdb = run_phenix_geo(map_path, inp_pdb, out_suffix, r)

    inp_pdb = pdb_path
    out_suffix = "_rsr"
    out_pdb = inp_pdb.replace(".pdb", f"{out_suffix}.pdb")
    out_log = out_pdb.replace(".pdb", f".vc")
    rsr_vcx = out_pdb.replace(".pdb", f".vcx")
    if not os.path.exists(rsr_vcx):
        ret, out_pdb = run_phenix_rsr(map_path, inp_pdb, out_suffix, r)

    inp_pdb = out_pdb
    out_suffix = "_eff1"
    out_pdb = inp_pdb.replace(".pdb", f"{out_suffix}.pdb")
    out_log = out_pdb.replace(".pdb", f".vc")
    eff1_vcx = out_pdb.replace(".pdb", f".vcx")
    if not os.path.exists(eff1_vcx):
        ret, out_pdb = run_phenix_eff(
            map_path, inp_pdb, out_suffix, r, cc_threshold=0.7
        )

    inp_pdb = out_pdb
    out_suffix = "_eff2"
    out_pdb = inp_pdb.replace(".pdb", f"{out_suffix}.pdb")
    out_log = out_pdb.replace(".pdb", f".vc")
    eff2_vcx = out_pdb.replace(".pdb", f".vcx")
    if not os.path.exists(eff2_vcx):
        ret, out_pdb = run_phenix_eff(
            map_path, inp_pdb, out_suffix, r, cc_threshold=0.8
        )

    eff_res = load_vc(eff2_vcx)
    gmn_vcx = out_pdb.replace(".pdb", f".vcx")

    if eff_res["clashscore"] > 10:
        logger.info("Clashscore >10, start geometry minimization")

        inp_pdb = out_pdb
        out_suffix = "_gmn"
        out_pdb = inp_pdb.replace(".pdb", f"{out_suffix}.pdb")
        out_log = out_pdb.replace(".pdb", f".vc")
        gmn_vcx = out_pdb.replace(".pdb", f".vcx")
        if not os.path.exists(gmn_vcx):
            ret, out_pdb = run_phenix_geo(map_path, inp_pdb, out_suffix, r)

    vcx_list = [rsr_vcx, eff1_vcx, eff2_vcx]
    if os.path.exists(gmn_vcx) and os.path.getsize(gmn_vcx) > 0:
        vcx_list.append(gmn_vcx)

    res_vcx = select_vc(vcx_list)
    res_pdb = res_vcx.replace(".vcx", ".pdb")
    # res_emdb=os.path.basename(res_pdb).split(".")[0].split("_")[0]
    # refine_pdb=os.path.join(wkdir,res_emdb+"_refine.pdb")
    refine_pdb = pdb_path.replace(".pdb", "_refine.pdb")
    refine_vcx = pdb_path.replace(".pdb", "_refine.vcx")
    os.system(f"cp {res_pdb} {refine_pdb}")
    os.system(f"cp {res_vcx} {refine_vcx}")


if __name__ == "__main__":

    # breakpoint()
    
    pdb_path = os.path.realpath(sys.argv[1])
    map_path = os.path.realpath(sys.argv[2])



    if len(sys.argv) > 3:
        r = float(sys.argv[3])
    else:
        emdb = os.path.basename(pdb_path).split(".")[0].split("_")[0]
        r = resolu_dict.get(emdb, 4.0)
        if r == 0.0:
            logger.warning(f"No resolution for emdb {emdb} in the index file!!!")
            r = RESOLU

    if len(sys.argv) > 4:
        opt = sys.argv[4]
    else:
        opt = None

    if opt is None:
        perform_refine(pdb_path, map_path, r)

    else:
        if opt == "vcx":
            out_suffix = "_eff5_gmn"

            # gmn_log = eff_pdb.replace(".pdb", f"{out_suffix}.vc")
            gmn_log = pdb_path.replace(".pdb", f"{out_suffix}.vc")
            gmn_vcx = pdb_path.replace(".pdb", f"{out_suffix}.vcx")

            if not os.path.exists(gmn_vcx) and os.path.exists(gmn_log):
                vcx_path = gmn_log.replace(".vc", ".vcx")
                save_vc(gmn_log)
                print("vcx saved.")
