import os,sys
import logging
import pyrosetta
from pyrosetta import logging_support
import time
import subprocess
# import argparse

# adjust these paths before running
cdir=os.getcwd()
mdir="/data/share/map"
pdir="/data/share/1211rf"
resolu_idx="/data/share/1211rf/CFDB_res.tsv"

logger = logging.getLogger(__file__)
consoleHandler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
logger.setLevel(logging.ERROR)
# logger.setLevel(logging.INFO)

M_SUF=".map"
P_SUF=".pdb"
RESOLU=5.0 # default resolution
BFACTOR_DEFAULT="100.00 " #default b-factor


# create an emdb-resolution dictionary
resolu_dict=dict()
with open(resolu_idx,'r') as rf:
    lines=rf.readlines()
for l in lines:
    l=l.strip().split()
    resolu_dict[l[0]]=l[1]

# phenix.real_space_refine parameters
phenix_params={
    "nproc":"8",
    "refinement.macro_cycles":2,
    "ncs_constraints":"False",
    "secondary_structure.enabled":"True",
    "remove_outliers":"False",
    "run":"minimization_global+simulated_annealing+local_grid_search+adp+occupancy+nqh_flips+morphing+rigid_body",
    "simulated_annealing":"every_macro_cycle",
    "morphing":"once",
    "max_reasonable_bond_distance":"100",

    "output.suffix":"_rf",
    "model_format":"pdb",
    "scattering_table":"electron",
    "pdb_interpretation.secondary_structure.protein.remove_outliers":"True",
    "write_initial_geo_file":"False",
    "skip_map_model_overlap_check":"True",

}

# pyrosetta fast_relax parameters
pyr_init_cmd=[
    "-mute all",
    "-relax:default_repeats 2",
    "-default_max_cycles 200",
    "-relax:constrain_relax_to_start_coords",
    "-relax:ramp_constraints false",
    "-relax:dualspace true",
    "-relax:minimize_bond_angles",
    "-relax:jump_move true",
    "-relax:bb_move true",
    "-relax:chi_move true"
]

def execCmd(cmd):
    r=os.popen(cmd)
    text=r.read()
    r.close()
    return text

def run_validation(wkdir:str,pdb_path:str,out_suffix="_val"):
    emdb=os.path.basename(pdb_path).split(".")[0].split("_")[0]
    emdb="".join(filter(str.isdigit,emdb))
    # print(emdb)
    log_path=os.path.join(work_dir,emdb+out_suffix+".log")
    map_path=os.path.join(mdir,emdb+M_SUF)
    # os.system(f"chmod o+r {map_path}")

    r=resolu_dict.get(emdb,0.0)
    if r==0.0:
        logger.warning(f"No resolution for emdb {emdb} in the index file!!!")
        r=RESOLU

    cmd=f"phenix.validation_cryoem {pdb_path} {map_path} resolution={str(r)} > {log_path}"
    logger.info(cmd)

    t=time.perf_counter()
    ret=os.system(cmd)
    validation_time=time.perf_counter()-t
    logger.info("Validation time for {:s}: {:.4f}s".format(emdb,validation_time))

    
    if ret==0 and os.path.exists(log_path): 
        return True,log_path
    else:
        return False,log_path

def run_pyr(wkdir:str,pdb_path:str,out_suffix="_pyr"):
    emdb=os.path.basename(pdb_path).split(".")[0].split("_")[0]
    emdb="".join(filter(str.isdigit,emdb))
    out_path=os.path.join(wkdir,emdb+out_suffix+P_SUF)

    # turn off the stdout of pyrosetta
    sys.stdout = open(os.devnull, 'w')

    pyrosetta.init(" ".join(pyr_init_cmd))
    pose=pyrosetta.pose_from_pdb(pdb_path)
    scorefxn = pyrosetta.create_score_function('ref2015_cart')
    fr = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn)
    fr.constrain_relax_to_start_coords(True)
    t=time.perf_counter()
    fr.apply(pose)
    relaxation_time=time.perf_counter()-t

    # turn on stdout
    sys.stdout = sys.__stdout__
    logger.info("Relaxation time for {:s}: {:.4f}s".format(emdb,relaxation_time))
    pose.dump_pdb(out_path)

    if os.path.getsize(out_path)>0:
        return True, out_path
    else:
        return False, out_path


def run_phenix_rsr(wkdir:str,pdb_path:str,out_suffix=phenix_params["output.suffix"]):
    emdb=os.path.basename(pdb_path).split(".")[0].split("_")[0]
    emdb="".join(filter(str.isdigit,emdb))
    log_path=os.path.join(wkdir,emdb+out_suffix+".log")
    map_path=os.path.join(mdir,emdb+M_SUF)
    r=resolu_dict.get(emdb,0.0)
    if r==0.0:
        logger.warning(f"No resolution for emdb {emdb} in the index file!!!")
        r=RESOLU
    
    cmd=f"cd {wkdir};phenix.real_space_refine {pdb_path} {map_path} resolution={r} "
    params=" ".join([f"{k}={v}" for k,v in phenix_params.items()])
    params+=f" output_dir={wkdir}"
    params+=f" >{log_path}"
    logger.info(cmd+params)


    t=time.perf_counter()
    ret=os.system(cmd+params)
    rsr_time=time.perf_counter()-t
    logger.info("Phenix real_space_refine time for {:s}: {:.4f}s".format(emdb,rsr_time))


    # out_path=os.path.join(wkdir,emdb+out_suffix+"_000"+P_SUF)
    out_path=execCmd(f"ls {os.path.join(wkdir,emdb)}*{out_suffix}*{P_SUF}").split()[-1].strip()
    if ret==0 and out_path!=None: return True,out_path
    else: return False, out_path

def concat(wkdir:str,bfactor:str,others:str,out_suffix="_cct"):
    name=os.path.basename(others).split(".")[0].split("_")[0]
    out_path=os.path.join(wkdir,name+"_cct"+P_SUF)

    with open(bfactor,'r') as r1:
        bfactor_lines=r1.readlines()
    with open(others,'r') as r2:
        others_lines=r2.readlines()

    temp_dict=dict()
    for l in bfactor_lines:
        if not l.startswith("ATOM"):continue
        k=l[21:26]
        v=l[60:67]
        temp_dict[k]=v
    
    outLines=list()
    for l in others_lines:
        if not l.startswith("ATOM"):
            outLines.append(l)
            continue
        else:
            b_factor=temp_dict.get(l[21:26],BFACTOR_DEFAULT)
            outLines.append(l[:60]+b_factor+l[67:])
    
    with open(out_path,'w') as wf:
        wf.writelines(outLines)
    if len(outLines)>0: return True,out_path
    else: return False, out_path
    



if __name__=='__main__':
    work_dir=sys.argv[1]
    emdb=sys.argv[2]
    if not os.path.exists(work_dir): os.mkdir(work_dir)

    os.system(f"cd {work_dir}")

    # pipeline:
    # step 1: run phenix.real_space_refine
    ini_pdb=os.path.join(pdir,emdb+P_SUF)
    ret,rsr_pdb=run_phenix_rsr(wkdir=work_dir,pdb_path=ini_pdb)


    if ret!=True: 
        logger.error(f"{emdb}:Error occurs during phenix.real_space_refine")
        exit()

    # validation of the first step
    _,log1=run_validation(wkdir=work_dir,pdb_path=rsr_pdb,out_suffix="_v1")
    

    # # step 2: run pyrosetta relax
    ret,pyr_pdb=run_pyr(wkdir=work_dir,pdb_path=rsr_pdb)
    if ret!=True: 
        logger.error(f"{emdb}:Error occurs during pyrosetta relax")
        exit()

    
    # # validation of the 2nd step  
    _,log2=run_validation(wkdir=work_dir,pdb_path=pyr_pdb,out_suffix="_v2")
    

    # # step 3: concatenate two pdbs
    ret,cct_pdb=concat(wkdir=work_dir,bfactor=rsr_pdb,others=pyr_pdb)
    if ret!=True: 
        logger.error(f"{emdb}:Error occurs during pdb concatenation")
        exit()

    # # validation of the 3rd step
    _,log3=run_validation(wkdir=work_dir,pdb_path=cct_pdb,out_suffix="_v3")
   


    

    

    
    

   
    

    








