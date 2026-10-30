#!/usr/bin/env python
import sys
from  pathlib import Path
import random

def parse_cc(ccfile, per_chain_cc=False):

    CC_mask, CC_volume, CC_peaks, CC_box = None, None, None, None
    CC_mc, CC_sc = None, None
    clashscore = None
    rama_outliers, rama_allowed, rama_favored = None, None, None

    is_rama_outliers = False
    is_rotamer_outliers = False

    rotamer_outliers, rotamer_allowed, rotamer_favored = None, None, None
    cbeta_deviations = None
    chain_cc={}
    chain_cc_region = False
    Bond, Angle, Chirality, Planarity, Dihedral = None, None, None, None, None
    # if per_chain_cc :
    #     Per chain:
    rama_z = None
    with open(ccfile, 'r') as f:
        output = f.readlines()
        # output = output.split('\n')
        CIS_PROLINE = 0
        CIS_GENERAL = 0
        TWISTED_PROLINE = 0
        TWISTED_GENERAL = 0
        for idx in range(len(output)):
            if   output[idx].startswith('    All-atom Clashscore'):
                clashscore = float(output[idx].split(':')[1].strip())
            elif output[idx].startswith('    Ramachandran Plot:'):
                is_rama_outliers = True
                is_rotamer_outliers = False
            elif output[idx].startswith('    Rotamer:'):
                is_rama_outliers = False
                is_rotamer_outliers = True
            # elif output[idx].startswith('  ROTAMER OUTLIERS :'):
                # rotamer_outliers = float(output[idx].split(':')[1].strip())
                # rotamer_outliers = float(output[idx].split(':')[1].strip().split()[0].strip())

                # rotamer_allowed = round(random.uniform(0, 10),2)
                # rotamer_favored = 100 - rotamer_allowed

            elif is_rama_outliers and output[idx].startswith('      Outliers :'):
                rama_outliers = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif is_rama_outliers and output[idx].startswith('      Allowed  :'):
                rama_allowed = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif is_rama_outliers and output[idx].startswith('      Favored  :'):
                rama_favored = float(output[idx].split(':')[1].strip().split()[0].strip())
            
            elif is_rotamer_outliers and output[idx].startswith('      Outliers :'):
                rotamer_outliers = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif is_rotamer_outliers and output[idx].startswith('      Allowed  :'):
                rotamer_allowed = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif is_rotamer_outliers and output[idx].startswith('      Favored  :'):
                rotamer_favored = float(output[idx].split(':')[1].strip().split()[0].strip())

            elif output[idx].startswith('    Bond      :'):
                Bond = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('    Angle     :'):
                Angle = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('    Chirality :'):
                Chirality = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('    Planarity :'):
                Planarity = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('    Dihedral  :'):
                Dihedral = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('    whole:'):
                rama_z = float(output[idx].split(':')[1].strip().split()[0].strip())
            
            
            elif output[idx].startswith('    Cbeta Deviations :'):
                cbeta_deviations = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('      Cis-proline     :'):
                CIS_PROLINE = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('      Cis-general     :'):
                CIS_GENERAL = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('      Twisted Proline :'):
                TWISTED_PROLINE = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('      Twisted General :'):
                TWISTED_GENERAL = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('Map-model CC (overall)'):
                CC_mask, CC_volume, CC_peaks, CC_box = [ float(line.split(':')[1].strip()) for line in output[idx+2:idx+6] ]
            elif output[idx].startswith('Main chain:'):
                CC_mc = float(output[idx+2].strip().split()[0])
                chain_cc_region=False
            elif per_chain_cc and chain_cc_region:
                fields = output[idx].strip().split()
                # import pdb;pdb.set_trace()
                print(output[idx].strip())
                chain_cc[fields[0]] = fields[1]
            elif per_chain_cc and output[idx].startswith('chain ID  CC'):
                chain_cc_region = True
            elif output[idx].startswith('Side chain:'):
                try:
                    CC_sc = float(output[idx+2].strip().split()[0])
                except IndexError:
                    CC_sc = 0.0
    # import pdb;pdb.set_trace()
    return {
        'CC_mask':     CC_mask, 
        'CC_volume':   CC_volume, 
        'CC_peaks':    CC_peaks, 
        'CC_box':      CC_box,
        'CC_mc':       CC_mc,
        'CC_sc':       CC_sc,
        'CC_per_chain':chain_cc,
        'clashscore':  clashscore,

        'rama_outliers':rama_outliers,
        'rama_allowed':rama_allowed,
        'rama_favored':rama_favored,

        'rotamer_outliers':rotamer_outliers,
        'rotamer_allowed':rotamer_allowed,
        'rotamer_favored':rotamer_favored,

        'cbeta_deviations':cbeta_deviations,
        'cis_proline':CIS_PROLINE,
        'cis_general':CIS_GENERAL,
        'twisted_proline':TWISTED_PROLINE,
        'twisted_general':TWISTED_GENERAL,
        "Bond": Bond,
        "Angle": Angle,
        "Chirality": Chirality,
        "Planarity": Planarity,
        "Dihedral": Dihedral,
        "rama_z": rama_z
    }

inputfile = sys.argv[1]
suffix = sys.argv[2]

per_chain_cc = False
inf = Path(inputfile)
if inf.exists():
    cc_dict = parse_cc(inputfile, per_chain_cc)
    pdbname = inf.name.replace(suffix,"").replace(".log","")
    names = pdbname.split('_')
    if len(names)==2:
        pdbname = names[0]
        mapname = names[1]
    else:
        mapname = '---'
    line = f"{pdbname}"
    for k in cc_dict.keys():
        if k != 'CC_per_chain':
            line += f" {k}: {cc_dict[k]}"
    # line = f"{pdbname} CC_mask: {cc_dict['CC_mask']} CC_volume: {cc_dict['CC_volume']} CC_peaks: {cc_dict['CC_peaks']} CC_box: {cc_dict['CC_box']} CC_mc: {cc_dict['CC_mc']} CC_sc: {cc_dict['CC_sc']}  "
    print(line)

    line_per_chain = "\n".join([f"{k} {v}" for k,v in cc_dict['CC_per_chain'].items() ])+"\n"
    path_per_chain = f"{inf.parent}/{pdbname}_cc_per_chain.list"
    with open(path_per_chain,"w") as f:
        f.write(line_per_chain)
    # print("per chain file: ", path_per_chain)
    # import pdb;pdb.set_trace()
        

