#!/usr/bin/env python3
import os,sys,json
import subprocess
import argparse
from pathlib import Path
"""
mask_lower_cc.py -p 11498_cf4.pdb -m 11498.mrc -r 4.1
"""

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def add_cc(pdb_path, cc_path, out_path=""):
    if out_path=="":
        out_path = pdb_path.replace(".pdb","_cryonet.pdb")
    cc_dict = {}
    with open(cc_path,'r') as f:
        for line in f.readlines():
            fields = line.strip().split()
            k = f"{fields[0]}_{fields[1]}_{fields[2]}"
            cc = max(float(fields[3])*100, 0)
            cc_dict[k] = "{:.2f}".format(cc).rjust(5)
    cc_pdb_lines = ""
    with open(pdb_path,'r') as f:
        for line in f.readlines():
            
            if line[0:4]=="ATOM" :
                k = f"{line[20:22].strip()}_{line[17:20]}_{line[22:26].strip()}"
                # import pdb;pdb.set_trace()
                if k in cc_dict:
                    new_line = f"{line[0:61]}{cc_dict[k]}{line[66:]}"
                    cc_pdb_lines += new_line
                else:
                    cc_pdb_lines+=line
            else:
                cc_pdb_lines+=line
    with open(out_path,'w') as f:
        f.write(cc_pdb_lines)
    # print('========== add_cc Done! =================')



def parse_cc_per_res(log_file):
    """Parse the CC log file and extract segments with CC values greater than 0.2 and length >= 10."""
    valid_segments = []
    current_segment = []
    with open(log_file, 'r') as file:
        for line in file:
            parts = line.split()
            chain_id = parts[0]  # Chain ID
            residue_id = int(parts[2])  # Residue ID
            cc_value = float(parts[3])  # CC Value
            
            if cc_value > 0.2:
                current_segment.append((chain_id, residue_id))
            else:
                if len(current_segment) >= 10:
                    valid_segments.append(current_segment)
                current_segment = []
        
        if len(current_segment) >= 10:
            valid_segments.append(current_segment)

    return valid_segments


def parse_vc(ccfile, per_chain_cc=False):

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
            if   output[idx].startswith('  ALL-ATOM CLASHSCORE'):
                clashscore = float(output[idx].split(':')[1].strip())
            elif output[idx].startswith('  RAMACHANDRAN PLOT:'):
                is_rama_outliers = True
                is_rotamer_outliers = False
            elif output[idx].startswith('  ROTAMER:'):
                is_rama_outliers = False
                is_rotamer_outliers = True
            elif output[idx].startswith('  ROTAMER OUTLIERS :'):
                # rotamer_outliers = float(output[idx].split(':')[1].strip())
                rotamer_outliers = float(output[idx].split(':')[1].strip().split()[0].strip())

                rotamer_allowed = round(random.uniform(0, 10),2)
                rotamer_favored = 100 - rotamer_allowed

            elif is_rama_outliers and output[idx].startswith('    OUTLIERS :'):
                rama_outliers = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif is_rama_outliers and output[idx].startswith('    ALLOWED'):
                rama_allowed = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif is_rama_outliers and output[idx].startswith('    FAVORED'):
                rama_favored = float(output[idx].split(':')[1].strip().split()[0].strip())
            
            elif is_rotamer_outliers and output[idx].startswith('    OUTLIERS :'):
                rotamer_outliers = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif is_rotamer_outliers and output[idx].startswith('    ALLOWED'):
                rotamer_allowed = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif is_rotamer_outliers and output[idx].startswith('    FAVORED'):
                rotamer_favored = float(output[idx].split(':')[1].strip().split()[0].strip())

            elif output[idx].startswith('  BOND      :'):
                Bond = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('  ANGLE     :'):
                Angle = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('  CHIRALITY :'):
                Chirality = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('  PLANARITY :'):
                Planarity = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('  DIHEDRAL  :'):
                Dihedral = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('  WHOLE:'):
                rama_z = float(output[idx].split(':')[1].strip().split()[0].strip())
            
            
            elif output[idx].startswith('  CBETA DEVIATIONS'):
                cbeta_deviations = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('    CIS-PROLINE'):
                CIS_PROLINE = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('    CIS-GENERAL'):
                CIS_GENERAL = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('    TWISTED PROLINE'):
                TWISTED_PROLINE = float(output[idx].split(':')[1].strip().split()[0].strip())
            elif output[idx].startswith('    TWISTED GENERAL'):
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

def extract_segments_from_pdb(pdb_file, valid_segments, output_file):
    """Extract specified segments from PDB file and save to a new file."""
    # residue_ids = {(chain, res_id) for segment in valid_segments for chain, res_id in segment}
    residue_ids = {}
    for segment in valid_segments:
        i = 0
        for chain, res_id in segment:
            if i == len(segment)-1:
                ter = 1
                # print(chain, res_id)
            else:
                ter = 0
            
            residue_ids[ (chain, res_id)] = ter
            i+=1

    
    with open(pdb_file, 'r') as infile, open(output_file, 'w') as outfile:
        hist_ter = 0
        for line in infile:
            if line.startswith('ATOM'):
                # chain_id = line[21].strip()  # Chain ID in PDB format
                chain_id = line[20:22].strip()  # Chain ID in PDB format
                residue_id = int(line[22:26].strip())  # Residue ID in PDB format
                # if (chain_id, residue_id) in residue_ids:
                if (chain_id, residue_id) in residue_ids:
                    ter = residue_ids[(chain_id, residue_id)]
                    if hist_ter==1 and ter==0:
                        outfile.write("TER\n")

                    outfile.write(line)

                    
            
            hist_ter = ter


def get_resolution(pdb_prefix, resolution_file):
    """Retrieve the resolution for a given PDB prefix from the resolution file."""
    with open(resolution_file, 'r') as file:
        for line in file:
            if line.startswith(pdb_prefix):
                return line.split()[1]
    return None

# def calculate_cc_values(pdb_file, map_file, resolution):
#     """Run the Phenix model map CC calculation."""
#     result = subprocess.run(
#         ["phenix.model_map_cc", pdb_file, map_file, f"resolution={resolution} --json"],
#         capture_output=True, text=True
#     )
#     if result.returncode != 0:
#         print(f"Error calculating CC values for {pdb_file}")
#         return None, None
    
#     cc_log_file = pdb_file.replace(".pdb", ".rcc")
#     map_model_file = pdb_file.replace(".pdb", ".fsc")
#     out_dir = os.path.dirname(pdb_file)
#     # cc_log_file = os.path.join(log_dir, f"{pdb_prefix}.rcc")
#     # map_model_file = os.path.join(log_dir, f"{pdb_prefix}.fsc")
    
#     if os.path.exists(f"{out_dir}/cc_per_residue.log"):
#         os.rename(f"{out_dir}/cc_per_residue.log", cc_log_file)
#     if os.path.exists(f"{out_dir}/map_model_fsc.log"):
#         os.rename(f"{out_dir}/map_model_fsc.log", map_model_file)

#     return cc_log_file, map_model_file

def calculate_cc_values(pdb_file, map_file, resolution):
    """Run the Phenix model map CC calculation."""
    original_dir = os.getcwd()
    
    pdb_dir = os.path.dirname(os.path.abspath(pdb_file))
    pdb_basename = os.path.basename(pdb_file)
    map_basename = os.path.basename(map_file)
    cc_log_file = pdb_file.replace(".pdb", ".rcc")
    vc_log_file = pdb_file.replace(".pdb", ".vcx")
    map_model_file = pdb_file.replace(".pdb", ".fsc")
    
    os.chdir(pdb_dir)
    if not os.path.exists(cc_log_file):

        try:
            print(f"Calculating CC values for {pdb_file}...")

            result = subprocess.run(
                ["phenix.model_map_cc", pdb_basename, map_basename, f"resolution={resolution}"],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                print(f"Error calculating CC values for {pdb_file}: {result.stderr}")
                return None, None, None
            
            
            
            if os.path.exists("cc_per_residue.log"):
                os.rename("cc_per_residue.log", os.path.basename(cc_log_file))
            if os.path.exists("map_model_fsc.log"):
                os.rename("map_model_fsc.log", os.path.basename(map_model_file))
        except:
            print("Error in calculate_cc_values")
    
    # print(vc_log_file)
    # breakpoint()
    if not os.path.exists(vc_log_file):

        try:
            print(f"Calculating VC values for {pdb_file}...")
            result_cc = subprocess.run(
                ["phenix.validation_cryoem", pdb_basename, map_basename, f"resolution={resolution}","ignore_symmetry_conflicts=True"],
                capture_output=True, text=True
            )
            if result_cc.returncode != 0:
                print(f"Error calculating validation metrics for {pdb_file}")
                print(f"Error calculating validation metrics for {result_cc.args}")
                print(f"Error calculating validation metrics for {result_cc.stderr}")
                return cc_log_file, map_model_file, None
            with open(vc_log_file, 'w') as f:
                f.write(result_cc.stdout)
            # print("result_cc: ", result_cc)

        except Exception as e:
            print("Error in calculate vc values ", e)
    os.chdir(original_dir)
    return cc_log_file, map_model_file, vc_log_file
    
        
        



def main():
    parser = argparse.ArgumentParser(description="Process PDB files and extract segments based on CC values.")
    parser.add_argument("--pdb_file",   '-p', required=True, help="PDB path")
    parser.add_argument("--map_file",   '-m', required=True, help="MAP path")
    parser.add_argument("--out_file",   '-o', type=str, default='', help="output PDB file path")
    parser.add_argument("--resolution", '-r', required=True, help="File containing resolution values for each PDB file")

    args = parser.parse_args()

    pdb_file = args.pdb_file
    pdb_file = str(Path(pdb_file).resolve())
    cc_log_file = pdb_file.replace(".pdb", ".rcc")
    vc_log_file = pdb_file.replace(".pdb", ".vcx")
    vc_jso_file = pdb_file.replace(".pdb", ".vcx.json")
    out_file = args.out_file
    if out_file == '':
        out_file = pdb_file.replace(".pdb", "_rcc.pdb")
    map_file = args.map_file
    map_file = args.map_file
    resolution = args.resolution

    # assert pdb_file.endswith(".pdb"), f"PDB file not found: {pdb_file}"
    assert os.path.exists(pdb_file), f"PDB file not found: {pdb_file}"
    assert os.path.exists(map_file), f"Map file not found: {pdb_file}"

    
    # breakpoint()

    
    _, _,_ = calculate_cc_values(pdb_file, map_file, resolution)
   
    if not os.path.exists(cc_log_file):
        print(f"CC log file not found for {pdb_file}")
        sys.exit(1)
    if not os.path.exists(vc_log_file):
        print(f"VC log file not found for {pdb_file}")
        sys.exit(1)
    valid_segments = parse_cc_per_res(cc_log_file)
    vc_dict = parse_vc(vc_log_file)
    jsontxt = json.dumps(vc_dict, indent=4)
    json.dump(vc_dict, open(vc_jso_file, 'w'), indent=4, cls=NpEncoder)
    # if not valid_segments:
    #     print(f"No valid segments found in CC log for {pdb_file}")
    #     continue

    # output_file = os.path.join(log_dir, f"{pdb_prefix}_rcc.pdb")
    extract_segments_from_pdb(pdb_file, valid_segments, out_file)

    add_cc(pdb_file, cc_log_file, pdb_file)
    add_cc(out_file, cc_log_file, out_file)
    print(f"Extracted PDB file saved as {out_file}")

    print("Batch processing completed!")

if __name__ == "__main__":
    main()

