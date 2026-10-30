#!/usr/bin/env python
import os,pickle,gzip,shutil,sys
import argparse

conda_path = shutil.which('python')
conda_lib = conda_path.replace('bin/python', 'lib')

if os.path.exists(conda_lib):
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = conda_lib + ":" + os.environ['LD_LIBRARY_PATH'].lstrip(':')
    else:
        os.environ['LD_LIBRARY_PATH'] = conda_lib

from cryofold.data.get_msa import generate_input_features_pklfile
from cryofold.data.get_msa_api import generate_input_features_pklfile_api
from cryofold.data.templates_utils import process_templates
from cryofold.model.inference import run_inference
from cryofold.model.inference_de import run_inference_de
from cryofold.libs.structure.density import DensityInfo



def validate_input_files(map_path, seq_path, tep_path):
    assert os.path.exists(map_path), f"Map file not found at {map_path}"
    msg = 'Validating input map file...\n'
    try:
        den = DensityInfo(map_path)
    except Exception as e:
        # print(e)
        msg += f'Error in reading map file: {type(e)}: {e}. '
        msg.replace("<class","")
        msg.replace(">","")
        msg += f"\nSolution: Check the map file {map_path} format and try again."
        print(msg)
        sys.exit(1)

        # if 'KeyError' in msg:
        #     msg += "\nSolution: Open and save the map as a new *.mrc file using Chimera(X), and re-submit the job."
        # jdata.update({
        #     'error_code' :1,
        #     'msg': msg,
        # })
    assert os.path.exists(seq_path), f"Sequence file not found at {seq_path}"
    # assert os.path.exists(tep_path), f"Template file not found at {tep_path}"


def argparser():
    parser = argparse.ArgumentParser(description='cif2pdb')

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--map',      type=str, help="Cryo-EM density map file path in mrc format")
    parser.add_argument('-s', '--sequence', type=str, help="Sequence file path in fasta format")
    parser.add_argument('-t', '--template', type=str, help="Template file path in mmCIF format")
    parser.add_argument('-r', '--resolution', type=float, help="Resolution of the density map")
    parser.add_argument('-f', '--features_pklgz', type=str, default='', help="Features file path in *.pkl.gz format")
    parser.add_argument('-d', '--mode',  type=int, default=0, help='Inference mode, 0 for respecting Template, 1 for respecting density map. ')
    parser.add_argument('--msa_only',     action='store_true', help='perform only one step of generating input features (MSA and templates).')
    parser.add_argument('--msa_local',    action='store_true', help='generate input features (MSA and templates) using local database.')
    parser.add_argument('--msa_wo',    action='store_true', help='generate input features (MSA and templates) using local database.')
    
    args = parser.parse_args()
    return args 



if __name__ == '__main__':

    args = argparser()
    map_path = args.map
    seq_path = args.sequence
    tep_path = args.template
    resolution = args.resolution
    fea_path = args.features_pklgz
    wrk_dir = os.path.dirname(map_path)
    validate_input_files(map_path, seq_path, tep_path)
    tep_dir = f"{wrk_dir}/.templ"
    tep_name = os.path.basename(tep_path).split('.')[0]
    cus_templ_path = f"{tep_dir}/{tep_name}"
    msa_wo = args.msa_wo
    

    if not os.path.exists(fea_path):
        fea_path = f"{wrk_dir}/features.pkl.gz"
    
    if msa_wo:
        os.system(f"touch {fea_path}")

    if not os.path.exists(fea_path):
        if args.msa_local:
            print(f"Generating input features (MSA and templates) using local database")
            generate_input_features_pklfile(seq_path, fea_path)
        else:
            try:
                generate_input_features_pklfile_api(seq_path, fea_path)
                assert os.path.exists(fea_path), f"Failed to generate input features (MSA and templates) file using API"

            except Exception as e:
                print(f"Re-generate input features (MSA and templates) using local database")
                generate_input_features_pklfile(seq_path, fea_path)
        assert os.path.exists(fea_path), f"Failed to generate input features (MSA and templates) using local database"
    else:
        print(f"Using existing input features (MSA and templates) file at {fea_path}")
        try:
            features = pickle.load(gzip.open(fea_path, 'rb'))
        except:
            print(f"Failed to load input features (MSA and templates) file at {fea_path}, please check the file or re-generate it.")
    if not args.msa_only:
        # try:
        process_templates(tep_path, tep_dir, map_path)
        if args.mode == 0:
            run_inference(seq_path, fea_path, map_path, tep_dir, cus_templ_path=cus_templ_path, msa_wo=msa_wo)
        elif args.mode == 1:
            run_inference_de(seq_path, fea_path, map_path, tep_dir, cus_templ_path=cus_templ_path)
        else:
            print(f"Invalid mode: {args.mode}, please choose 0 or 1.")
        # except Exception as e:
        #     print(e)
    



