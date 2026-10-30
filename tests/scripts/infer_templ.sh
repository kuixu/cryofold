#!/bin/bash

in=tests/backbone/8638_A.pdb
o0=tests/backbone/8638_A_i0.pdb
o1=tests/backbone/8638_A_i1.pdb

in=tests//8638/8638.fasta
wd=tests//8638/templ
map_file=tests//8638/8638.mrc
cus_templ_path=tests//8638/templ/8638
# python cryofold/model/backbone_imp.py -i $in -o $o0 -p $o1
    # infer_templates(input_fasta, work_dir, ncores=ncores, min_pdb_templ_cov=min_pdb_templ_cov, min_identity=min_identity, map_file=map_file, cus_templ_path=cus_templ_path)
# parser.add_argument('-i', '--input_fasta', type=str,            required=True, help="Protein sequence fasta file.")
#     parser.add_argument('-w', '--work_dir',    type=str,            required=True, help='Working directory.')
#     parser.add_argument('--exclude_pdbs',      type=str,            required=False, help='PDB list to excluded, multiple PDB seperated by comma')
#     parser.add_argument('--ncores',            type=int,            default=8,      help='Max number of cores to processing templates')
#     parser.add_argument('--min_pdb_templ_cov', type=float,          default=0.01,   help='Minimun coverage of PDB template of sequence')
#     parser.add_argument('--min_identity',      type=float,          default=0.2,   help='Minimun coverage of PDB template of sequence')
    
#     parser.add_argument('--max_pdb_date',      type=str,            required=False, help='Maximun PDB template date (e.g. 2021-12-05)')
#     parser.add_argument('--max_pdb_templ',     type=int,            default=1000,     help='Maximun PDB templates (default: 100)')
#     parser.add_argument('--pdb_mode',          type=str,            choices=['none', 'phmmer', 'exactly'], default='phmmer',   help="none: don't use PDB,  phmmer: search PDB  with mismatch, exactly: search PDB  without mismatch")
#     parser.add_argument('--map_file',          type=str,            default=None,   help="MRC file")
#     parser.add_argument('--cus_templ_path',        type=str,            default=None,   help="CIF file")
python infer_templ.py -i $in -w $wd --map_file $map_file --cus_templ_path $cus_templ_path