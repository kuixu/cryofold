#!/bin/bash

in=tests/backbone/8638_A.pdb
o0=tests/backbone/8638_A_i0.pdb
o1=tests/backbone/8638_A_i1.pdb

in=tests//8638/templ_cifdb//0/8638_B:1-324.pdb 
o0=tests//8638/templ_cifdb//0/msa_idx_0.pdb 
o1=tests//8638/templ_cifdb//0/msa_idx_1.pdb
# python cryofold/model/backbone_imp.py -i $in -o $o0 -p $o1
python backbone_imp.py -i $in -o $o0 -p $o1