#!/bin/bash
# ./run_phenix_v1.sh /path/to/pdb /path/to/map /map/resolution 
pdb=$1
map=$2
r=$3
nproc=8

phenix.real_space_refine $pdb $map \
             secondary_structure.enabled=True \
             remove_outliers=False \
             nproc=${nproc} \
             resolution=${r} \
             ncs_constraints=False \
             model_format=pdb \
             scattering_table=electron \
             refinement.macro_cycles=5 \
             pdb_interpretation.secondary_structure.protein.remove_outliers=True \
             run=minimization_global+simulated_annealing+local_grid_search+nqh_flips+occupancy+adp+morphing \
             simulated_annealing=every_macro_cycle \
             morphing=once\
             write_initial_geo_file=False \
             skip_map_model_overlap_check=True \
             max_reasonable_bond_distance=100 \
             

