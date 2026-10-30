#!/usr/bin/env python



import os,sys
import numpy as np
from cryofold.np.relax import relax
from cryofold.np import protein

RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
# RELAX_ENERGY_TOLERANCE = 2.39

RELAX_ENERGY_TOLERANCE = 0.1

# RELAX_STIFFNESS = 10.0
RELAX_STIFFNESS = 0.1
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 2
max_attempts = 200

# def do_relax()
pdb_path = sys.argv[1]
relaxed_output_path = sys.argv[2]
debug = 0


if len(sys.argv) > 3:
    debug = 1
    max_attempts = 200


amber_relaxer = relax.AmberRelaxation(
      max_iterations=RELAX_MAX_ITERATIONS,
      tolerance=RELAX_ENERGY_TOLERANCE,
      stiffness=RELAX_STIFFNESS,
      exclude_residues=RELAX_EXCLUDE_RESIDUES,
      max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
      # max_attempts=max_attempts,
      use_gpu=True)


# output_dir = os.path.dirname(pdb_path)
name = os.path.basename(pdb_path)
unrelaxed_protein = protein.from_pdb_string(open(pdb_path).read())
if debug:
      chain_ids = np.unique(unrelaxed_protein.chain_index).tolist()
      relaxed_pdb_str = ""
      for chain_id in chain_ids:
            print(f"Relaxing chain: {chain_id}")
            unrelaxed_chain = unrelaxed_protein.filter_by_chain(chain_id)
            try:
                  relaxed_chain_str, debug_data, violations = amber_relaxer.process(prot=unrelaxed_chain)
                  relaxed_chain = protein.from_pdb_string(relaxed_chain_str)
                  relaxed_chain.chain_index = relaxed_chain.chain_index + chain_id
                  relaxed_chain_str = protein.to_pdb(relaxed_chain)
                  relaxed_chain_arr = relaxed_chain_str.split("\n")
                  relaxed_chain_str = "\n"+"\n".join(relaxed_chain_arr[1:-3])


            except:
                  print(f"Error: Could not relax chain {chain_id}")
                  relaxed_chain_str = protein.to_pdb(unrelaxed_chain)
                  relaxed_chain_arr = relaxed_chain_str.split("\n")
                  relaxed_chain_str = "\n"+"\n".join(relaxed_chain_arr[1:-3])
            relaxed_pdb_str += relaxed_chain_str




      # import pdb;pdb.set_trace()
else:
      # Relax the prediction.
      # t_0 = time.time()
      relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)

# Save the relaxed PDB.
# relaxed_output_path = os.path.join(
#     output_dir, f'{name}')
with open(relaxed_output_path, 'w') as f:
    f.write(relaxed_pdb_str)