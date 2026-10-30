#!/bin/bash
pdb_id=$1

emd_id=$(grep $pdb_id" " /home/jsptgpu/data/db/emdb/emdb_list_info.csv|awk '{print($1)}')
echo $emd_id
exp/f1/dl_f1_emd.sh $emd_id
