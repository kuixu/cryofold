#!/bin/bash
emd_id=$1

datadir=/home/jsptgpu/data/db/emdb/

pdb_id=$(grep ^$emd_id" " /home/jsptgpu/data/db/emdb/emdb_list_info.csv|awk '{print($2)}')
echo $pdb_id
pdb=$datadir/${pdb_id}.mrc
map=$datadir/emd_${emd_id}.map
mapgz=$datadir/emd_${emd_id}.map.gz
if [ ! -f "$map" ]; then
    echo "${map} does not exist."
    # wget ftp://ftp.pdbj.org/pub/emdb/structures/EMD-${emd_id}/map/emd_${emd_id}.map.gz -O $name.map.gz
    # rsync -rlpt -v -z --delete ftp.pdbe.org::emdb/structures/EMD-${emd_id}/map/emd_${emd_id}.map.gz $datadir/ 
    rsync -rlpt -v -z --delete rsync.ebi.ac.uk::pub/databases/emdb/structures/EMD-${emd_id}/map/emd_${emd_id}.map.gz $datadir/ 
    # >>/dev/null 2>&1
    if [ -f "$mapgz" ]; then
        gunzip $mapgz
    fi

fi
if [ "----" != "$pdb_id" ] && [ ! -f "$pdb" ] && [ -f "$map" ]; then
    cp $map $pdb
    

fi