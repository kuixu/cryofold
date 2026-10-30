#!/bin/bash

# ./cryofold  -m tests/8638/8638.mrc -s tests/8638/8638.fasta -t tests/8638/8638.cif
# j=$1
b=$1
r=/data/jobs/f5/rerun_${b}
d=/data/jobs/f5/jobs_24${b}_templ

cd $d
ls */*cf.pdb|awk -F'/' '{print($1)}' > rxlr.list;awk 'NR==FNR{a[$1]=1}NR>FNR{ if(a[$1]!=1)print($1)}' rxlr.list all_seq.list > all_seq_u.list; 

wc -l all_templ.list
wc -l all_seq.list
wc -l all_seq_u.list

mkdir -p $r
awk -v d=$r '{system("cp "$1"/"$1"_cf.pdb "d"/"$1".pdb ")}' rxlr.list

echo "all_seq_u: "
cat all_seq_u.list|sort -nk1|awk '{system("wc -l "$1"/"$1".fasta")}'