import os
import time
from datetime import datetime,timedelta
import glob


def load_fasta(seqFn, rem_tVersion=False, load_annotation=False):
    """
    seqFn               -- Fasta file
    rem_tVersion        -- Remove version information. ENST000000022311.2 => ENST000000022311
    load_annotation     -- Load sequence annotation

    Return:
        {tid1: seq1, ...} if load_annotation==False
        {tid1: seq1, ...},{tid1: annot1, ...} if load_annotation==True
    """
    fasta = {}
    annotation = {}
    cur_tid = ''
    cur_seq = ''
    
    for line in open(seqFn):
        if line[0] == '>':
            if cur_seq != '':
                fasta[cur_tid] = cur_seq
                cur_seq = ''
            data = line[1:-1].split(None, 1)
            cur_tid = data[0]
            annotation[cur_tid] = data[1] if len(data)==2 else ""
            if rem_tVersion and '.' in cur_tid: 
                cur_tid = ".".join(cur_tid.split(".")[:-1])
        elif cur_tid != '':
            cur_seq += line.rstrip()
    
    if cur_seq != '':
        fasta[cur_tid] = cur_seq
    
    if load_annotation:
        return fasta, annotation
    else:
        return fasta

dir_path="/data/jobs/f5/jobs_241210_templ"
dir_path="/data/jobs/f5/jobs_241210_pdb"
log_files= glob.glob(f"{dir_path}/*/*.log")

for log in log_files:
    pdb_dir=os.path.dirname(log)
    pdb_name=os.path.basename(log).split(".")[0].replace("_cf","")
    seq_file=os.path.join(pdb_dir,f"{pdb_name}.fasta")
    seq_dict = load_fasta(seq_file,load_annotation=False)
    seq_len=0
    for seq in seq_dict.values():
        seq_len+=len(seq)
    # print(f"{pdb_name} {seq_len}")
    with open(log,'r') as rlog:
        lines=rlog.readlines()
        start_time=datetime.strptime(lines[0].strip(), "%a %b %d %H:%M:%S CST %Y")
        end_time=datetime.strptime(lines[-1].strip(), "%a %b %d %H:%M:%S CST %Y")
        duration=(end_time-start_time).total_seconds()
        print("{} {} {:.1f}".format(pdb_name, seq_len,duration))


