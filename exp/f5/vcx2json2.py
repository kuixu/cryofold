from pathlib import Path
import json
import sys

# user defined
vcx_path=sys.argv[1]
# fasta_path=sys.argv[2]
status_path=sys.argv[2]
# vcx_path="0109.vcx"
# fasta_path="0109.faa"
# status_path="0109_status.json"
#json_path=vcx_path.replace("vcx","json")

json_path = Path(vcx_path).parent / "status"  # 修改为同目录下的 status.json



vc_lines=open(vcx_path,'r').readlines()
vc_dict=dict()
for l in vc_lines:
    items=l.strip().split()
    score=dict()
    emdb=items[0]
    for i in range(0,(len(items)-1)//2):
        score[items[i*2+1].strip(":")]=float(items[i*2+2])
    vc_dict[emdb]=score

json_list=list()
for idx,s in vc_dict.items():
    score=dict()

    score['metrics']=s
   
    # status
    with open(status_path,'r') as rf:
        status_dict=json.load(rf)
    for k,v in status_dict.items():
        score[k]=v
    # print(score)
    json_list.append(score)

json_list = json_list[0]
with open(json_path,'w') as jf:
    json.dump(json_list,jf,indent=4)

