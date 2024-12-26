#!/usr/bin/env python
import os,sys, json, requests, time, math, argparse, hashlib
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor

requests.packages.urllib3.disable_warnings()

upload_url = dir_prefix = check_job_start_time = None
host = 'https://cryonet.ai'
gen_sig_url    = f'{host}/api/gen_sig/'
create_job_url = f'{host}/api/create_job/'
query_job_url  = f'{host}/api/query_job/'
stop_job_url   = f'{host}/api/stop_job'
check_md5_url  = f'{host}/api/check_md5'

files = ['', '']

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--map', help="Cryo-EM density map")
parser.add_argument('-s', '--sequence', help="Sequence")

args = parser.parse_args()

if args.map is None:
  print('No map file found.')
  sys.exit()
else:
  files[0] = args.map

if args.sequence is None:
  print('No sequence file found.')
  sys.exit()
else:
  files[1] = args.sequence

def get_dir_prefix():
  global dir_prefix, upload_url
  response = requests.get(f'{gen_sig_url}?filename=""', verify=False)
  dir_prefix = response.json()['dir_prefix']
  upload_url = response.json()['oss_url']

def generate_upload_params(file_name):
  response = requests.get(f'{gen_sig_url}?filename={file_name}', verify=False)
  if response.status_code == 200:
    obj = response.json()
    return {
      'name': file_name,
      'key': f'{dir_prefix}' + '/${filename}',
      'policy': obj['policy'],
      'OSSAccessKeyId': obj['OSSAccessKeyId'],
      'success_action_status': obj['success_action_status'],
      'signature': obj['signature'],
    }
  else:
    return {}


def get_map_seq_md5_v1(mappath, seqpath):
    md5_hash1 = hashlib.md5()
    with open(mappath, 'rb') as file1:
        while True:
            data = file1.read(1024 * 10)
            if not data:
                break
            md5_hash1.update(data)
    
    md5_hash2 = hashlib.md5()
    with open(seqpath, 'rb') as file2:
        while True:
            data = file2.read(1024 * 10)
            if not data:
                break
            md5_hash2.update(data)
    
    md51 = md5_hash1.digest() 
    md52 = md5_hash2.digest()
    total_md5 = hashlib.md5(md51 + md52).hexdigest()

    return total_md5


def get_map_seq_md5(mappath, seqpath):
    md5_hash1 = hashlib.md5()
    with open(mappath, 'rb') as file1:
        while True:
            data = file1.read(1024 * 10)
            if not data:
                break
            md5_hash1.update(data)
    
    md5_hash2 = hashlib.md5()
    with open(seqpath, 'rb') as file2:
        while True:
            data = file2.read(1024 * 10)
            if not data:
                break
            md5_hash2.update(data)
    
    md51 = md5_hash1.hexdigest()
    md52 = md5_hash2.hexdigest()
    code = md51 + md52
    total_md5 = hashlib.md5(code.encode('utf-8')).hexdigest()

    return total_md5


def upload_file(file_url, file_name):
  def progress_callback(monitor):
    print('\r', end="")
    progress = int((monitor.bytes_read / monitor.len) * 100)
    print("{}, progress: {}% ".format(file_name, progress), end="")

  params = {
    **generate_upload_params(file_name),
    'file': (
      file_name, 
      open(file_url, 'rb'),
    ),
  }

  start_time = time.time()
  e = MultipartEncoder(fields=params)
  m = MultipartEncoderMonitor(e, progress_callback)
  response = requests.post(upload_url, data=m, headers={ 'Content-Type': e.content_type })
  end_time = time.time()
  if response.status_code == 200:
    print(f'\n{file_name}, ok,  {math.floor(end_time - start_time)}s')
  else:
    print(f'\n{file_name}, failed\n', json.dumps(response.json(), indent=2))
    sys.exit()

def create_job(map_name, map_file, seq_file):
  print(f'Creating job...')
  params = {
    'mapname': map_name,
    'mapfile': map_file,
    'seqfile': seq_file,
    'name': map_name,
    'mode': '41',
  }
  response = requests.post(create_job_url, data=params, verify=False)
  jdata = response.json()
  if response.status_code == 200 and jdata['error_code'] == 0:
    job_id = response.json()['jobid']
    return job_id
  else:
    # print(f'Failed in creating job\n', json.dumps(response.json(), indent=2))
    print(f'Failed in creating job\n', jdata['msg'])
    sys.exit()

def check_job(job_id):
  stg = 0
  while stg != 5:
    response = requests.get(f'{query_job_url}?jobid={job_id}&r=2', verify=False)
    if response.status_code == 200:
      data = response.json()
      stg = data['stg']
      if stg < 5  :
        if 'msg' in data and 'progress' in data:
            msg = data['msg']
            progress = data['progress']
            diff = math.floor(time.time() - check_job_start_time)
            print('\r', end="")
            print(f'status: {msg}, progress: {progress}%, time: {diff}s', end="")
            print(f"\033[K", end="")
            time.sleep(10)
      else:
        pd = data['pdb']['pd']
        response = requests.get(f'{host}/{pd}', verify=False)
        pdb_name = pd.split("/")[-1]
        open(f'./{pdb_name}', 'wb').write(response.content)
        print(f'Complete!')
        print(f'Model saved at {pdb_name}.')
    else:
      print(f'\n{job_id} check job failed\n', json.dumps(data, indent=2))
      break

def check_md5(mappath, seqpath):
    jmd5 = get_map_seq_md5(mappath, seqpath)
    print(jmd5)
    response = requests.get(f'{check_md5_url}?md5={jmd5}', verify=False)
    if response.status_code == 200:
        obj = response.json()
        if obj['error_code'] == 0:
            return obj['jobid']
        else:
           return None
    else:
        return None

def main():
    global check_job_start_time

    get_dir_prefix()

    map_file = files[0]
    seq_file = files[1]
    map_file_name = os.path.splitext(os.path.basename(map_file))[0]
    seq_file_name = os.path.splitext(os.path.basename(seq_file))[0]

    print(f'{len(files)} files')

    job_id = check_md5(map_file, seq_file)

    if job_id is None: 
        print(f"New job: with {map_file_name} and {seq_file_name}")
        upload_file(map_file, map_file_name)
        upload_file(seq_file, seq_file_name)

        job_id = create_job(map_name=map_file_name, map_file=f'{dir_prefix}/{map_file_name}', seq_file=f'{dir_prefix}/{seq_file_name}')
    else:
        print("Job is in the running.")
    print(f"Visualize and download results: https://cryonet.ai/vis?jobid={job_id}")
    check_job_start_time = time.time()
    check_job(job_id)
  
main()