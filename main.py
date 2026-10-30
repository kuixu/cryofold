#!/usr/bin/env python
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
import json, requests, time, math, sys, argparse

requests.packages.urllib3.disable_warnings()

upload_url = dir_prefix = check_job_start_time = None
host = 'https://cryonet.ai'
gen_sig_url = f'{host}/api/gen_sig/'
create_job_url = f'{host}/api/create_job/'
query_job_url = f'{host}/api/query_job/'
stop_job_url = f'{host}/api/stop_job'
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
  if response.status_code == 200:
    job_id = response.json()['jobid']
    return job_id
  else:
    print(f'Failed in creating job\n', json.dumps(response.json(), indent=2))
    sys.exit()

def check_job(job_id):
  stg = 0
  while stg != 5:
    response = requests.get(f'{query_job_url}?jobid={job_id}&r=2', verify=False)
    if response.status_code == 200:
      data = response.json()
      stg = data['stg']
      if stg < 5:
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

def main():
  global check_job_start_time

  get_dir_prefix()

  map_file = files[0]
  seq_file = files[1]
  map_file_name = map_file.split("/")[-1]
  seq_file_name = seq_file.split("/")[-1]

  print(f'{len(files)} files')

  upload_file(map_file, map_file_name)
  upload_file(seq_file, seq_file_name)

  job_id = create_job(map_name=map_file_name, map_file=f'{dir_prefix}/{map_file_name}', seq_file=f'{dir_prefix}/{seq_file_name}')

  check_job_start_time = time.time()
  check_job(job_id)
  
main()
