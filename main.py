from hashlib import sha1 as sha
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
import uuid, json, base64, hmac, requests, time, math, sys, argparse

requests.packages.urllib3.disable_warnings()

access_key_id = 'LTAI5tMYMwUVxXoKJkw4eULx'
access_key_secret = 'csYGnHaOgwtg0uvLNPG5qk5E25VcgY'
upload_url = 'https://cryonet.oss-accelerate.aliyuncs.com'
host = 'https://cryonet.ai'
create_job_url = f'{host}/api/create_job/'
query_job_url = f'{host}/api/query_job/'
stop_job_url = f'{host}/api/stop_job'
email = 'cryonet@cryonet.ai'
random_32_str = ''
dir_prefix = ''
check_job_start_time = 0
files = [
  './7770.mrc',
  './7770.fasta'
]

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

def random_string(len):
  return str(uuid.uuid4()).replace('-', '')[:32]

def generate_signature(access_key_secret, expiration, conditions):
  policy_dict = {
    'expiration': expiration,
    'conditions': conditions
  }
  policy = json.dumps(policy_dict).strip()
  policy_encode = base64.b64encode(policy.encode())
  h = hmac.new(access_key_secret.encode(), policy_encode, sha)
  sign_result = base64.b64encode(h.digest()).strip()
  return sign_result.decode()

def generate_upload_params(file_name):
  policy = {
    'expiration': '2028-01-01T12:00:00.000Z',
    'conditions': [
      ['content-length-range', 0, 2147483648]
    ]
  }
  signature = generate_signature(access_key_secret, policy.get('expiration'), policy.get('conditions'))
  response = {
    'name': file_name,
    'key': f'{dir_prefix}' + '/${filename}',
    'policy': base64.b64encode(json.dumps(policy).encode('utf-8')).decode(),
    'OSSAccessKeyId': access_key_id,
    'success_action_status': '200',
    'signature': signature,
  }
  # print('~~~~~', json.dumps(response, indent=2))
  return response

def upload_file(file_url, file_name):
  def progress_callback(monitor):
    print('\r', end="")
    progress = int((monitor.bytes_read / monitor.len) * 100)
    print(f"{file_name}, : {progress}% ", end="")

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
    print(f'\n{file_name}, ok, {math.floor(end_time - start_time)} s')
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
    'email': email,
    'mode': '41',
  }
  response = requests.post(create_job_url, data=params, verify=False)
  if response.status_code == 200:
    job_id = response.json()['jobid']
    # print(f'job id: ' + job_id)
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
        print(f'status: {msg} {progress}%,  {diff}s', end="")
        print(f"\033[K", end="")
        time.sleep(10)
      else:
        pd = data['pdb']['pd']
        # for key in pdb:
        #   response = requests.get(f'{host}/{pdb[key]}', verify=False)
        #   pdb_name = pdb[key].split("/")[-1]
        #   open(f'./{pdb_name}', 'wb').write(response.content)
        response = requests.get(f'{host}/{pd}', verify=False)
        pdb_name = pd.split("/")[-1]
        open(f'./{pdb_name}', 'wb').write(response.content)
        print(f'Complete!')
        print(f'Model saved at {pdb_name}.')


    else:
      print(f'\n{job_id} check job failed\n', json.dumps(data, indent=2))
      break

def main():
  global random_32_str, dir_prefix, check_job_start_time

  random_32_str = random_string(32)
  dir_prefix = f'jobs/{random_32_str}'

  map_file = files[0]
  seq_file = files[1]
  map_file_name = map_file.split("/")[-1]
  seq_file_name = seq_file.split("/")[-1]

  print(f'{len(files)} files')

  upload_file(map_file, map_file_name)
  upload_file(seq_file, seq_file_name)

  job_id = create_job(map_name=map_file_name, map_file=f'{dir_prefix}/{map_file_name}', seq_file=f'{dir_prefix}/{seq_file_name}')

  # check job
  check_job_start_time = time.time()
  check_job(job_id)
  
main()
