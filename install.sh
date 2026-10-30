#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

pwd_dir=$PWD
echo "CryoFold parameter file exist."
is_env_installed=$(conda info --envs | grep cryofold -c)
if [[ "${is_env_installed}" == "0" ]];then
  conda create -y -n cryofold python=3.12 gcc_linux-64 gxx_linux-64
else
  echo "Detected an existing cryofold environment, exiting installation";
  exit 1;
fi


if [[ `command -v activate` ]]
then
  source `which activate` cryofold
else
  conda activate cryofold
fi
  
if [[ "${CONDA_DEFAULT_ENV}" != "cryofold" ]]
then
  echo "Could not run conda activate cryofold, please check the errors";
  exit 1;
fi
conda install -y pdbfixer go gxx==9.5 mkl=2021 
conda install -y hmmer mmseqs2 kalign2 hhsuite -c bioconda
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install dm-tree biopython==1.84 absl-py pytorch_lightning pandas deepspeed ml-collections mrcfile git+https://github.com/NVIDIA/dllogger.git
pip install biotite==0.41.2 msgpack_numpy biotite cloudpathlib einops transformers==4.46.3 tenacity brotli importlib_metadata easydict
# fp
pip installtorch_geometric==2.6.1 torch_cluster==1.6.3 e3nn==0.5.4

cd misc/msa-cli
go version
go build --mod=vendor -o msa-cli
cp msa-cli $pwd_dir/tools/msa-cli
cd $pwd_dir

python misc/perf_mod.py

python_exc="${CONDA_PREFIX}/bin/python"
export CUDA_HOME=/usr/local/cuda-12.1
$python_exc setup.py install
echo "done!"
