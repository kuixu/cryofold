# Copyright 2024 CryoFold team
# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from setuptools import setup, Extension, find_packages
import subprocess

from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


version_dependent_macros = [
    '-DVERSION_GE_1_1',
    '-DVERSION_GE_1_3',
    '-DVERSION_GE_1_5',
]

extra_cuda_flags = [
    '-std=c++14', 
    '-maxrregcount=50', 
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__', 
    '--expt-relaxed-constexpr', 
    '--expt-extended-lambda'
]

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

cc_flag = ['-gencode', 'arch=compute_70,code=sm_70']
_, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
if int(bare_metal_major) >= 11:
    cc_flag.append('-gencode')
    cc_flag.append('arch=compute_80,code=sm_80')

extra_cuda_flags += cc_flag

def cuda_funs(name, paths):
    return CUDAExtension(
        name=name,
        sources=[
            os.path.join('cryofold/utils/kernel/csrc', path) for path in paths
        ],
        include_dirs=[
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                'cryofold/utils/kernel/csrc/'
            )
        ],
        extra_compile_args={
            'cxx': ['-O3'] + version_dependent_macros,
            'nvcc': (
                ['-O3', '--use_fast_math'] + 
                version_dependent_macros + 
                extra_cuda_flags
            ),
        }
    )

setup(
    name='cryofold',
    version='1.0.0',
    description='',
    author='',
    author_email='xukui@tsinghua.edu.cn',
    license='MIT License',
    url='https://github.com/kuixu/cryofold',
    packages=find_packages(exclude=["tests", "scripts"]),
    include_package_data=True,
    package_data={
        "cryofold": ['utils/kernel/csrc/*'],
        "": ["np/stereo_chemical_props.txt"]
    },
    ext_modules=[
        cuda_funs("cryofold_attn_cuda", ["attn_cuda.cpp", "attn_cuda_kernel.cu"]),
        cuda_funs("cryofold_fastsoftmax_cuda", ['fastsoftmax_cuda.cpp', 'fastsoftmax_cuda_kernel.cu']),
        cuda_funs("cryofold_layernorm_cuda", ["layernorm_cuda.cpp", "layernorm_cuda_kernel.cu"]),
    ],
    cmdclass={'build_ext': BuildExtension},
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8,' 
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
