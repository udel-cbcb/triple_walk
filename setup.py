import setuptools
from setuptools import find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import glob
import os
import torch

with open('README.md') as f:
    long_description = f.read()

def get_extension():

    # get the sources
    sources = glob.glob('csrc/**/*.cpp',recursive=True)
    #sources.extend(glob.glob('csrc/**/*.h',recursive=True))

    # get the cuda sources
    if torch.cuda.is_available() and torch.version.cuda:
        sources_cuda = glob.glob('csrc/**/*.cu',recursive=True)   
        sources.extend(sources_cuda)

        # remove file names having hip
        sources_cleaned = []
        for file_name in sources:
            if "hip" not in file_name:
                sources_cleaned.append(file_name)

        sources = sources_cleaned
        
    else:
        sources_hip = glob.glob('csrc/**/*.hip',recursive=True)   
        sources.extend(sources_hip)
    
    print(sources)

    # openmp
    extra_compile_args = {'cxx': ['-O2']}
    extra_compile_args['cxx'] += ['-DAT_PARALLEL_OPENMP']
    extra_compile_args['cxx'] += ['-fopenmp']

    # cuda
    define_macros = []
    define_macros += [('WITH_CUDA', None)]
    nvcc_flags = os.getenv('NVCC_FLAGS', '')
    nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
    nvcc_flags += ['-arch=sm_52', '-O2']
    extra_compile_args['nvcc'] = nvcc_flags

    # set include dirs
    include_dirs = ["csrc"]

    if torch.cuda.is_available() and torch.version.hip:
        # add rocrand headers
        include_dirs.extend([
            "/opt/rocm/include/rocrand",
            "/opt/rocm/include/hiprand"
        ])

    extension = CUDAExtension(
        'triple_walk_native',
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args
    )

    return extension

setuptools.setup(
    name="triple_walk",
    version="0.0.3",
    author="Sachin Gavali",
    author_email="saching@udel.edu",
    description="A pytorch extension library to perform triple walks on knowledge graphs",
    long_description = long_description,
    long_description_content_type="text/markdown",
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
    ext_modules=[get_extension()],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch==1.9.0',
        'numpy==1.19.5',
        'pandas==1.3.2',
        'networkx==2.6.2',
        'pytest==6.2.4',
    ],
    packages=find_packages(),
)