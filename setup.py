import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))



setup(
    name='sim3solve',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(name='sim3solve',
            sources=['fastloop/solve.cpp'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            },
            include_dirs=[
                osp.join(ROOT, 'thirdparty/eigen-3.4.0')]
            )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

