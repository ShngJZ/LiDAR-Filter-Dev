from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='LiDARClean_ops',
    ext_modules=[
        CUDAExtension('LiDARClean_ops', [
            'LiDARClean_ops_cpp.cpp',
            'LiDARClean_ops_cu.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })