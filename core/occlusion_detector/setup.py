from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='occdtc_ops',
    ext_modules=[
        CUDAExtension('occdtc_ops', [
            'occdtc_cpp.cpp',
            'occdtc_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })