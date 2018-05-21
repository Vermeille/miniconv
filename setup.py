from distutils.core import setup
from distutils.extension import Extension

setup(
    name="miniconv",
    python_requires='>3.5.2',
    author='Guillaume "Vermeille" Sanchez',
    description="A minimalistic deep learning library",
    author_email='guillaume.v.sanchez@gmail.com',
    ext_modules=[
        Extension(
            "miniconv",
            ["conv.cpp"],
            libraries=["boost_python3", "boost_numpy3"],
            extra_compile_args=["-O3", "-march=native"], )
    ])
