from distutils.core import setup
from distutils.extension import Extension

setup(
    name="miniconv",
    python_requires='>3.5.2',
    ext_modules=[
        Extension(
            "miniconv",
            ["conv.cpp"],
            libraries=["boost_python3", "boost_numpy3"],
            extra_compile_args=["-ggdb3", "-O0", "-D_GLIBCXX_DEBUG", "-march=native"], )
    ])
