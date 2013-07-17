from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = "interpoltools",
    ext_modules = cythonize('interpoltools.pyx'),
    include_dirs = [numpy.get_include()]
)
