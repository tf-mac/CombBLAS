from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("rcm.pyx", annotate=True),
    include_dirs=[np.get_include()]
)
