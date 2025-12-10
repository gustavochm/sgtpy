from setuptools import setup, Extension, find_packages
import numpy

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

ext_modules = []
if use_cython:
    ext_modules = cythonize([
        Extension('sgtpy.coloc_cy', ['sgtpy/src/coloc_cy.pyx'],
                  include_dirs=[numpy.get_include()]),
        Extension('sgtpy.sgt.cijmix_cy', ['sgtpy/src/cijmix_cy.pyx'],
                  include_dirs=[numpy.get_include()])
    ])
else:
    ext_modules = [
        Extension('sgtpy.coloc_cy', ['sgtpy/src/coloc_cy.c'],
                  include_dirs=[numpy.get_include()]),
        Extension('sgtpy.sgt.cijmix_cy', ['sgtpy/src/cijmix_cy.c'],
                  include_dirs=[numpy.get_include()])
    ]

setup(
    packages=find_packages(include=['sgtpy', 'sgtpy.*']),
    ext_modules=ext_modules,
    include_package_data=True,
    zip_safe=False
)
