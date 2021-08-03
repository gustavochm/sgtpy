from setuptools import setup, Extension
# from Cython.Distutils import build_ext

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [Extension('SGTPy.coloc_cy',
                              ['SGTPy/src/coloc_cy.pyx']),
                    Extension('SGTPy.sgt.cijmix_cy',
                              ['SGTPy/src/cijmix_cy.pyx'])]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [Extension('SGTPy.coloc_cy', ['SGTPy/src/coloc_cy.c']),
                    Extension('SGTPy.sgt.cijmix_cy',
                    ['SGTPy/src/cijmix_cy.c'])]

"""
cmdclass = {}
ext_modules = []


ext_modules += [Extension('SGTPy.coloc_cy',
                          ['SGTPy/src/coloc_cy.pyx']),
                Extension('SGTPy.sgt.cijmix_cy',
                          ['SGTPy/src/cijmix_cy.pyx'])]
cmdclass.update({'build_ext': build_ext})
"""

setup(
  name='SGTPy',
  license='MIT',
  version='0.0.12',
  description='SAFT-VR-MIE EOS and SGT',
  author='Gustavo Chaparro Maldonado, Andres Mejia Matallana, Erich A. Muller',
  author_email='gustavochaparro@udec.cl',
  url='https://github.com/gustavochm/SGTPy',
  download_url='https://github.com/gustavochm/SGTPy.git',
  long_description=open('long_description.rst').read(),
  packages=['SGTPy', 'SGTPy.vrmie_mixtures',  'SGTPy.vrmie_pure',
            'SGTPy.gammamie_mixtures', 'SGTPy.gammamie_pure',
            'SGTPy.sgt', 'SGTPy.equilibrium', 'SGTPy.fit'],
  cmdclass=cmdclass,
  ext_modules=ext_modules,
  package_data={'SGTPy': ['database/*']},
  install_requires=['numpy', 'scipy', 'cython', 'numba', 'pandas', 'openpyxl'],
  platforms=["Windows", "Linux", "Mac OS", "Unix"],
  keywords=['SAFT-VR-Mie', 'SGT'],
  zip_safe=False
)
