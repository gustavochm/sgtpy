from setuptools import setup, Extension
from Cython.Distutils import build_ext


cmdclass = {}
ext_modules = []


ext_modules += [Extension('SGTPy.coloc_cy',
                          ['SGTPy/src/coloc_cy.pyx']),
                Extension('SGTPy.sgt.cijmix_cy',
                          ['SGTPy/src/cijmix_cy.pyx'])]
cmdclass.update({'build_ext': build_ext})


setup(
  name='SGTPy',
  license='MIT',
  version='0.0.2',
  description='SAFT-VR-MIE EOS and SGT',
  author='Gustavo Chaparro Maldonado, Andres Mejia Matallana, Erich A. Muller',
  author_email='gustavochaparro@udec.cl',
  url='https://github.com/gustavochm/SGTPy',
  download_url='https://github.com/gustavochm/SGTPy.git',
  long_description=open('long_description.rst').read(),
  packages=['SGTPy', 'SGTPy.mixtures',  'SGTPy.pure',   'SGTPy.sgt',
            'SGTPy.equilibrium', 'SGTPy.fit'],
  cmdclass=cmdclass,
  ext_modules=ext_modules,
  install_requires=['numpy', 'scipy', 'cython'],
  platforms=["Windows", "Linux", "Mac OS", "Unix"],
  keywords=['SAFT-VR-Mie', 'SGT'],
  zip_safe=False
)
