from setuptools import setup, Extension

setup(
  name = 'SGTPy',
  license='MIT',
  version = '0.0.1',
  description = 'SAFT-VR-MIE EOS',
  author = 'Gustavo Chaparro Maldonado, Andres Mejia Matallana',
  author_email = 'gustavochaparro@udec.cl',
  url = 'https://github.com/gustavochm/SGTPy',
  download_url = 'https://github.com/gustavochm/SGTPy.git',
  long_description = open('long_description.rst').read(),
  packages = ['saftvrmie', 'saftvrmie.mixtures', 'saftvrmie.pure'],
  install_requires = ['numpy','scipy', 'phasepy'],
  platforms = ["Windows", "Linux", "Mac OS", "Unix"],
  keywords = ['SAFT-VR-Mie', 'SGT'],
  zip_safe = False
)
