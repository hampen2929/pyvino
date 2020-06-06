import codecs
import os
from setuptools import setup, find_packages

PACKAGE = 'pyvino'
README = 'README.md'
REQUIREMENTS = 'requirements.txt'

VERSION = '0.0.3'

DESCRIPTION = 'This is the python implementation of OpenVINO models.'


def read(fname):
    # file must be read as utf-8 in py3 to avoid to be bytes
    return codecs.open(os.path.join(os.path.dirname(__file__), fname),
                       encoding='utf-8').read()

setup(name=PACKAGE,
      version=VERSION,
      # long_description=read(README),
      install_requires=list(read(REQUIREMENTS).splitlines()),
      url='https://github.com/hampen2929/pyvino',
      author='hampen2929',
      author_email='yuya.mochimaru.ym@gmail.com',
      packages=find_packages(),
      license='Apache'
      )
