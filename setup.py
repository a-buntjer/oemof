#! /usr/bin/env python

"""
This file is part of project oemof (github.com/oemof/oemof). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location oemof/setup.py

SPDX-License-Identifier: GPL-3.0-or-later
"""

from setuptools import find_packages, setup
import os

import oemof


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='oemof',
      version=oemof.__version__,
      author='oemof developer group',
      author_email='oemof@rl-institut.de',
      description='The open energy modelling framework',
      url='https://oemof.org/',
      namespace_package=['oemof'],
      long_description=read('README.rst'),
      packages=find_packages(),
      install_requires=['blinker < 2.0',
                        'dill < 0.4',
                        'numpy >= 1.7.0, < 1.18',
                        'pandas >= 0.18.0, < 0.26',
                        'pyomo >= 4.4.0, < 5.7',
                        'networkx < 3.0'],
      extras_require={'datapackage': ['datapackage']},
      entry_points={
          'console_scripts': [
              'oemof_installation_test = ' +
              'oemof.tools.console_scripts:check_oemof_installation']})
