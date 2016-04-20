#!/usr/bin/env python
from setuptools import setup, find_packages

def readme():
  with open('README.md') as f:
    return f.read()

setup(name='NLLFitter',
    version='0.1',
    description='NLL fitting tools for bump hunting.',
    url='https://github.com/naodell/amumu',
    author='Nathaniel Odell',
    author_email='naodell@gmail.com',
    license='NU',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False)
