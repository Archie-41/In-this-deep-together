#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:56:15 2022

@author: sjguo
"""
from setuptools import setup

setup(
    name='soldet',
    version='0.0.1',
    packages=['soldet'],
    long_description=open('README.md').read(),
    python_requires='>=3.8,<3.9',
    install_requires = [
        'numpy==1.22.0',
        'scikit-learn==0.23.1',
        'scipy==1.6.3',
        'tensorflow==2.7.2',
        'tqdm==4.47.0',
        'matplotlib==3.2.2',
        'lmfit==1.0.1',
        'h5py==3.1.0',
        'pandas==1.0.5',
        'seaborn==0.10.1',
        'protobuf>=3.9.2,<4.0',
        'pyparsing>=2.0.1,<3.0']
)

