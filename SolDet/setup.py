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
            'pyparsing>=2.0.1,<3.0',
            'absl-py',
            'ipykernel',
            'wrapt',
            # TensorFlow 2.7.2 required dependencies
            'astunparse>=1.6.0',
            'flatbuffers<3.0,>=1.12',
            'gast<0.5.0,>=0.2.1',
            'google-pasta>=0.1.1',
            'grpcio<2.0,>=1.24.3',
            'keras<2.8,>=2.7.0rc0',
            'keras-preprocessing>=1.1.1',
            'libclang>=9.0.1',
            'opt-einsum>=2.3.2',
            'tensorboard~=2.6',
            'tensorflow-estimator<2.8,~=2.7.0rc0',
            'tensorflow-io-gcs-filesystem>=0.21.0',
            'termcolor>=1.1.0'
        ]
)

