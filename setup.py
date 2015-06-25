#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from setuptools import setup, find_packages
 
import ICode
 
setup( 
    name='ICode',
 
    version=ICode.__version__,

    packages=find_packages(),
 
    author="Hubert Pelle",
 
    author_email="hubert.pelle.x2012@gmail.com",
 
    description="Post-processing fMRI datas, for tudying Hurst Exponent",
 
    long_description=open('README.md').read(),
 
    include_package_data=True,
 
    url='https://github.com/JFBazille/ICode.git',
 

    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 1 - Planning",
        "License :: OSI Approved",
        "Natural Language :: French",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Communications",
    ],
 
 

    #entry_points = no entry point for now
 
    license="WTFPL",
 
)