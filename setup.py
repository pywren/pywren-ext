#!/usr/bin/env python
import os
import sys

#import pkgconfig
from setuptools import setup, find_packages

if sys.version_info < (2,7):
    sys.exit('Sorry, Python < 2.7 is not supported')

if sys.version_info > (3,) and sys.version_info < (3, 4) :
    sys.exit('Sorry, Python3 version < 3.4 is not supported')

# http://stackoverflow.com/questions/6344076/differences-between-distribute-distutils-setuptools-and-distutils2


setup(
    name='pywren-ext',
    version=0.01,
    url='http://pywren.io',
    author='Eric Jonas',
    description='Utilities and miscellaneary for PyWren', 
    author_email='jonas@eecs.berkeley.edu',
    packages=find_packages(),
    install_requires=[
        'pywren'
    ],
    # entry_points =
    # { 'console_scripts' : ['pywren=pywren.scripts.pywrencli:main', 
    #                        'pywren-setup=pywren.scripts.setupscript:interactive_setup', 
    #                        'pywren-server=pywren.scripts.standalone:server']},
    # package_data={'pywren': ['default_config.yaml', 
    #                          'ec2_standalone_files/ec2standalone.cloudinit.template', 
    #                          'ec2_standalone_files/supervisord.conf', 
    #                          'ec2_standalone_files/supervisord.init', 
    #                          'ec2_standalone_files/cloudwatch-agent.config', 
    # ]},
    include_package_data=True
)
