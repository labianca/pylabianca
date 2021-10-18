#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2021 Mikołaj Magnuski
# <mmagnuski@swps.edu.pl>

from setuptools import find_packages, setup


DESCRIPTION = "Python tools of labianca group: mostly spike and LFP analysis."


if __name__ == "__main__":
    setup(name='pylabianca', version='0.1.0', maintainer='Mikołaj Magnuski',
          maintainer_email='mmagnuski@swps.edu.pl', description=DESCRIPTION,
          long_description=open('README.md').read(),
          license='MIT', url='https://github.com/mmagnuski/borsar',
          download_url='https://github.com/mmagnuski/borsar',
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=['Intended Audience :: Science/Research',
                       'License :: OSI Approved :: MIT License',
                       'Programming Language :: Python',
                       'Topic :: Scientific/Engineering'],
          platforms='any',
          packages=find_packages()
          )