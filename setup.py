#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    try:
        from ez_setup import use_setuptools
        use_setuptools()
        from setuptools import setup
    except Exception, e:
        print "Forget setuptools, trying distutils..."
        from distutils.core import setup

setup(
    name="deepnet",
    version="0.1.0",
    author="Eric Hunsberger",
    author_email="ehunsber@uwaterloo.ca",
    packages=['deepnet'],
    scripts=[],
    url="https://github.com/hunse/deepnet",
    license="LICENSE.rst",
    description="Tools for training deep neural networks",
    long_description=open('README.md').read(),
    requires=[],
    # test_suite='nengo.tests',
)
