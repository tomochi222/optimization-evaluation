# coding: utf-8
from setuptools import setup, find_packages
from optimization_evaluation import __author__, __version__, __license__

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
        name             = 'optimization_evaluation',
        version          = __version__,
        description      = 'A library to support the benchmarking of functions for optimization evaluation, similar to algorithm-test',
        long_description = long_description,
        license          = __license__,
        author           = __author__,
        author_email     = 'tomochika.keita.tz1@is.naist.jp',
        url              = 'https://github.com/keit0222/optimization-evaluation',

        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering',

            # Pick your license as you wish (should match "license" above)
            'License :: OSI Approved :: MIT License',

            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],

        # What does your project relate to?
        keywords='Optimization algorithm test',
        packages         = find_packages(),
        install_requires = [],
        )
