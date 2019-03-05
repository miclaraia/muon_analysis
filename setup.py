"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the version string from the VERSION file
with open(path.join(here, 'VERSION'), 'r') as f:
    version = f.readline().strip()

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='muon',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version,
    description='Muon hunters machine analysis tools',
    long_description=long_description,
    author='Michael Laraia',
    author_email='larai002@umn.edu',
    license='MIT',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'matplotlib',
        'numpy',
        'pylint',
        'sklearn',
        'scipy',
        'panoptes_client==1.0.3',
        'scikit-image',
        'pandas',
        'click',
        'tqdm',
        'astropy',
        'pillow',
        'sqlite3',
    ],
    extras_require={
        'clustering': ['dec-keras', 'redec-keras']
    },
    entry_points={
        'console_scripts': [
            'muon=muon.__main__:main',
        ],
    },
)
