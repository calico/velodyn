import sys
if sys.version_info < (3, 6,):
    sys.exit('velodyn requires Python >= 3.6')
from pathlib import Path

from setuptools import setup, find_packages

try:
    from velodyn import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ''


long_description = '''
RNA velocity infers a rate of change for each transcript in an RNA-sequencing experiment based on the ratio of intronic to exonic reads. This inferred velocity vectors serves as a prediction for the future transcriptional state of a cell, while the current read counts serve as a measurement of the instantaneous state. Qualitative analysis of RNA velocity has been used to establish the order of gene expression states in a sequence, but quantitative analysis has generally been lacking.\n
\n
velodyn adopts formalisms from dynamical systems to provide a quantitative framework for RNA velocity analysis. The tools provided by velodyn along with their associated usage are described below. All velodyn tools are designed to integrate with the scanpy ecosystem and anndata structures.\n
\n
We have released velodyn in association with a recent pre-print. Please cite our pre-print if you find velodyn useful for your work.\n
\n
Differentiation reveals the plasticity of age-related change in murine muscle progenitors\n
Jacob C Kimmel, David G Hendrickson, David R Kelley\n
bioRxiv 2020.03.05.979112; doi: https://doi.org/10.1101/2020.03.05.979112
'''

setup(
    name='velodyn',
    version='0.1.0',
    description='Dynamical systems approaches for RNA velocity analysis',
    long_description=long_description,
    url='http://github.com/calico/velodyn',
    author=__author__,
    author_email=__email__,
    license='Apache',
    python_requires='>=3.6',
    install_requires=[
        l.strip() for l in
        Path('requirements.txt').read_text('utf-8').splitlines()
    ],
    packages=find_packages(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
