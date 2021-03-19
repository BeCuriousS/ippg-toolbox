# FIXME revise this file according to https://packaging.python.org/tutorials/packaging-projects/

import setuptools

setuptools.setup(
    name='ippgtoolbox',
    version='0.1',
    description='toolbox for imaging photoplethysmography',
    author='Matthieu Scherpf',
    author_email='Matthieu.Scherpf@tu-dresden.de',
    packages=[
        'ippgtoolbox/benchmark',
        'ippgtoolbox/detection',
        'ippgtoolbox/processing',
        'ippgtoolbox/utils',
    ],
    install_requires=[
        'opencv-python',
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'ipython',
    ])
