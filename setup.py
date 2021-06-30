# FIXME revise this file according to https://packaging.python.org/tutorials/packaging-projects/

import setuptools

setuptools.setup(
    name='ippgtoolbox',
    version='0.1',
    description='toolbox for imaging photoplethysmography',
    author='Matthieu Scherpf',
    author_email='Matthieu.Scherpf@tu-dresden.de',
    packages=setuptools.find_packages(
        exclude=[
            'detection/skin/deeplab',
        ],
    ),
    install_requires=[
        'opencv-python',
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'ipython',
    ],
    include_package_data=True,
)
