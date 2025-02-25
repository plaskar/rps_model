from setuptools import setup, find_packages

setup(
    name='rps_package',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
    ],
)
