from setuptools import setup, find_packages

setup(
    name='library',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'os',
        'numpy',
        're',
    ],
)
