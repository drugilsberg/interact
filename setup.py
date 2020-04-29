"""Install package."""
import os
from setuptools import setup, find_packages

REQUIREMENTS = []
if os.path.exists('requirements.txt'):
    for line in open('requirements.txt'):
        REQUIREMENTS.append(line.strip())

scripts = []

setup(
    name='interact',
    version='0.1',
    description=(
        'Interaction Network InfErence from VectoR RepresentATion of Words'
    ),
    long_description=open('README.md').read(),
    author='Matteo Manica, Roland Mathis, Joris Cadow',
    author_email='tte@zurich.ibm.com, lth@zurich.ibm.com, dow@zurich.ibm.com',
    packages=find_packages('.'),
    install_requires=REQUIREMENTS,
    zip_safe=False,
    scripts=scripts
)
