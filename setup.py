# -*- coding: utf-8 -*-
"""Setup module."""
import os

from setuptools import find_packages, setup


def read(filename: str) -> str:
    """Return file as a string."""
    with open(os.path.join(os.path.dirname(__file__), filename)) as file:
        return file.read()


def parse_requirements() -> tuple:
    """Parse requirements.txt for install_requires."""
    requirements = read('requirements.txt')
    return tuple(requirements.split('\n'))


setup(
    name='multi_label_classifier',
    packages=find_packages(exclude=('tests',)),
    python_requires='~=3.7',
    include_package_data=True,
    install_requires=parse_requirements(),
)
