#!/usr/bin/env python

# SkyTropD installation script
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="skytropd",
    version="2.13.0",
    description="Community-maintained tropical width metrics package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="LGPL-3.0-only",
    author="Qianye Su",
    author_email="suqianye2000@gmail.com",
    maintainer="Qianye Su",
    maintainer_email="suqianye2000@gmail.com",
    requires=["numpy", "matplotlib", "scipy"],
    install_requires=["numpy>=1.19", "scipy>=1.5"],
    python_requires=">=3.8",
    packages=find_packages(),
    include_package_data=True,
    license_files=("LICENSE",),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    package_data={
        "skytropd/ValidationData": ["*.nc"],
        "skytropd/ValidationMetrics": ["*.nc"],
    },
)
