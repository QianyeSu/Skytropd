#!/usr/bin/env python

# SkyTropD installation script
from os import path
from pathlib import Path
import shutil

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py

this_directory = path.abspath(path.dirname(__file__))
project_root = Path(this_directory)
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


class build_py(_build_py):
    """Copy validation resources into the packaged wheel."""

    package_name = "skytropd"
    resource_dirs = ("ValidationData", "ValidationMetrics")

    def run(self):
        super().run()
        build_package_dir = Path(self.build_lib) / self.package_name
        for resource_dir in self.resource_dirs:
            src = project_root / resource_dir
            dst = build_package_dir / resource_dir
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


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
    python_requires=">=3.9",
    packages=find_packages(),
    include_package_data=True,
    license_files=("LICENSE",),
    cmdclass={"build_py": build_py},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    package_data={
        "skytropd": [
            "ValidationData/*.nc",
            "ValidationMetrics/*.nc",
            "ValidationMetrics/figs/*.png",
        ]
    },
)