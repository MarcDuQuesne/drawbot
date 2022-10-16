#!/usr/bin/env python
from setuptools import setup, find_packages


def requirements(reqs_file="requirements.txt"):
    with open(reqs_file) as f:
        required = f.read().splitlines()
    return [line for line in required if not line.startswith("#")]


setup(
    name="crayion",
    author="Matteo Giani",
    author_email="matteo.giani.87@gmail.com",
    version="0.0.1",
    install_requires=requirements("requirements.txt"),
    tests_require=requirements("requirements_tests.txt"),
    extras_require={
        "dev": requirements("requirements_dev.txt"),
        "test": requirements("requirements_tests.txt"),
    },
    packages=find_packages(exclude=["notebooks", "tests"]),
    entry_points={
        "console_scripts": [
            "crayion = craiyon.craiyon_api:call_api",
        ],
    },
    package_data={"craiyon": ["images/*"]},
)
