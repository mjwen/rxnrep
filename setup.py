from pathlib import Path

from setuptools import find_packages, setup


def get_version(fname=Path.cwd().joinpath("rxnrep", "__init__.py")):
    with open(fname) as fin:
        for line in fin:
            line = line.strip()
            if "__version__" in line:
                v = line.split("=")[1]
                if "'" in v:
                    version = v.strip("' ")
                elif '"' in v:
                    version = v.strip('" ')
                break
    return version


setup(
    name="rxnrep",
    version=get_version(),
    packages=find_packages(),
    entry_points={"console_scripts": ["bondnet = predict:cli"]},
    install_requires=[
        "numpy",
        "scikit-learn",
        "requests",
        "pyyaml",
        "pytorch-lightning",
        "pandas",
    ],
    author="Mingjian Wen",
    author_email="wenxx151@gmail.com",
    url="https://github.com/mjwen/rxnrep",
    description="short description",
    long_description="long description",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Common Development and Distribution License 1.0 (CDDL-1.0)",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
