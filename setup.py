from setuptools import find_packages, setup
import subprocess

with open("README.md", "r") as fh:
    long_description = fh.read()

# get all the git tags from the cmd line that follow our versioning pattern
git_tags = subprocess.Popen(
    ["git", "tag", "--list", "v*[0-9]", "--sort=version:refname"],
    stdout=subprocess.PIPE,
)
tags = git_tags.stdout.read()
git_tags.stdout.close()
tags = tags.decode("utf-8").split("\n")
tags.sort()

# PEP 440 won't accept the v in front, so here we remove it, strip the new line and decode the byte stream
VERSION_FROM_GIT_TAG = tags[-1][1:]

setup(
    name="sig-decomp",
    version=VERSION_FROM_GIT_TAG,
    description="Optimzation-based signal decomposition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires=["setuptools>=18.0"],
    install_requires=[
        "cvxpy",
        "matplotlib", 
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "qss"
    ],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
