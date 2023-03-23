from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="sig-decomp",
    version="0.0.2",
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
