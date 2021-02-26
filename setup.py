import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyGPA",
    version="0.0.1",
    author="Tobias A. de Jong",
    author_email="tobiasadejong@gmail.com",
    description="A Python package for Geometric Phases Analysis and related techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TAdeJong/pyGPA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires='>=3.6',
    install_requires=[
        "dask",
        "scipy",
        "numpy",
        "numba",
        "scikit-image",
        ],
)

