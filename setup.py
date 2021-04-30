import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="pyGPA",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
    python_requires='>=3.7',
    install_requires=requirements,
)

