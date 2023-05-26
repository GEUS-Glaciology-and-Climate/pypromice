import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pypromice",
    version="1.2.1",
    author="GEUS Glaciology and Climate",
    description="PROMICE/GC-Net data processing toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GEUS-Glaciology-and-Climate/pypromice",
    project_urls={
        "Bug Tracker": "https://github.com/GEUS-Glaciology-and-Climate/pypromice/issues",
        "Documentation": "https://pypromice.readthedocs.io",
    "Source Code": "https://github.com/GEUS-Glaciology-and-Climate/pypromice"
    },
    keywords="promice gc-net aws climate glaciology greenland geus",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    include_package_data = True,
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=['numpy>=1.23.0', 'pandas>=1.5.0', 'xarray>=2022.6.0', 'toml', 'scipy>=1.9.0', 'scikit-learn>=1.1.0', 'Bottleneck', 'netcdf4', 'pyDataverse'],
    scripts=['bin/getData', 'bin/getL0tx', 'bin/getL3', 'bin/joinL3', 'bin/getWatsontx', 'bin/getBUFR', 'bin/getMsg'],
)
