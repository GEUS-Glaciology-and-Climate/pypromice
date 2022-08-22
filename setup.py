import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pypromice",
    version="0.0.1",
    author="Penelope How",
    author_email="pho@geus.dk",
    description="PROMICE data processing toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-processing",
    project_urls={
        "Bug Tracker": "https://github.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-processing/issues",
        "Documentation": "https://github.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-processing",
    "Source Code": "https://github.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-processing"
    },
    keywords="promice gc-net aws climate glaciology greenland geus",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    include_package_data = True,
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=['numpy', 'pandas', 'xarray', 'toml'],
    scripts=['bin/getData', 'bin/getL0tx', 'bin/getL3'],
)
