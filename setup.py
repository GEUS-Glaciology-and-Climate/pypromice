import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pypromice",
    version="1.3.1",
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
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
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
    entry_points={
    'console_scripts': [
        'get_promice_data = pypromice.get.get_promice_data:get_promice_data',
        'get_l0tx = pypromice.tx.get_l0tx:get_l0tx',
        'get_l3 = pypromice.process.get_l3:get_l3',
        'join_l3 = pypromice.process.join_l3:join_l3',
        'get_watsontx = pypromice.tx.get_watsontx:get_watsontx',
        'get_bufr = pypromice.postprocess.get_bufr:get_bufr',
        'get_msg = pypromice.tx.get_msg:get_msg'
    ],
},
)
