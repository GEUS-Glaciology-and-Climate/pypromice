import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pypromice",
    version="1.5.1",
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
    python_requires=">=3.10",
    package_data={
    	"pypromice.tx": ["payload_formats.csv", "payload_types.csv"],
        "pypromice.qc.percentiles": ["thresholds.csv"],
        "pypromice.postprocess": ["positions_seed.csv"],
    },
    install_requires=['numpy~=1.23', 'pandas>=1.5.0', 'xarray>=2022.6.0', 'toml', 'scipy>=1.9.0', 'Bottleneck', 'netcdf4', 'pyDataverse==0.3.1', 'eccodes', 'scikit-learn>=1.1.0'],
#    extras_require={'postprocess': ['eccodes','scikit-learn>=1.1.0']},
    entry_points={
    'console_scripts': [
        'get_promice_data = pypromice.get.get_promice_data:get_promice_data',
        'get_l0tx = pypromice.tx.get_l0tx:get_l0tx',
        'join_l2 = pypromice.process.join_l2:main',
        'join_l3 = pypromice.process.join_l3:main',
        'get_l2 = pypromice.process.get_l2:main',
        'get_l2tol3 = pypromice.process.get_l2tol3:main',
        'make_metadata_csv = pypromice.postprocess.make_metadata_csv:main',
        'get_watsontx = pypromice.tx.get_watsontx:get_watsontx',
        'get_bufr = pypromice.postprocess.get_bufr:main',
        'create_bufr_files = pypromice.postprocess.create_bufr_files:main',
        'bufr_to_csv = pypromice.postprocess.bufr_to_csv:main',
        'get_msg = pypromice.tx.get_msg:get_msg'
    ],
},
)
