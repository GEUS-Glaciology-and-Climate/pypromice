import setuptools

with open("README.org", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="promice-aws-processing",
    version="0.0.1",
    author="Kenneth D. Mankoff",
    author_email="kdm@geus.dk",
    description="PROMICE AWS processing toolbox",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-processing",
    project_urls={
        "Bug Tracker": "https://github.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-processing/issues",
        "Documentation": "https://github.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-processing",
    "Source Code": "https://github.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-processing"
    },
    keywords="promice aws climate glaciology greenland",
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
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=['numpy', 'pandas', 'xarray', 'toml'],
    scripts=['bin/promiceAWS'],
)
