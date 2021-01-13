import setuptools
import os

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="muon",
    version="0.1.0",
    author="Danila Bredikhin",
    author_email="danila.bredikhin@embl.de",
    description="Multimodal omics analysis framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gtca/muon",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "h5py",
        "anndata",
        "scanpy",
        "sklearn",
        "umap-learn",
        "numba",
        "loompy",
        "protobuf",
        "tqdm",
    ],
    include_package_data=True,
)
