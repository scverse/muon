import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="muon-gtca",
    version="0.0.1",
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
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'h5py',
        'anndata',
        'scanpy',
        'loompy',
        'protobuf'
    ]
)
