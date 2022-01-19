import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = "0.1.4"
PACKAGE_NAME = "ADLStream"
AUTHOR = "Pedro Lara-Benítez & Manuel Carranza-García"
AUTHOR_EMAIL = "plbenitez@us.es"
URL = "https://adlstream.readthedocs.io"
PROJECT_URLS = {
    "Download": "https://github.com/pedrolarben/ADLStream/tags",
    "API Documentation": "https://adlstream.readthedocs.io/API/overview/",
    "Paper": "https://doi.org/10.3233/ICA-200617",
}
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Environment :: GPU :: NVIDIA CUDA :: 11.2",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
LICENSE = "MIT"
DESCRIPTION = "ADLStream is a novel asynchronous dual-pipeline deep learning framework for data stream mining"
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
    "numpy",
    "tensorflow>=2.1.0",
    "tensorflow-addons>=0.11.0",
    "keras-tcn",
    "matplotlib",
    "scikit-learn",
    "kafka-python",
    "matplotlib",
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    license=LICENSE,
    author_email=AUTHOR_EMAIL,
    url=URL,
    project_urls=PROJECT_URLS,
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    setup_requires=["pytest-runner"],
    tests_require=["pytest>=4.4.1"],
)
