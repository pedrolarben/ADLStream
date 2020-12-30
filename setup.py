import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = "0.1.2"
PACKAGE_NAME = "ADLStream"
AUTHOR = "Pedro Lara-Benítez & Manuel Carranza-García"
AUTHOR_EMAIL = "plbenitez@us.es"
URL = "https://adlstream.readthedocs.io"

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
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    setup_requires=["pytest-runner"],
    tests_require=["pytest==4.4.1"],
)
