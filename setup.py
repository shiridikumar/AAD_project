from setuptools import setup
from distutils.core import setup

setup(
    name="skpalgotools",
    version="0.0.7",
    description="A one stop solution for solving your required algorithms",
    author_email="shiridikumarpeddinti836@gmail.com",
    py_modules=["skpalgotools"],
    package_dir={"":"src"},
    install_requires=["matplotlib","pandas","numpy","seaborn"]

)