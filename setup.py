from setuptools import setup
from distutils.core import setup

setup(
    name="algotoolsshiridi",
    version="0.0.9",
    description="A one stop solution for solving your required algorithms",
    author_email="shiridikumarpeddinti836@gmail.com",
    py_modules=["algospackage"],
    package_dir={"":"src"},
    install_requires=["matplotlib","pandas","numpy","seaborn"]

)