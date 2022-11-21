import os
import re

import setuptools

# for simplicity we actually store the version in the __version__ attribute in the source
here = os.path.realpath(os.path.dirname(__file__))
print(here)
with open(os.path.join(here, 'fastDP', '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

with open(os.path.join(here, 'README.md')) as f:
    readme = f.read()

setuptools.setup(
    name="fastDP",
    version=version,
    author="Zhiqi Bu",
    author_email="woodyx218@gmail.com",
    description="Efficient implementation of differentially private optimization for large-scale deep learning.",
    long_description=readme,
    url="",
    packages=setuptools.find_packages(exclude=['examples', 'tests']),
    python_requires='~=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
