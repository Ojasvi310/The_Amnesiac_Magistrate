"""Package setup for Continual Counsel.

Using setup.py rather than pyproject.toml to stay compatible with the range of
Python 3.11 environments that Colab may ship (some older Colab images don't have
pip >= 21.3 which is required for pyproject.toml editable installs).
"""
from setuptools import setup, find_packages

setup(
    name="continual-counsel",
    version="0.1.0",
    packages=find_packages(where="src") + find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.11",
    description="Continual-learning compliance assistant with O-LoRA + TIES-merge",
)
