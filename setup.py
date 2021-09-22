from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name = "fgvc_cl",
    version = "0.0.1",
    author = "Fangrui Liu",
    author_email = "fangrui.liu@outlook.com",
    license = "MIT",
    install_requires = requirements,
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    tests_require=['pytest'],
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
)