# setup.py
import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description. It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.abspath(os.path.dirname(__file__)), fname)).read()

# Get the list of requirements from requirements.txt (core dependencies only)
def get_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

setup(
    name="vibroml",
    version="0.0.1", # Start with a small version number
    author="Rogerio Gouvea",
    author_email="rogeriog.em@gmail.com",
    description=("A Python toolkit for ML-driven vibrational analysis of crystalline materials."),
    license="MIT", # Or your chosen license
    keywords="materials science, phonons, machine learning, AIMD, stability",
    url="https://github.com/rogeriog/VibroML", # Replace with your actual GitHub repo URL
    packages=find_packages(exclude=["tests", "docs"]), # Automatically find packages in the 'vibroml' directory
    long_description=read('README.md'),
    long_description_content_type='text/markdown', # Specify the content type for README.md
    install_requires=get_requirements(), # Use the function to get core requirements only
    extras_require={
        'mace': ['mace-torch==0.3.13', 'torch==2.7.1', 'e3nn==0.4.4'],
        'm3gnet': ['m3gnet==0.2.4', 'tensorflow==2.19.0'],
        'all': [
            'mace-torch==0.3.13',
            'torch==2.7.1',
            'e3nn==0.4.4',
            'm3gnet==0.2.4',
            'tensorflow==2.19.0',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha", # Change as your project matures
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Materials Science",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    # If you have a command-line entry point, define it here
    entry_points={
        'console_scripts': [
            'vibroml=vibroml.main:main', # Assuming vibroml/main.py has a main() function for CLI
        ],
    },
    python_requires='>=3.9', # Minimum Python version required
)