#!/usr/bin/env python3
"""
Setup script for PerfCount cross-platform prediction system.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "PerfCount: Cross-platform performance prediction system"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements-core.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="perfcount",
    version="0.1.0",
    description="Cross-platform performance prediction system using machine learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="PerfCount Team",
    author_email="perfcount@example.com",
    url="https://github.com/your-org/perfcount",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Benchmark",
        "Topic :: System :: Performance",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "advanced": [
            "xgboost>=1.6.0",
            "catboost>=1.1.0", 
            "lightgbm>=3.3.0",
            "torch>=1.12.0",
            "torchvision>=0.13.0",
            "seaborn>=0.11.0",
            "statsmodels>=0.13.0",
            "psutil>=5.9.0",
            "memory-profiler>=0.60.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
            "pyyaml>=6.0",
            "configparser>=5.3.0",
            "mypy>=0.991",
            "types-requests>=2.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "perfcount=cross_platform_prediction.cross_platform:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
