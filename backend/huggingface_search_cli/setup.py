#!/usr/bin/env python3
"""Quick install script for development."""

from setuptools import setup, find_packages

setup(
    name="huggingface-search-cli",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.5.0",
        "requests>=2.31.0",
        "urllib3>=2.0.0",
        "httpx>=0.27.0",
        "huggingface_hub>=0.20.0",
        "datasets>=2.16.0",
        "PyYAML>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "hf-search=huggingface_search_cli.cli.main:main",
        ],
    },
)
