from setuptools import setup, find_packages

setup(
    name="forge-ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "typer",
        "mlflow",
        "PyYAML",
        "torch",
        "fastapi",
        "uvicorn",
        "dvc",
        "pytest",
    ],
    entry_points={
        "console_scripts": [
            "mlfactory=cli.main:app",
        ],
    },
)
