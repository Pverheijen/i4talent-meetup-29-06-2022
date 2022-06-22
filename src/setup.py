from setuptools import setup, find_packages


base_requirements = []

dev_requirements = ["jupyterlab", "pandas-profiling[notebook]", "black"]

test_requirements = ["pytest", "pytest-cov", "hypothesis"]

setup(
    name="iris_classifier",
    packages=find_packages("."),
    install_requires=base_requirements,
    extras_require={"dev": dev_requirements, "test": test_requirements},
)
