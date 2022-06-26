from setuptools import find_packages, setup

base_requirements = []

dev_requirements = ["jupyterlab", "pandas-profiling[notebook]", "black"]

test_requirements = ["pytest", "pytest-cov", "hypothesis"]

setup(
    name="training_template",
    packages=find_packages("."),
    install_requires=base_requirements,
    extras_require={"dev": dev_requirements, "test": test_requirements},
)
