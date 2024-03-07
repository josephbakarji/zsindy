from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="PyZSindy",
    version="0.1.0",
    author="Joseph Bakarji and Andrei Klishin",
    author_email="joseph.bakarji0@aub.edu.lb",
    description="A Bayesian statistical mechanical approach to sparse equation discovery",
    packages=find_packages(),
    install_requires=requirements,
    # entry_points={
    #     "console_scripts": [
    #         "my_script=my_package.scripts.my_script:main",
    #     ],
    # },
)
