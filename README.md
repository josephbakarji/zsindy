## Description

PyZSindy is an open-source tool for the identification of differential equations from data. It builds on the Sparse identification of nonlinear dynamics (SINDy) and uses a Bayesian approach, from a statistical mechanics perspective, to quantify model uncertainty and analyze algorithmic phase transitions in the low data and sparsity penalty limits. For more information, refer to the paper [Statistical Mechanics of Dynamical System Identification](https://arxiv.org/abs/2403.01723). 

## Installation

### Installation for developers
If you're developing this code, create a virtual environment using `pipenv`, which allows you to edit the code while using it. If you don't have pipenv, install it with `pip install pipenv`. Then run

`pipenv install -e .` 

in the main directory (where there is a `setup.py` file). This creates a Pipfile which manages your dependencies, and the flag `-e` makes it editable, unlike a normal `pip` package (this might take a while). To activate the project's virtualenv, run `pipenv shell`.

Use `pipreqs ./` in the main directory if you want to update the requirements file (you might need to `--force` if it already exists).


## Usage

