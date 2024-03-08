## Description

ZSindy is an open-source tool for the identification of differential equations from data. It builds on the Sparse identification of nonlinear dynamics (SINDy) and uses a Bayesian approach, from a statistical mechanics perspective, to quantify model uncertainty and analyze algorithmic phase transitions in the low data and sparsity penalty limits. For more information, refer to the paper [Statistical Mechanics of Dynamical System Identification](https://arxiv.org/abs/2403.01723). 

## Installation

You can install the package using pip:

`pip install zsindy`

## Quickstart

Start with importing the package and defining the parameters:

```python

import numpy as np
import matplotlib.pyplot as plt
from zsindy.dynamical_models import DynamicalSystem, lorenz
from zsindy.ml_module import ZSindy

# ZSindy parameters
poly_degree = 2
lmbda = 1e5 # Regularization parameter
zsindy_num_terms = 4 # Maximum number of terms to include in the model
eta = 0.2 # Noise

# Simulate the dynamical system
x0 = (0, 1, 1.05)
tend = 20
dt = 0.01
rho = dt/eta*np.sqrt(2) # Resolution parameter
args = (10, 28, 8/3)

```

ZSindy has a few built-in dynamical systems, such as the Lorenz system. You can also define your own dynamical system. 

```python


# Built-in simulators for dynamical systems
dyn = DynamicalSystem(lorenz, poly_degree, args, num_variables=len(x0))
X = dyn.solve(x0, dt, tend)
true_coefs = dyn.true_coefficients
varnames = dyn.varnames
lambda_norm = dyn.time/dt * rho**2
num_dims = X.shape[1]

# Add Noise
X += eta * np.random.randn(*X.shape)

```

Now, we can use ZSindy to identify the dynamical system from the data. 

```python

## Z-Sindy
zmodel = ZSindy(rho=rho, 
                lmbda=lmbda, 
                max_num_terms=zsindy_num_terms, 
                poly_degree=poly_degree,
                variable_names=varnames)

zmodel.fit(X, dyn.time)

zmodel.print()
z_x_pred = zmodel.simulate(x0, dyn.time)
z_xdot_pred = zmodel.predict()

```

    ( x )' =  + -9.8881 x + 9.9020 y
    ( y )' =  + 27.4575 x + -0.8720 y + -0.9853 x*z
    ( z )' =  + -2.6625 z + 0.9982 x*y

The `print` method provides a summary of the identified model. 

### Installation for developers
If you're developing this code, create a virtual environment using `pipenv`, which allows you to edit the code while using it. If you don't have pipenv, install it with `pip install pipenv`. Then run

`pipenv install -e .` 

in the main directory (where there is a `setup.py` file). This creates a Pipfile which manages your dependencies, and the flag `-e` makes it editable, unlike a normal `pip` package (this installation might take a while). To activate the project's virtualenv, run `pipenv shell`.

Use `pipreqs ./` in the main directory if you want to update the requirements file (you might need to `--force` if it already exists).


## Usage

