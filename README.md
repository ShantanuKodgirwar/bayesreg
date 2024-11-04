# bayesreg

Just a repository for understanding the basics of Bayesian linear regression. Initially, we look at the classical linear regression applied to a polynomial fitting problem, which is followed by the Bayesian approach.

## Installing dependencies

Clone the repository and install the dependencies using the python package manager [poetry](https://github.com/python-poetry/poetry) as
```bash
git clone https://github.com/ShantanuKodgirwar/bayesreg.git
cd bayesreg
poetry install --with dev --sync
```
This installs the package in a virtual environment and can be accessed with `poetry shell`.

## Linear regression

Fitting a polynomial of degree $K$ to pairs $(x_n, y_n)$ such that the least-squares residual.

$$
\chi^2(\Theta) = \sum_{n=1}^N [y_n - f(x_n; \Theta)]^2
$$

is minimized. Here, $f(x; \Theta)$ is a polynomial of degree $K$, i.e.

$$
f(x; \Theta) = \sum_{k=0}^{K-1} \theta_k x^{k} = \theta_0 + \theta_1 x + \theta_2 x^2 + \ldots
$$

This problem can also be solved analytically using the [**Moore-Penrose inverse**](https://en.wikipedia.org/wiki/Moore-Penrose_inverse).

## Bayesian Linear regression
*TODO: Add details here*

## Further learning

* [**Regularizer**](https://en.wikipedia.org/wiki/Regularization_(mathematics)) to cope with [**overfitting**](https://en.wikipedia.org/wiki/Overfitting)

* [**Bayesian linear regression**](https://en.wikipedia.org/wiki/Bayesian_linear_regression)

* [**Markov chain Monte Carlo**](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) sampling to [**quantify the uncertainty**](https://en.wikipedia.org/wiki/Uncertainty_quantification) of model parameters

* [**Sparsity**](https://en.wikipedia.org/wiki/Compressed_sensing) prior, gradient-based priors

## Literature

Literature in order of increasing level of complexity:

* **Bishop**: Chapter 1 of "Pattern Recognition and Machine Learning" (a very good book!)

* Ribes: Background on applications in imaging

* Hansen: Deconvolution and regularization

* MacKay: Bayesian perspective

* Minka: Bayesian linear regression
