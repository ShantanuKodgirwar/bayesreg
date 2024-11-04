# bayesreg

Just a repository for understanding the basics of Bayesian linear regression by applying it to a simple polynomial fitting problem.

## Installing dependencies

Clone the repository and install the dependencies using the python package manager [poetry](https://github.com/python-poetry/poetry) as
```bash
git clone https://github.com/ShantanuKodgirwar/bayesreg.git
cd bayesreg
poetry install --with dev --sync
```
This installs the package in a virtual environment and can be accessed with `poetry shell`.

## Topics to understand

Implementations here mainly involve topics in Bayesian inference which are explained with some derivations under [here](documentation/bayesian_inference.pdf). The following topics must also be reviewed, 

* [**Markov chain Monte Carlo**](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) sampling to [**quantify the uncertainty**](https://en.wikipedia.org/wiki/Uncertainty_quantification) of model parameters
* [**Sparsity**](https://en.wikipedia.org/wiki/Compressed_sensing) prior, gradient-based priors

## Literature

The following literature is considered in order of the level of complexity, mainly focusing on imaging.

* Bishop: Chapter 1 of "Pattern Recognition and Machine Learning"
* Ribes: Background on applications in imaging
* Hansen: Deconvolution and regularization
* MacKay: Bayesian perspective
* Minka: Bayesian linear regression
