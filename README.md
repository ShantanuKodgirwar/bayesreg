The simplest problem to look into with linear regression is polynomial fitting. We look at this from the classical point of view followed by the Bayesian perspective.

## Linear regression

Fitting a polynomial of degree $K$ to pairs $(x_n, y_n)$ such that the least-squares residual.

$$
\chi^2(w) = \sum_{n=1}^N [y_n - \Phi(x_n; w)]^2
$$

is minimized. Here, $\Phi(x; w)$ is a polynomial of degree $K$, i.e.

$$
\Phi(x; w) = \sum_{k=0}^{K-1} w_k x^{k} = w_0 + w_1 x + w_2 x^2 + \ldots
$$

This problem can also be solved analytically using the [**Moore-Penrose inverse**](https://en.wikipedia.org/wiki/Moore-Penrose_inverse).

## Bayesian Linear regression
*TODO: Add details here*

## Learning

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
