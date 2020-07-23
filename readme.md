Our focus is on solving [**inverse problems**](https://en.wikipedia.org/wiki/Inverse_problem) using Bayesian methods. To get started, we will first look at a problem in 

# Linear regression

Fitting a polynomial of degree $K$ to pairs $(x_n, y_n)$ such that the least-squares residual 
$$
\chi^2(w) = \sum_{n=1}^N [y_n - \Phi(x_n; w)]^2
$$
is minimized. Here, $\Phi(x; w)$ is a polynomial of degree $K$, i.e.
$$
\Phi(x; w) = \sum_{k=0}^{K-1} w_k x^{k} = w_0 + w_1 x + w_2 x^2 + \ldots
$$
This problem can be solved analytically using the [**Moore-Penrose inverse**](https://en.wikipedia.org/wiki/Moore-Penrose_inverse).

# Python code for polynomial fitting

The idea is to develop an object-oriented fitter for [**polynomial regression**](https://en.wikipedia.org/wiki/Polynomial_regression). Naturally, the classes that we would need to implement in order to solve this problem are:

* LinearModel: An abstract parent class that implements a [**linear model**](https://en.wikipedia.org/wiki/Linear_model) typically using some basis functions such as polynomials or trigonometric functions

* Cost: A cost function that is minimized to obtain the best fitting model. A typical cost function is the sum of squares of deviations between data and model predictions used in [**least squares**](https://en.wikipedia.org/wiki/Least_squares) fitting

Based on the example code, please start to implement a model and fitting procedure for polynomials. Instead of using a grid search (as shown in "straightline_demo.py") it is much more efficient to use Linear Algebra to do the parameter fitting. 

# Future ideas

* [**Regularizer**](https://en.wikipedia.org/wiki/Regularization_(mathematics)) to cope with [**overfitting**](https://en.wikipedia.org/wiki/Overfitting)

* [**Bayesian linear regression**](https://en.wikipedia.org/wiki/Bayesian_linear_regression)

* [**Markov chain Monte Carlo**](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) sampling to [**quantify the uncertainty**](https://en.wikipedia.org/wiki/Uncertainty_quantification) of model parameters

* [**Sparsity**](https://en.wikipedia.org/wiki/Compressed_sensing) prior, gradient-based priors

# Applications

* Linear regression

* Linear inverse problems such as those arising in imaging

# Literature

In order of increasing levels of sophistication... but I would rather start with the wikipedia pages linked above and the book by Bishop.

* **Bishop**: Chapter 1 of "Pattern Recognition and Machine Learning" (a very good book!) Many PDF copies circulate the web. One is **[here](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)** for example.

* Ribes: Background on applications in imaging

* Hansen: Deconvolution and regularization

* MacKay: Bayesian perspective

* Minka: Bayesian linear regression
