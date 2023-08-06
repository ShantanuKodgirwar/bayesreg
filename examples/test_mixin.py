import numpy as np
from scipy.optimize import rosen


class BaseClass:
    pass


class Mixin1(object):
    def test(self):
        print("Mixin1")


class Mixin2(object):
    def test(self):
        print("Mixin2")


class MyClass(BaseClass, Mixin1, Mixin2):
    pass


class Cost:
    @property
    def has_gradient(self):
        return hasattr(self, "gradient")


class GoodnessOfFit(Cost):
    pass


class LeastSquares(GoodnessOfFit):
    def gradient(self, params):
        r = self.residuals(params)
        X = self.model.design_matrix(self.cost.x)
        return X.T.dot(r)


class SumOfCosts(Cost):
    @property
    def has_gradient(self):
        return all([cost.has_gradient for cost in self])

    def __init__(self, *costs):
        self._costs = costs

    def __iter__(self):
        return iter(self._costs)


class Fitter:
    pass


class GradientDescent(Fitter):
    def __init__(self, cost):
        assert cost.has_gradient

        super().__init__(cost)


if __name__ == "__main__":
    import matplotlib.pylab as plt

    x = MyClass()
    cost = Cost()
    print(cost.has_gradient)
    lsq = LeastSquares()
    print(lsq.has_gradient)
    total = SumOfCosts(lsq, lsq)
    print("2*lsq:", total.has_gradient)
    total = SumOfCosts(lsq, cost)
    print("cost + lsq:", total.has_gradient)

    x = np.linspace(-1.0, 1.0, 100) * 0.1
    grid = np.reshape(np.meshgrid(x, x), (2, -1)).T

    vals = np.array(list(map(rosen, grid)))
    vals = vals.reshape((len(x), -1))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.contour(x, x, vals)
