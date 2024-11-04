"""
Numerical computation of derivatives (gradient and Hessian matrix).
"""
import matplotlib.pylab as plt
import numpy as np
import scipy.optimize as opt

import bayesreg as reg


class TestFunction:
    def __init__(self, phases):
        self.phases = np.array(phases)

    @property
    def dim(self):
        return len(self.phases)

    def __call__(self, x):
        assert self.dim == len(x)
        return np.linalg.norm(np.sin(x + self.phases))

    def __iadd__(self, other):
        assert self.dim == other.dim
        return self.phases + other.phases

    def gradient(self, x):
        assert self.dim == len(x)
        return 0.5 * np.sin(2 * (x + self.phases)) / (self(x) + 1e-10)

    def hessian(self, x):
        assert self.dim == len(x)
        g = self.gradient(x)
        return (
            -np.multiply.outer(g, g) + np.eye(self.dim) * np.cos(2 * (x + self.phases))
        ) / (self(x) + 1e-10)


def test_derivatives(d=2):
    x = np.random.random(d)

    print(
        "checking gradient:",
        np.corrcoef(reg.calc_gradient(func, x), func.gradient(x))[0, 1],
        np.fabs(reg.calc_gradient(func, x) - func.gradient(x)).max(),
    )

    H_true = func.hessian(x).flatten()
    H_num = reg.calc_hessian(func, x).flatten()

    print(
        "checking the Hessian:",
        np.corrcoef(H_true, H_num)[0, 1],
        np.fabs(H_true - H_num).max(),
    )


if __name__ == "__main__":
    if False:
        # needs to be checked some other time
        func = TestFunction(np.random.random(3))
        func2 = TestFunction(np.random.random(func.dim))

        print(func.phases)
        print(func2.phases)
        func += func2
        print(func.phases)

    if not False:
        func = TestFunction(np.zeros(2))
        x = np.linspace(-1.0, 1.0, 200) * 1e-9
        grid = np.array(np.meshgrid(x, x))
        grid = np.transpose(grid.reshape(2, -1))

        d = 2
        func = TestFunction(0 * np.random.random(d))

        x_opt = opt.fmin_bfgs(func, np.random.random(func.dim), fprime=func.gradient)
        x_opt2 = np.pi * np.random.randint(-3, 3, size=func.dim) - func.phases
        if False:
            x_opt2 = np.zeros(func.dim)

        print("scipy.opt.fmin")
        print("function value", func(x_opt))
        print("norm of gradient", np.linalg.norm(func.gradient(x_opt)))
        print("eigvals of hessian", np.linalg.eigvalsh(func.hessian(x_opt)))

        print("true minimum")
        print("function value", func(x_opt2))
        print("norm of gradient", np.linalg.norm(func.gradient(x_opt2)))
        print("eigvals of hessian", np.linalg.eigvalsh(func.hessian(x_opt2)))

    if not False:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        if not False:
            ax.contour(x, x, np.reshape(list(map(func, grid)), (len(x), -1)))
        else:
            ax.imshow(
                np.reshape(list(map(func, grid)), (len(x), -1)),
                extent=(x.min(), x.max(), x.min(), x.max()),
            )
        ax.scatter(*x_opt, color="r", s=200, alpha=0.75, label="scipy")
        ax.scatter(*x_opt2, color="orange", s=400, alpha=0.75, label="truth")

        ax.legend()
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(x.min(), x.max())
        fig.tight_layout()
        plt.show()
