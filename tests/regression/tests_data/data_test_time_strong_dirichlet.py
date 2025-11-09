import numpy as np
import ufl

delta = 1.0e-20


# Triangle function approximation
def _trg(x):
    return 1.0 - 2.0 * ufl.acos((1 - delta) * ufl.sin(2.0 * np.pi * x)) / np.pi


# Square function approximation
def _sqr(x):
    return 2.0 * ufl.atan(ufl.sin(2.0 * np.pi * x) / delta) / np.pi


# Sawtooth function approximation
def _swt(x):
    return 1.0 + _trg((2.0 * x - 1.0) / 4.0) * _sqr(x / 2.0) / 2.0


# Levelset and RHS expressions taken from: https://academic.oup.com/imajna/article-abstract/42/1/333/6041856?redirectedFrom=fulltext
def generate_levelset(mode):
    if mode.__name__ == "numpy":

        def round(x):
            return np.round(x)
    elif mode.__name__ == "ufl":

        def round(x):
            return (x - 0.5) - _swt(x - 0.5) + 1.5

    def levelset(x):
        r = mode.sqrt(x[0] ** 2 + x[1] ** 2)
        R = 2.0
        theta = mode.atan2(x[1], x[0])

        sigma = 10.0

        num_petals = 10.0
        radius = 0.5
        val = (
            r
            * (
                1.0
                - radius
                * (
                    mode.sqrt(
                        radius**2
                        - (
                            num_petals * theta / 2.0 / np.pi
                            - round(num_petals * theta / 2.0 / np.pi)
                        )
                        ** 2
                    )
                )
            )
            - R
        )

        val = mode.exp(-(((r - R) / sigma) ** 2)) * val + (
            1.0 - mode.exp(-(((r - R) / sigma) ** 2))
        ) * (r - R)

        return val

    return levelset


def source_term(x):
    x1 = 2.0 * (np.cos(np.pi / 8.0) + np.sin(np.pi / 8.0))
    y1 = 0.0
    r1 = (
        np.sqrt(2.0)
        * 2.0
        * (np.sin(np.pi / 8.0) + np.cos(np.pi / 8.0))
        * np.sin(np.pi / 8.0)
    )

    val = np.square(x[0, :] - np.full_like(x[0, :], x1)) + np.square(
        x[1, :] - np.full_like(x[1, :], y1)
    )

    return np.where(val <= np.square(r1) / 2.0, 10.0, 0.0)
