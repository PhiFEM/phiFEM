import numpy as np

"""
=============================
 Flower smooth levelset data 
=============================
"""


def _atan_r(x, radius=1.0, slope=1.0):
    r = np.sqrt(np.square(x[0, :]) + np.square(x[1, :]))
    r0 = np.full_like(r, radius)
    val = np.arctan(slope * (r - r0))
    return val


# Implementation of a graded smooth-min function inspired from: https://iquilezles.org/articles/smin/
def _smin(x, y_1, y_2, kmin=0.0, kmax=1.0):
    k = kmax * ((np.pi / 2.0 - _atan_r(x, radius=2.0, slope=50.0)) / np.pi / 2.0) + kmin
    return np.maximum(k, np.minimum(y_1, y_2)) - np.linalg.norm(
        np.maximum(np.vstack([k, k]) - np.vstack([y_1, y_2]), 0.0), axis=0
    )


# Levelset and RHS expressions taken from: https://academic.oup.com/imajna/article-abstract/42/1/333/6041856?redirectedFrom=fulltext
# This smooth definition of the levelset is used in the phiFEM formulation only (and not during the mesh tagging).
def levelset(x):
    def phi0(x):
        r = np.full_like(x[0, :], 2.0)
        return np.square(x[0, :]) + np.square(x[1, :]) - np.square(r)

    val = phi0(x)

    for i in range(1, 9):
        xi = 2.0 * (np.cos(np.pi / 8.0) + np.sin(np.pi / 8.0)) * np.cos(i * np.pi / 4.0)
        yi = 2.0 * (np.cos(np.pi / 8.0) + np.sin(np.pi / 8.0)) * np.sin(i * np.pi / 4.0)
        ri = (
            np.sqrt(2.0)
            * 2.0
            * (np.sin(np.pi / 8.0) + np.cos(np.pi / 8.0))
            * np.sin(np.pi / 8.0)
        )

        def phi_i(x):
            return (
                np.square(x[0, :] - np.full_like(x[0, :], xi))
                + np.square(x[1, :] - np.full_like(x[1, :], yi))
                - np.square(np.full_like(x[0, :], ri))
            )

        # val *= phi_i(x)
        val = _smin(x, val, phi_i(x))
    return val


# A non-smooth detection expression of the levelset is used only for mesh tagging purposes in order to avoid possible non connected sets to be selected if the smooth expression was used.
def detection_levelset(x):
    def phi0(x):
        r = np.full_like(x[0, :], 2.0)
        return np.square(x[0, :]) + np.square(x[1, :]) - np.square(r)

    val = phi0(x)

    for i in range(1, 9):
        xi = 2.0 * (np.cos(np.pi / 8.0) + np.sin(np.pi / 8.0)) * np.cos(i * np.pi / 4.0)
        yi = 2.0 * (np.cos(np.pi / 8.0) + np.sin(np.pi / 8.0)) * np.sin(i * np.pi / 4.0)
        ri = (
            np.sqrt(2.0)
            * 2.0
            * (np.sin(np.pi / 8.0) + np.cos(np.pi / 8.0))
            * np.sin(np.pi / 8.0)
        )

        def phi_i(x):
            return (
                np.square(x[0, :] - np.full_like(x[0, :], xi))
                + np.square(x[1, :] - np.full_like(x[1, :], yi))
                - np.square(np.full_like(x[0, :], ri))
            )

        val = np.minimum(val, phi_i(x))
    return val


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


# Not necessary (added here for the sake of the demo)
def dirichlet_data(x):
    return np.zeros_like(x[0, :])
