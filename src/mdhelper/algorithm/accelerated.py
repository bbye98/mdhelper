"""
Numba algorithms
================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains miscellaneous Numba-accelerated algorithms.
"""

import numba
import numpy as np

@numba.njit(fastmath=True)
def dot_1d_1d(a: np.ndarray[float], b: np.ndarray[float]) -> float:

    r"""
    Serial Numba-accelerated dot product between two one-dimensional
    NumPy arrays :math:`\mathbf{a}` and :math:`\mathbf{b}`, each with
    shape :math:`(3,)`.

    .. math::

       \mathbf{a}\cdot\mathbf{b}=a_1b_1+a_2b_2+a_3b_3

    Parameters
    ----------
    a : `np.ndarray`
        First vector :math:`\mathbf{a}`.

        **Shape**: :math:`(3,)`.

    b : `np.ndarray`
        Second vector :math:`\mathbf{b}`.

        **Shape**: :math:`(3,)`.

    Returns
    -------
    ab : `float`
        Dot product of the two vectors, 
        :math:`\mathbf{a}\cdot\mathbf{b}`.
    """
    
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@numba.njit(fastmath=True)
def delta_fourier_transform_1d_1d(
        q: np.ndarray[float], r: np.ndarray[float]) -> complex:

    r"""
    Serial Numba-accelerated Fourier transform of a Dirac delta 
    function involving two one-dimensional NumPy arrays 
    :math:`\mathbf{q}` and :math:`\mathbf{r}`, each with shape 
    :math:`(3,)`.

    .. math::

       \mathcal{F}[\delta(\mathbf{q}-\mathbf{r})]
       =\exp(i\mathbf{q}\cdot\mathbf{r})

    Parameters
    ----------
    q : `np.ndarray`
        First vector :math:`\mathbf{q}`.

        **Shape**: :math:`(3,)`.

    r : `np.ndarray`
        Second vector :math:`\mathbf{r}`.

        **Shape**: :math:`(3,)`.

    Returns
    -------
    F : `complex`
        Fourier transforms of the Dirac delta functions, 
        :math:`\mathcal{F}[\delta(\mathbf{q}-\mathbf{r})]`.
    """

    return np.exp(1j * dot_1d_1d(q, r))

@numba.njit(fastmath=True)
def delta_fourier_transform_2d_2d(
        qs: np.ndarray[float], rs: np.ndarray[float]) -> np.ndarray[complex]:

    r"""
    Serial Numba-accelerated Fourier transforms of Dirac delta 
    functions involving all possible combinations of multiple 
    one-dimensional NumPy arrays :math:`\mathbf{q}` and 
    :math:`\mathbf{r}`, each with shape :math:`(3,)`.

    .. math::

       \mathcal{F}[\delta(\mathbf{q}-\mathbf{r})]
       =\exp(i\mathbf{q}\cdot\mathbf{r})

    Parameters
    ----------
    qs : `np.ndarray`
        Multiple vectors :math:`\mathbf{q}`.

        **Shape**: :math:`(N_q,\,3)`.

    rs : `np.ndarray`
        Multiple vectors :math:`\mathbf{r}`.

        **Shape**: :math:`(N_r,\,3,)`.

    Returns
    -------
    F : `np.ndarray`
        Fourier transforms of the Dirac delta functions, 
        :math:`\mathcal{F}[\delta(\mathbf{q}-\mathbf{r})]`.

        **Shape**: :math:`(N_q,\,N_r)`.
    """

    F = np.empty(qs.shape[0], dtype=np.complex128)
    for i in range(qs.shape[0]):
        F[i] = 0.0j
        for j in range(rs.shape[0]):
            F[i] += delta_fourier_transform_1d_1d(qs[i], rs[j])
    return F

@numba.njit(fastmath=True, parallel=True)
def delta_fourier_transform_parallel_2d_2d(
        qs: np.ndarray[float], rs: np.ndarray[float]) -> np.ndarray[complex]:

    r"""
    Parallel Numba-accelerated Fourier transforms of Dirac delta 
    functions involving all possible combinations of multiple 
    one-dimensional NumPy arrays :math:`\mathbf{q}` and 
    :math:`\mathbf{r}`, each with shape :math:`(3,)`.

    .. math::

       \mathcal{F}[\delta(\mathbf{q}-\mathbf{r})]
       =\exp(i\mathbf{q}\cdot\mathbf{r})

    Parameters
    ----------
    qs : `np.ndarray`
        Multiple vectors :math:`\mathbf{q}`.

        **Shape**: :math:`(N_q,\,3)`.

    rs : `np.ndarray`
        Multiple vectors :math:`\mathbf{r}`.

        **Shape**: :math:`(N_r,\,3,)`.

    Returns
    -------
    F : `np.ndarray`
        Fourier transforms of the Dirac delta functions, 
        :math:`\mathcal{F}[\delta(\mathbf{q}-\mathbf{r})]`.

        **Shape**: :math:`(N_q,\,N_r)`.
    """

    F = np.empty(qs.shape[0], dtype=np.complex128)
    for i in numba.prange(qs.shape[0]):
        F[i] = 0.0j
        for j in range(rs.shape[0]):
            F[i] += delta_fourier_transform_1d_1d(qs[i], rs[j])
    return F

@numba.njit(fastmath=True)
def inner_2d_2d(
        qs: np.ndarray[float], rs: np.ndarray[float]) -> np.ndarray[float]:

    r"""
    Serial Numba-accelerated inner product between all possible
    combinations of multiple one-dimensional NumPy arrays
    :math:`\mathbf{q}` and :math:`\mathbf{r}`, each with shape
    :math:`(3,)`.

    .. math::

       \mathbf{q}\cdot\mathbf{r}=q_1r_1+q_2r_2+q_3r_3

    Parameters
    ----------
    qs : `np.ndarray`
        Multiple vectors :math:`\mathbf{q}`.

        **Shape**: :math:`(N_q,\,3)`.

    rs : `np.ndarray`
        Multiple vectors :math:`\mathbf{r}`.

        **Shape**: :math:`(N_r,\,3,)`.

    Returns
    -------
    s : `np.ndarray`
        Inner products of the vectors, 
        :math:`\mathbf{q}\cdot\mathbf{r}`.

        **Shape**: :math:`(N_q,\,N_r)`.
    """

    s = np.empty((qs.shape[0], rs.shape[0]))
    for i in range(qs.shape[0]):
        for j in range(rs.shape[0]):
            s[i, j] = dot_1d_1d(qs[i], rs[j])
    return s

@numba.njit(fastmath=True, parallel=True)
def inner_parallel_2d_2d(
        qs: np.ndarray[float], rs: np.ndarray[float]) -> np.ndarray[float]:

    r"""
    Parallel Numba-accelerated inner product between all possible
    combinations of multiple one-dimensional NumPy arrays
    :math:`\mathbf{q}` and :math:`\mathbf{r}`, each with shape
    :math:`(3,)`.

    .. math::

       \mathbf{q}\cdot\mathbf{r}=q_1r_1+q_2r_2+q_3r_3

    Parameters
    ----------
    qs : `np.ndarray`
        Multiple vectors :math:`\mathbf{q}`.

        **Shape**: :math:`(N_q,\,3)`.

    rs : `np.ndarray`
        Multiple vectors :math:`\mathbf{r}`.

        **Shape**: :math:`(N_r,\,3,)`.

    Returns
    -------
    s : `np.ndarray`
        Inner products of the vectors, 
        :math:`\mathbf{q}\cdot\mathbf{r}`.

        **Shape**: :math:`(N_q,\,N_r)`.
    """

    s = np.empty((qs.shape[0], rs.shape[0]))
    for i in numba.prange(qs.shape[0]):
        for j in range(rs.shape[0]):
            s[i, j] = dot_1d_1d(qs[i], rs[j])
    return s

@numba.njit(fastmath=True)
def pythagorean_trigonometric_identity_1d(r: np.ndarray[float]) -> float:

    r"""
    Serial Numba-accelerated evaluation of the Pythagorean trigonometric
    identity for a one-dimensional NumPy array :math:`\mathbf{r}`.

    .. math::

       \left(\sum_{i=1}^3\cos(r_i)\right)^2
       +\left(\sum_{i=1}^3\sin(r_i)\right)^2

    Parameters
    ----------
    r : `np.ndarray`
        Vector :math:`\mathbf{r}`.

        **Shape**: :math:`(N_r,)`.

    Returns
    -------
    c2_s2 : `float`
        Pythagorean trigonometric identity for the vector 
        :math:`\mathbf{r}`.
    """

    c = s = 0
    for i in range(r.shape[0]):
        c += np.cos(r[i])
        s += np.sin(r[i])
    return c ** 2 + s ** 2

@numba.njit(fastmath=True)
def cross_pythagorean_trigonometric_identity_1d(
        r: np.ndarray[float], s: np.ndarray[float]) -> float:

    r"""
    Serial Numba-accelerated evaluation of the cross Pythagorean 
    trigonometric identity for two one-dimensional NumPy arrays
    :math:`\mathbf{r}` and :math:`\mathbf{s}`.

    .. math::

       2\left(\sum_{i=1}^3\cos(r_i)\sum_{j=1}^3\cos(s_j)
       +\sum_{i=1}^3\sin(r_i)\sum_{j=1}^3\sin(s_j)\right)

    Parameters
    ----------
    r : `np.ndarray`
        First vector :math:`\mathbf{r}`.

        **Shape**: :math:`(N_r,)`.

    s : `np.ndarray`
        Second vector :math:`\mathbf{s}`.

        **Shape**: :math:`(N_s,)`.

    Returns
    -------
    c2_s2 : `float`
        Cross Pythagorean trigonometric identity for the vectors
        :math:`\mathbf{r}` and :math:`\mathbf{s}`.
    """

    c1 = c2 = s1 = s2 = 0
    for i in range(r.shape[0]):
        c1 += np.cos(r[i])
        s1 += np.sin(r[i])
    for j in range(s.shape[0]):
        c2 += np.cos(s[j])
        s2 += np.sin(s[j])
    return 2 * (c1 * c2 + s1 * s2)