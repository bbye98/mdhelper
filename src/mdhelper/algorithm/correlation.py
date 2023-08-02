"""
Statistical correlation
=======================
.. moduleauthor:: Benjamin B. Ye <bye@caltech.edu>

This module contains algorithms for computing the correlation between 
data with various shapes that evolve over time. This includes real- and
Fourier-space evaluations of the autocorrelation and cross-correlation 
functions, and by extension, the mean squared and cross displacements.
"""

import warnings

import numpy as np
from scipy import fft

def correlation_fft(
        arr1: np.ndarray, arr2: np.ndarray = None, axis: int = None, *,
        average: bool = False, double: bool = False, vector: bool = False
    ) -> np.ndarray:

    r"""
    Evaluates the autocorrelation function (ACF) or cross-correlation
    function (CCF) of a time series using fast Fourier transforms (FFT). 

    The algorithm, better known as the Fast Correlation Algorithm (FCA)
    [1]_ [2]_, is a result of the Wiener–Khinchin theorem and has a time
    complexity of :math:`\mathcal{O}(N\log{N})`. Effectively, the ACF
    can be computed from the raw data :math:`\mathbf{r}(t)` with two 
    FFTs using

    .. math::

       \begin{gather*}
         \hat{\mathbf{r}}(\xi)=\mathrm{FFT}(\mathbf{r}(t))\\
         A(\tau)=\mathrm{FFT}^{-1}(\hat{\mathbf{r}}(\xi)\hat{\mathbf{r}}^*(\xi))
       \end{gather*}

    where :math:`\tau` is the time lag and the asterisk (:math:`^*`) 
    denotes the complex conjugate.

    Similarly, the CCF for species :math:`i` and :math:`j` is evaluated
    using

    .. math::

       C_{ij}(\tau)=\mathrm{FFT}^{-1}(\mathrm{FFT}(\mathbf{r}_i(t))
       \cdot\mathrm{FFT}(\mathbf{r}_j(t)))

    Parameters
    ----------
    arr1 : `numpy.ndarray`
        Time evolution of :math:`N` entities over :math:`N_\mathrm{b}`
        blocks of :math:`N_t` frames each. 

        .. container::

           **Shape**: 
           
           * Scalar: :math:`(N_t,)`, :math:`(N_t,\,N)`,
             :math:`(N_\mathrm{b},\,N_t)`, or
             :math:`(N_\mathrm{b},\,N_t,\,N)`.
           * Vector: :math:`(N_t,\,d)`, 
             :math:`(N_t,\,N,\,N_\mathrm{d})`,
             :math:`(N_\mathrm{b},\,N_t,\,N_\mathrm{d})`, or 
             :math:`(N_\mathrm{b},\,N_t,\,N,\,N_\mathrm{d})`, where 
             :math:`N_\mathrm{d}` is the number of dimensions each 
             vector has.

    arr2 : `numpy.ndarray`, optional
        Time evolution of another :math:`N` entities. If provided, the
        CCF for `arr1` and `arr2` is calculated. Otherwise, the ACF for 
        `arr1` is calculated.
        
        **Shape**: Same as `arr1`.

    axis : `int`, optional
        Axis along which to evaluate the ACF/CCF. If `arr1` contains a
        full, unsplit trajectory, the ACF/CCF should be evaluated along
        the first axis (:code:`axis=0`). If `arr1` contains a 
        trajectory split into multiple blocks, the ACF/CCF should be 
        evaluated along the second axis (:code:`axis=1`). If not
        specified, the axis is determined automatically using the shape
        of `arr1`.

    average : `bool`, keyword-only, default: :code:`True`
        Determines whether the ACF/CCF is averaged over all entities if
        the arrays contain information for multiple entities.

    double : `bool`, keyword-only, default: :code:`False`
        If :code:`True`, the ACF is doubled or the CCFs for the negative
        and positive time lags are combined. Useful for evaluating the 
        mean squared or cross displacement. See
        :func:`mdhelper.algorithm.correlation.msd_fft` for more 
        information.

    vector : `bool`, keyword-only, default: :code:`False`
        Specifies whether `arr1` and `arr2` contain vectors. If 
        :code:`True`, the ACF/CCF is summed over the last dimension.

    Returns
    -------
    corr : `numpy.ndarray`
        Autocorrelation or cross-correlation function.

        .. container::

           **Shape**:

           For ACF, the shape is that of `arr1` but with the following
           modifications:

           * If :code:`average=True`, the axis containing the :math:`N`
             entities is removed.
           * If :code:`vector=True`, the last dimension is removed.

           For CCF, the shape is that of `arr1` but with the following
           modifications:

           * If :code:`average=True`, the axis containing the :math:`N`
             entities is removed.
           * If :code:`double=False`, the axis containing the 
             :math:`N_t` times now has a length of :math:`2N_t-1` to
             accomodate negative and positive time lags.
           * If :code:`vector=True`, the last dimension is removed.

    References
    ----------
    .. [1] Kneller, G. R.; Keiner, V.; Kneller, M.; Schiller, M. 
       NMOLDYN: A Program Package for a Neutron Scattering Oriented 
       Analysis of Molecular Dynamics Simulations. *Computer Physics
       Communications* **1995**, *91* (1–3), 191–214. 
       https://doi.org/10.1016/0010-4655(95)00048-K.

    .. [2] Calandrini, V.; Pellegrini, E.; Calligari, P.; Hinsen, K.;
       Kneller, G. R. NMoldyn - Interfacing Spectroscopic Experiments,
       Molecular Dynamics Simulations and Models for Time Correlation
       Functions. *JDN* **2011**, *12*, 201–232.
       https://doi.org/10.1051/sfn/201112010.
    """

    # Ensure arrays have valid dimensionality
    if not isinstance(arr1, np.ndarray):
        arr1 = np.array(arr1)
    if arr1.size == 0:
        raise ValueError("The arrays must not be empty.")
    ndim = arr1.ndim
    if not 1 <= ndim <= 4:
        emsg = ("The arrays must be one-, two-, three-, or four-"
                "dimensional.")
        raise ValueError(emsg)
    if arr2 is not None:
        if not isinstance(arr2, np.ndarray):
            arr2 = np.array(arr2)
        if arr1.shape != arr2.shape:
            emsg = "The arrays must have the same dimensions."
            raise ValueError(emsg)

    # Check or set axis along which to compute the ACF/CCF
    if axis is None:
        if ndim == 4:
            axis = 1
        else:
            axis = 0
            if ndim > 1:
                wmsg = ("The axis along which to compute the ACF/CCF "
                        "was not specified and is ambiguous for a "
                        "multidimensional array. As such, it has been "
                        "set to the first axis by default.")
                warnings.warn(wmsg)
    elif axis not in {0, 1}:
        emsg = ("The ACF/CCF can only be evaluated along the first or "
                "second axis.")
        raise ValueError(emsg)

    # Get number of frames
    N_t = arr1.shape[axis]

    # Calculate the PSD by first zero-padding the arrays for linear
    # convolution, and then invert it to get the ACF/CCF
    if arr2 is None:
        f = fft.rfft(arr1, n=2 * N_t, axis=axis)
        corr = (double + 1) * fft.irfft(f * f.conjugate(), axis=axis)
        corr = corr[:, :N_t] if axis else corr[:N_t]
    else:
        f1 = fft.rfft(arr1, n=2 * N_t, axis=axis)
        f2 = fft.rfft(arr2, n=2 * N_t, axis=axis)
        f = f1.conjugate() * f2
        if double:
            corr = fft.irfft(f + f1 * f2.conjugate(), axis=axis)
            corr = corr[:, :N_t] if axis else corr[:N_t]
        else:
            corr = fft.irfft(f, axis=axis)

    # Determine the axes over which to expand the reversed
    # time array for correct matrix division
    if vector:
        axes = list(range(ndim - 1))
        if axis in axes:
            axes.remove(axis)

        # Sum over the last dimension if the arrays contain vectors
        corr = corr.sum(axis=-1)
    elif axis:
        axes = list(range(ndim))
        if axis in axes:
            axes.remove(axis)
    else:
        axes = list(range(1, ndim))

    # Normalize the ACF/CCF
    if axis:
        corr[:, :N_t] /= np.expand_dims(range(N_t, 0, -1), axes)
        if corr.shape[axis] != N_t:
            corr[:, 1 - N_t:] /= np.expand_dims(range(1, N_t), axes)
            corr = np.hstack((corr[:, 1 - N_t:], corr[:, :N_t]))
    else:
        corr[:N_t] /= np.expand_dims(range(N_t, 0, -1), axes)
        if corr.shape[axis] != N_t:
            corr[1 - N_t:] /= np.expand_dims(range(1, N_t), axes)
            corr = np.concatenate((corr[1 - N_t:], corr[:N_t]))
    
    # Average over all entities, if desired
    if average:
        axis_avg = ndim - 1 - vector
        if axis != axis_avg:
            return corr.mean(axis=axis_avg)
    return corr

def correlation_shift(
        arr1: np.ndarray, arr2: np.ndarray = None, axis: int = None, 
        vector: bool = False, *, average: bool = False, double: bool = False
    ) -> np.ndarray:

    r"""
    Evaluates the autocorrelation function (ACF) or cross-correlation
    function (CCF) of a time series directly by using sliding windows
    along the time axis.

    For scalars :math:`r` or vectors :math:`\mathbf{r}`, the ACF is 
    defined as

    .. math::

        A(\tau)=\langle\textbf{r}(t_0+\tau)\cdot\textbf{r}(t_0)\rangle
        =\dfrac{1}{N}\sum_{\alpha=1}^N
        \textbf{r}_\alpha(t_0+\tau)\cdot\textbf{r}_\alpha(t_0)
        
    while the CCF for species :math:`i` and :math:`j` is given by
    
    .. math::

        C_{ij}(\tau)=\langle\textbf{r}_i(t_0+\tau)\cdot
        \textbf{r}_j(t_0)\rangle
        =\dfrac{1}{N}\sum_{\alpha=1}^N\textbf{r}_{i,\alpha}(t_0+\tau)\cdot
        \textbf{r}_{j,\alpha}(t_0)

    where :math:`\tau` is the time lag, :math:`t_0` is an arbitrary 
    reference time, and :math:`N` is the number of entities. To reduce
    statistical noise, the ACF/CCF is calculated for and averaged over
    all possible reference times :math:`t_0`. As such, this algorithm 
    has a time complexity of :math:`\mathcal{O}(N^2)`.

    With large data sets, this approach is too slow to be useful. If 
    your machine supports the fast Fourier transform (FFT), consider 
    using the much more performant FFT-based algorithm implemented in
    :func:`mdhelper.algorithm.correlation.correlation_fft` instead.

    Parameters
    ----------
    arr1 : `numpy.ndarray`
        Time evolution of :math:`N` entities over :math:`N_\mathrm{b}`
        blocks of :math:`N_t` frames each. 

        .. container::

           **Shape**: 
           
           * Scalar: :math:`(N_t,)`, :math:`(N_t,\,N)`,
             :math:`(N_\mathrm{b},\,N_t)`, or
             :math:`(N_\mathrm{b},\,N_t,\,N)`.
           * Vector: :math:`(N_t,\,d)`, 
             :math:`(N_t,\,N,\,N_\mathrm{d})`,
             :math:`(N_\mathrm{b},\,N_t,\,N_\mathrm{d})`, or 
             :math:`(N_\mathrm{b},\,N_t,\,N,\,N_\mathrm{d})`, where 
             :math:`N_\mathrm{d}` is the number of dimensions each 
             vector has.

    arr2 : `numpy.ndarray`, optional
        Time evolution of another :math:`N` entities. If provided, the
        CCF for `arr1` and `arr2` is calculated. Otherwise, the ACF for 
        `arr1` is calculated.
        
        **Shape**: Same as `arr1`.

    axis : `int`, optional
        Axis along which to evaluate the ACF/CCF. If `arr1` contains a
        full, unsplit trajectory, the ACF/CCF should be evaluated along
        the first axis (:code:`axis=0`). If `arr1` contains a 
        trajectory split into multiple blocks, the ACF/CCF should be 
        evaluated along the second axis (:code:`axis=1`). If not
        specified, the axis is determined automatically using the shape
        of `arr1`.

    average : `bool`, keyword-only, default: :code:`True`
        Determines whether the ACF/CCF is averaged over all entities if
        the arrays contain information for multiple entities.

    double : `bool`, keyword-only, default: :code:`False`
        If :code:`True`, the ACF is doubled or the CCFs for the negative
        and positive time lags are combined. Useful for evaluating the 
        mean squared or cross displacement. See
        :func:`mdhelper.algorithm.correlation.msd_shift` for more 
        information.

    vector : `bool`, keyword-only, default: :code:`False`
        Specifies whether `arr1` and `arr2` contain vectors. If 
        :code:`True`, the ACF/CCF is summed over the last dimension.

    Returns
    -------
    corr : `numpy.ndarray`
        Autocorrelation or cross-correlation function.

        .. container::

           **Shape**:

           For ACF, the shape is that of `arr1` but with the following
           modifications:

           * If :code:`average=True`, the axis containing the :math:`N`
             entities is removed.
           * If :code:`vector=True`, the last dimension is removed.

           For CCF, the shape is that of `arr1` but with the following
           modifications:

           * If :code:`average=True`, the axis containing the :math:`N`
             entities is removed.
           * If :code:`double=False`, the axis containing the 
             :math:`N_t` times now has a length of :math:`2N_t-1` to
             accomodate negative and positive time lags.
           * If :code:`vector=True`, the last dimension is removed.
    """

    # Ensure arrays have valid dimensionality
    if not isinstance(arr1, np.ndarray):
        arr1 = np.array(arr1)
    if arr1.size == 0:
        raise ValueError("The arrays must not be empty.")
    ndim = arr1.ndim
    if not 1 <= ndim <= 4:
        emsg = ("The arrays must be one-, two-, three-, or four-"
                "dimensional.")
        raise ValueError(emsg)
    if arr2 is not None:
        if not isinstance(arr2, np.ndarray):
            arr2 = np.array(arr2)
        if arr1.shape != arr2.shape:
            emsg = "The arrays must have the same dimensions."
            raise ValueError(emsg)

    # Check or set axis along which to compute the ACF/CCF
    if axis is None:
        if ndim == 4:
            axis = 1
        else:
            axis = 0
            if ndim > 1:
                wmsg = ("The axis along which to compute the ACF/CCF "
                        "was not specified and is ambiguous for a "
                        "multidimensional array. As such, it has been "
                        "set to the first axis by default.")
                warnings.warn(wmsg)
    elif axis not in {0, 1}:
        emsg = ("The ACF/CCF can only be evaluated along the first or "
                "second axis.")
        raise ValueError(emsg)

    # Get number of frames
    N_t = arr1.shape[axis]

    # Calculate the ACF/CCF
    if arr2 is None:
        if ndim == 1:
            corr = np.fromiter(
                (np.dot(arr1[i:], arr1[:-i if i else None]) 
                 for i in range(N_t)),
                count=N_t, dtype=float
            )
        elif axis:
            axes = f"bt...{'d' * vector}"
            corr = np.stack(
                [np.einsum(f"{axes},{axes}->b...", 
                           arr1[:, i:], arr1[:, :-i if i else None]) 
                 for i in range(N_t)],
                axis=1
            )
        else:
            axes = f"t...{'d' * vector}"
            corr = np.stack(
                [np.einsum(f"{axes},{axes}->...", 
                           arr1[i:], arr1[:-i if i else None]) 
                 for i in range(N_t)]
            )
    else:
        start = np.r_[np.zeros(N_t - 1, dtype=int), 0:N_t]
        stop = np.r_[1:N_t + 1, N_t * np.ones(N_t - 1, dtype=int)]
        if ndim == 1:
            corr = np.fromiter(
                (np.dot(arr1[i:j], arr2[k:m]) 
                 for i, j, k, m in zip(start[::-1], stop[::-1], start, stop)),
                count=2 * N_t - 1, dtype=float
            )
        elif axis:
            axes = f"bt...{'d' * vector}"
            corr = np.stack(
                [np.einsum(f"{axes},{axes}->b...", arr1[:, i:j], arr2[:, k:m])
                 for i, j, k, m in zip(start[::-1], stop[::-1], start, stop)], 
                axis=1
            )
        else:
            axes = f"t...{'d' * vector}"
            corr = np.stack(
                [np.einsum(f"{axes},{axes}->...", arr1[i:j], arr2[k:m]) 
                 for i, j, k, m in zip(start[::-1], stop[::-1], start, stop)]
            )

    # Doubles the ACF or overlaps the CCF for negative and positive 
    # lags, if desired
    if double:
        if arr2 is None:
            corr *= 2
        elif axis:
            corr = corr[:, N_t - 1:] + corr[:, N_t - 1::-1]
        else:
            corr = corr[N_t - 1:] + corr[N_t - 1::-1]

    # Determine the axes over which to expand the reversed
    # time array for correct matrix division
    if axis or vector:
        axes = list(range(ndim - vector))
        if axis in axes:
            axes.remove(axis)
    else:
        axes = list(range(1, ndim))

    # Normalize the ACF/CCF
    if axis:
        corr[:, -N_t:] /= np.expand_dims(range(N_t, 0, -1), axes)
        if corr.shape[axis] != N_t:
            corr[:, :N_t - 1] /= np.expand_dims(range(1, N_t), axes)
    else:
        corr[-N_t:] /= np.expand_dims(range(N_t, 0, -1), axes)
        if corr.shape[axis] != N_t:
            corr[:N_t - 1] /= np.expand_dims(range(1, N_t), axes)

    # Average over all entities, if desired
    if average:
        axis_avg = ndim - 1 - vector
        if axis != axis_avg:
            return corr.mean(axis=axis_avg)
    return corr

def msd_fft(
        pos1: np.ndarray, pos2: np.ndarray = None, axis: int = None, *,
        average: bool = True) -> np.ndarray:

    r"""
    Calculates the mean squared displacement (MSD) or the analogous 
    cross displacement (CD) using fast Fourier transforms (FFT).
    
    The algorithm [1]_ [2]_ is
    
    .. math::

        \mathrm{MSD}_m&=\frac{1}{N_t-m}\sum_{k=0}^{N_t-m-1}
        \left[\textbf{r}_{k+m}-\textbf{r}_k\right]^2\\
        &=\frac{1}{N_t-m}\sum_{k=0}^{N_t-m-1}
        \left[\textbf{r}_{k+m}^2+\textbf{r}_k^2\right]
        -\frac{2}{N_t-m}\sum_{k=0}^{N_t-m-1}
        \textbf{r}_k\cdot\textbf{r}_{k+m}\\
        &=S_m-2A_m

    where :math:`m` is the index corresponding to time lag :math:`\tau`,
    :math:`A_m` is the autocorrelation of the positions, which is
    evaluated by calculating the power spectral density (PSD), inverting
    it, and then dividing by :math:`N_t-m`, and :math:`S_m` can be
    evaluated using the recursive relation

    .. math::

        \begin{gather*}
          D_k=\textbf{r}_k^2\\
          Q_{-1}=2\sum_{k=0}^{N_t-1}D_k\\
          Q_m=Q_{m-1}-D_{m-1}-D_{N_t-m}\\
          S_m=\dfrac{Q_m}{N_t-m}
        \end{gather*}

    Similarly, when two distinct sets of positions are provided, the CD 
    is computed using

    .. math::

       \mathrm{CD}_{ij,m}=S_m-2C_{ij,m}

    where :math:`C_{ij,m}` is the cross-correlation of the two sets of 
    positions and :math:`D_k` in :math:`S_m` is replaced with the
    analogous :math:`D_{ij,k} = \textbf{r}_{i,k}\cdot\textbf{r}_{j,k}`.

    .. note::
    
       To evaluate the sum in the expression used to calculate the 
       Onsager transport coefficients [3]_

       .. math::
            
          L_{ij}=\frac{1}{6k_\mathrm{B}T}\lim_{t\rightarrow\infty}
          \frac{d}{dt}\left\langle\sum_{\alpha=1}^{N_i}
          [\mathrm{r}_\alpha(t)-\mathrm{r}_\alpha(0)]\cdot
          \sum_{\beta=1}^{N_j}[\mathrm{r}_\beta(t)-\mathrm{r}_\beta(0)]
          \right\rangle

       `pos1` and `pos2` should be summed over all atoms before being
       passed to this function.

    Parameters
    ----------
    pos1 : `numpy.ndarray`
        Individual or averaged position(s) of the :math:`N` particles
        in the first particle group over :math:`N_\mathrm{b}`
        blocks of :math:`N_t` frames each. 
        
        **Shape**: :math:`(N_t,\,3)`, :math:`(N_t,\,N,\,3)`, or
        :math:`(N_\mathrm{b},\,N_t,\,N,\,3)`.
        
        **Reference unit**: :math:`\mathrm{Å}`.

    pos2 : `numpy.ndarray`, optional
        Individual or averaged position(s) of the :math:`N` particles
        in the second particle group over :math:`N_t` frames. 
        
        **Shape**: Same as `pos1`.

        **Reference unit**: :math:`\mathrm{Å}`.

    axis : `int`, optional
        Axis along which to evaluate the MSD/CD. If `pos1` and/or `pos2`
        contain a full, unsplit trajectory, the MSD/CD should be
        evaluated along the first axis (:code:`axis=0`). If `pos1`
        and/or `pos2` contain a trajectory split into multiple blocks,
        the MSD should be evaluated along the second axis 
        (:code:`axis=1`). If not provided, the axis is selected 
        automatically using the shape of `pos1`.

    average : `bool`, keyword-only, default: :code:`True`
        Determines whether the MSD is averaged over all particles if the
        position arrays contain information for multiple particles.

    Returns
    -------
    disp : `numpy.ndarray`
        Mean-squared or cross displacement.
        
        **Shape**: The shape of `pos`, except with the last dimension 
        removed. If :code:`average=True`, the axis containing the 
        :math:`N` entities is also removed.
        
        **Reference unit**: :math:`\text{Å}^2`.

    References
    ----------
    .. [1] Kneller, G. R.; Keiner, V.; Kneller, M.; Schiller, M. 
       NMOLDYN: A Program Package for a Neutron Scattering Oriented 
       Analysis of Molecular Dynamics Simulations. *Computer Physics
       Communications* **1995**, *91* (1–3), 191–214. 
       https://doi.org/10.1016/0010-4655(95)00048-K.

    .. [2] Calandrini, V.; Pellegrini, E.; Calligari, P.; Hinsen, K.;
       Kneller, G. R. NMoldyn - Interfacing Spectroscopic Experiments,
       Molecular Dynamics Simulations and Models for Time Correlation
       Functions. *JDN* **2011**, *12*, 201–232.
       https://doi.org/10.1051/sfn/201112010.

    .. [3] Fong, K. D.; Self, J.; McCloskey, B. D.; Persson, K. A.
       Onsager Transport Coefficients and Transference Numbers in
       Polyelectrolyte Solutions and Polymerized Ionic Liquids.
       *Macromolecules* **2020**, *53* (21), 9503–9512.
       https://doi.org/10.1021/acs.macromol.0c02001.
    """

    # Ensure arrays have valid dimensionality
    if not isinstance(pos1, np.ndarray):
        pos1 = np.array(pos1)
    if pos1.size == 0:
        raise ValueError("The position arrays must not be empty.")
    ndim = pos1.ndim
    if not 2 <= ndim <= 4:
        emsg = ("The position arrays must be two-, three-, or four-"
                "dimensional.")
        raise ValueError(emsg)
    if pos2 is not None:
        if not isinstance(pos2, np.ndarray):
            pos2 = np.array(pos2)
        if pos1.shape != pos2.shape:
            emsg = "The position arrays must have the same dimensions."
            raise ValueError(emsg)

    # Check or set axis along which to compute the MSD/CD
    if axis is None:
        if ndim == 4:
            axis = 1
        else:
            axis = 0
            if ndim == 3:
                emsg = ("The axis along which to compute the MSD/CD "
                        "was not specified and is ambiguous for a "
                        "three-dimensional array. As such, it has been "
                        "set to the first axis by default.")
                warnings.warn(emsg)
    elif axis not in {0, 1}:
        emsg = ("The MSD/CD can only be evaluated along the first or "
                "second axis.")
        raise ValueError(emsg)

    # Get number of frames
    N_t = pos1.shape[axis]

    # Get intermediate quantities required for the MSD/CD calculation
    s2 = correlation_fft(pos1, pos2, axis, average=False, double=True, 
                         vector=True)
    r1r2 = (pos1 * (pos1 if pos2 is None else pos2)).sum(axis=-1)

    if ndim - axis == 3:

        # Calculate the MSD/CD for each particle
        if not average:
            stack = np.vstack if ndim == 3 else np.hstack
            shape = np.array(pos1.shape[:-1])
            mask = np.ones(len(shape), dtype=bool)
            mask[axis] = False
            zeros = np.expand_dims(np.zeros(shape[mask]), (axis,))
            D = stack((r1r2, zeros))
            if axis:
                ssum = 2 * D.sum(axis=axis)[:, None] * np.ones((1, N_t, 1)) \
                       - np.cumsum(D[:, range(-1, N_t - 1)] + D[:, N_t:0:-1],
                                   axis=axis)
            else:
                ssum = 2 * D.sum(axis=axis) * np.ones((N_t, 1)) \
                       - np.cumsum(D[range(-1, N_t - 1)] + D[N_t:0:-1],
                                   axis=axis)
            return ssum / np.arange(N_t, 0, -1)[:, None] - s2

        # Average the intermediate quantities over all particles
        s2 = s2.mean(axis=ndim - 2)
        r1r2 = r1r2.mean(axis=ndim - 2)

    # Calculate the averaged MSD/CD
    if axis:
        ssum = 2 * r1r2.sum(axis=axis)[:, None] * np.ones((1, N_t)) \
               - np.insert(np.cumsum(r1r2[:, :N_t - 1] + r1r2[:, N_t - 1:0:-1],
                                     axis=axis),
                           0, 0, axis=axis)
    else:
        ssum = 2 * r1r2.sum() * np.ones(N_t) \
               - np.insert(np.cumsum(r1r2[:N_t - 1] + r1r2[N_t - 1:0:-1]), 0, 0)
    return ssum / np.arange(N_t, 0, -1) - s2

def msd_shift(
        pos1: np.ndarray, pos2: np.ndarray = None, axis: int = None, *,
        average: bool = True) -> np.ndarray:

    r"""
    Calculates the mean squared displacement (MSD) or the analogous 
    cross displacement (CD) using the Einstein relation.

    The MSD is defined as

    .. math::

        \mathrm{MSD}(\tau)=\langle\left[
        \textbf{r}(t_0+\tau)-\textbf{r}(t_0)\right]^2\rangle
        =\dfrac{1}{N}\sum_{\alpha=1}^N\left[
        \textbf{r}_\alpha(t_0+\tau)-\textbf{r}_\alpha(t_0)\right]^2

    while the CD between species :math:`i` and :math:`j` is given by

    .. math::

        \mathrm{CD}_{ij}(\tau)&=\langle
        (\textbf{r}_i(t_0+\tau)-\textbf{r}_i(t_0))\cdot
        (\textbf{r}_j(t_0+\tau)-\textbf{r}_j(t_0))\rangle\\
        &=\dfrac{1}{N}\sum_{\alpha=1}^N
        (\textbf{r}_{i,\alpha}(t_0+\tau)-\textbf{r}_{i,\alpha}(t_0))\cdot
        (\textbf{r}_{j,\alpha}(t_0+\tau)-\textbf{r}_{j,\alpha}(t_0))

    where :math:`\tau` is the time lag, :math:`t_0` is an arbitrary 
    reference time, and :math:`N` is the number of entities. To reduce
    statistical noise, the MSD/CD is calculated for and averaged over
    all possible reference times :math:`t_0`.

    .. note::
    
       To evaluate the sum in the expression used to calculate the 
       Onsager transport coefficients [1]_,

       .. math::
        
          L_{ij}=\frac{1}{6k_\mathrm{B}T}\lim_{t\rightarrow\infty}
          \frac{d}{dt}\left\langle\sum_{\alpha=1}^{N_i}
          [\mathrm{r}_\alpha(t)-\mathrm{r}_\alpha(0)]\cdot
          \sum_{\beta=1}^{N_j}[\mathrm{r}_\beta(t)-\mathrm{r}_\beta(0)]
          \right\rangle

       `pos1` and `pos2` should be summed over all atoms before being
       passed to this function.

    Parameters
    ----------
    pos1 : `numpy.ndarray`
        Individual or averaged position(s) of the :math:`N` particles
        in the first particle group over :math:`N_t` frames. 
        
        **Shape**: :math:`(N_t,\,3)`, :math:`(N_t,\,N,\,3)`, or
        :math:`(N_\mathrm{b},\,N_t,\,N,\,3)`.
        
        **Reference unit**: :math:`\mathrm{Å}`.

    pos2 : `numpy.ndarray`, optional
        Individual or averaged position(s) of the :math:`N` particles
        in the second particle group over :math:`N_t` frames. 
        
        **Shape**: Same as `pos1`.

        **Reference unit**: :math:`\mathrm{Å}`.

    axis : `int`, optional
        Axis along which to evaluate the MSD/CD. If `pos1` and/or `pos2`
        contain a full, unsplit trajectory, the MSD/CD should be
        evaluated along the first axis (:code:`axis=0`). If `pos1`
        and/or `pos2` contain a trajectory split into multiple blocks,
        the MSD should be evaluated along the second axis 
        (:code:`axis=1`). If not provided, the axis is selected 
        automatically using the shape of `pos1`.

    average : `bool`, keyword-only, default: :code:`True`
        Determines whether the MSD is averaged over all particles if the
        position arrays contain information for multiple particles.

    Returns
    -------
    disp : `numpy.ndarray`
        Mean-squared or cross displacement.
        
        **Shape**: The shape of `pos`, except with the last dimension 
        removed. If :code:`average=True`, the axis containing the 
        :math:`N` entities is also removed.
        
        **Reference unit**: :math:`\text{Å}^2`.

    References
    ----------
    .. [1] Fong, K. D.; Self, J.; McCloskey, B. D.; Persson, K. A.
       Onsager Transport Coefficients and Transference Numbers in
       Polyelectrolyte Solutions and Polymerized Ionic Liquids.
       *Macromolecules* **2020**, *53* (21), 9503–9512.
       https://doi.org/10.1021/acs.macromol.0c02001.
    """

    # Ensure arrays have valid dimensionality
    if not isinstance(pos1, np.ndarray):
        pos1 = np.array(pos1)
    if pos1.size == 0:
        raise ValueError("The position arrays must not be empty.")
    ndim = pos1.ndim
    if not 2 <= ndim <= 4:
        emsg = ("The position arrays must be two-, three-, or four-"
                "dimensional.")
        raise ValueError(emsg)
    if pos2 is not None:
        if not isinstance(pos2, np.ndarray):
            pos2 = np.array(pos2)
        if pos1.shape != pos2.shape:
            emsg = "The position arrays must have the same dimensions."
            raise ValueError(emsg)

    # Check or set axis along which to compute the MSD/CD
    if axis is None:
        if ndim == 4:
            axis = 1
        else:
            axis = 0
            if ndim == 3:
                emsg = ("The axis along which to compute the MSD/CD "
                        "was not specified and is ambiguous for a "
                        "three-dimensional array. As such, it has been "
                        "set to the first axis by default.")
                warnings.warn(emsg)
    elif axis not in {0, 1}:
        emsg = ("The MSD/CD can only be evaluated along the first or "
                "second axis.")
        raise ValueError(emsg)

    # Get number of frames
    N_t = pos1.shape[axis]
    
    # Calculate the MSD/CD for each atom
    if pos2 is None:
        if axis:
            disp = np.stack(
                [
                    (
                        (pos1[:, :-i if i else None] - pos1[:, i:]) ** 2
                    ).sum(axis=-1).mean(axis=axis) for i in range(N_t)
                ],
                axis=1
            )
        else:
            disp = np.stack(
                [
                    (
                        (pos1[:-i if i else None] - pos1[i:]) ** 2
                    ).sum(axis=-1).mean(axis=axis) for i in range(N_t)
                ]
            )
    else:
        if axis:
            disp = np.stack(
                [
                    np.einsum(
                        "bt...d,bt...d->bt...", 
                        pos1[:, :-i if i else None] - pos1[:, i:], 
                        pos2[:, :-i if i else None] - pos2[:, i:]
                    ).mean(axis=axis) for i in range(N_t)
                ],
                axis=1
            )
        else:
            disp = np.stack(
                [
                    np.einsum(
                        "t...d,t...d->t...", 
                        pos1[:-i if i else None] - pos1[i:], 
                        pos2[:-i if i else None] - pos2[i:]
                    ).mean(axis=axis) for i in range(N_t)
                ]
            )
    
    # Average over all particles, if desired
    if ndim - axis == 3 and average:
        disp = disp.mean(axis=ndim - 2)

    return disp