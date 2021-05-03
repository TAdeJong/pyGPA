#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A weighed phase unwrap algorithm implemented in pure Python

Copied from original repository https://github.com/TAdeJong/weighed_phase_unwrap,
but should please just be merged in scikit-image, see:
https://github.com/scikit-image/scikit-image/issues/4622

author: Tobias A. de Jong
Based on:
Ghiglia, Dennis C., and Louis A. Romero.
"Robust two-dimensional weighted and unweighted phase unwrapping that uses
fast transforms and iterative methods." JOSA A 11.1 (1994): 107-117.
URL: https://doi.org/10.1364/JOSAA.11.000107
and an existing MATLAB implementation:
https://nl.mathworks.com/matlabcentral/fileexchange/60345-2d-weighted-phase-unwrapping
Should maybe use a scipy conjugate descent.
"""

import numpy as np
from scipy.fft import dctn, idctn


def phase_unwrap_ref(psi, weight, kmax=100):

    # vector b in the paper (eq 15) is dx and dy
    dx = _wrapToPi(np.diff(psi, axis=1))
    dy = _wrapToPi(np.diff(psi, axis=0))

    # multiply the vector b by weight square (W^T * W)
    WW = weight**2

    # See 3. Implementation issues: eq. 34 from Ghiglia et al.
    # Improves number of needed iterations. Different from matlab implementation
    WWx = np.minimum(WW[:, :-1], WW[:, 1:])
    WWy = np.minimum(WW[:-1, :], WW[1:, :])
    WWdx = WWx * dx
    WWdy = WWy * dy

    # applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
    WWdx2 = np.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = np.diff(WWdy, axis=0, prepend=0, append=0)

    rk = WWdx2 + WWdy2
    normR0 = np.linalg.norm(rk)

    # start the iteration
    eps = 1e-9
    k = 0
    phi = np.zeros_like(psi)
    while (~np.all(rk == 0.0)):
        zk = solvePoisson(rk)
        k += 1

        # equivalent to (rk*zk).sum()
        rkzksum = np.tensordot(rk, zk)
        if (k == 1):
            pk = zk
        else:
            betak = rkzksum / rkzkprevsum
            pk = zk + betak * pk

        # save the current value as the previous values
        rkzkprevsum = rkzksum

        # perform one scalar and two vectors update
        Qpk = applyQ(pk, WWx, WWy)
        alphak = rkzksum / np.tensordot(pk, Qpk)
        phi += alphak * pk
        rk -= alphak * Qpk

        # check the stopping conditions
        if ((k >= kmax) or (np.linalg.norm(rk) < eps * normR0)):
            break
        #print(np.linalg.norm(rk), normR0)
    print(k, rk.shape)
    return phi


def solvePoisson(rho):
    """Solve the poisson equation "P phi = rho" using DCT
    """
    dctRho = dctn(rho)
    N, M = rho.shape
    I, J = np.ogrid[0:N, 0:M]
    with np.errstate(divide='ignore'):
        dctPhi = dctRho / 2 / (np.cos(np.pi*I/M) + np.cos(np.pi*J/N) - 2)
    dctPhi[0, 0] = 0  # handling the inf/nan value
    # now invert to get the result
    phi = idctn(dctPhi)
    return phi


def solvePoisson_precomped(rho, scale):
    """Solve the poisson equation "P phi = rho" using DCT

    Uses precomputed scaling factors `scale`
    """
    dctPhi = dctn(rho) / scale
    # now invert to get the result
    phi = idctn(dctPhi, overwrite_x=True)
    return phi


def precomp_Poissonscaling(rho):
    N, M = rho.shape
    I, J = np.ogrid[0:N, 0:M]
    scale = 2 * (np.cos(np.pi*I/M) + np.cos(np.pi*J/N) - 2)
    # Handle the inf/nan value without a divide by zero warning:
    # By Ghiglia et al.:
    # "In practice we set dctPhi[0,0] = dctn(rho)[0, 0] to leave
    #  the bias unchanged"
    scale[0, 0] = 1.
    return scale


def applyQ(p, WWx, WWy):
    """Apply the weighted transformation (A^T)(W^T)(W)(A) to 2D matrix p"""
    # apply (A)
    dx = np.diff(p, axis=1)
    dy = np.diff(p, axis=0)

    # apply (W^T)(W)
    WWdx = WWx * dx
    WWdy = WWy * dy

    # apply (A^T)
    WWdx2 = np.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = np.diff(WWdy, axis=0, prepend=0, append=0)
    Qp = WWdx2 + WWdy2
    return Qp


def _wrapToPi(x):
    """Wrap all values of x to the interval -pi,pi"""
    r = (x+np.pi) % (2*np.pi) - np.pi
    return r


def phase_unwrap(psi, weight=None, kmax=100):
    """
    Unwrap the phase of an image psi given weights weight

    This function uses an algorithm described by Ghiglia and Romero
    and can either be used with or without weight array.
    It is especially suited to recover a unwrapped phase image
    from a (noisy) complex type image, where psi would be
    the angle of the complex values and weight the absolute values
    of the complex image.
    """

    # vector b in the paper (eq 15) is dx and dy
    dx = _wrapToPi(np.diff(psi, axis=1))
    dy = _wrapToPi(np.diff(psi, axis=0))

    # multiply the vector b by weight square (W^T * W)
    if weight is None:
        # Unweighed case. will terminate in 1 round
        WW = np.ones_like(psi)
    else:
        WW = weight**2

    # See 3. Implementation issues: eq. 34 from Ghiglia et al.
    # Improves number of needed iterations. Different from matlab implementation
    WWx = np.minimum(WW[:, :-1], WW[:, 1:])
    WWy = np.minimum(WW[:-1, :], WW[1:, :])
    WWdx = WWx * dx
    WWdy = WWy * dy

    # applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
    WWdx2 = np.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = np.diff(WWdy, axis=0, prepend=0, append=0)

    rk = WWdx2 + WWdy2
    normR0 = np.linalg.norm(rk)

    # start the iteration
    eps = 1e-9
    k = 0
    phi = np.zeros_like(psi)
    scaling = precomp_Poissonscaling(rk)
    while (~np.all(rk == 0.0)):
        zk = solvePoisson_precomped(rk, scaling)
        k += 1

        # equivalent to (rk*zk).sum()
        rkzksum = np.tensordot(rk, zk)
        if (k == 1):
            pk = zk
        else:
            betak = rkzksum / rkzkprevsum
            pk = zk + betak * pk

        # save the current value as the previous values

        rkzkprevsum = rkzksum

        # perform one scalar and two vectors update
        Qpk = applyQ(pk, WWx, WWy)
        alphak = rkzksum / np.tensordot(pk, Qpk)
        phi += alphak * pk
        rk -= alphak * Qpk

        # check the stopping conditions
        if ((k >= kmax) or (np.linalg.norm(rk) < eps * normR0)):
            break
    return phi


def phase_unwrap_ref_prediff(dx, dy, weight=None, kmax=100):
    """dx, dy sized NxM-1 and N-1xM, weights sized NxM"""
    # vector b in the paper (eq 15) is dx and dy
    #dx = _wrapToPi(np.diff(psi, axis=1))
    #dy = _wrapToPi(np.diff(psi, axis=0))
    if weight is None:
        WWx = np.ones_like(dx)
        WWy = np.ones_like(dy)
        WWdx = dx
        WWdy = dy
    else:
        # multiply the vector b by weight square (W^T * W)
        WW = weight**2

        # See 3. Implementation issues: eq. 34 from Ghiglia et al.
        # Improves number of needed iterations. Different from matlab implementation
        WWx = np.minimum(WW[:, :-1], WW[:, 1:])
        WWy = np.minimum(WW[:-1, :], WW[1:, :])
        WWdx = WWx * dx
        WWdy = WWy * dy

    # applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
    WWdx2 = np.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = np.diff(WWdy, axis=0, prepend=0, append=0)

    rk = WWdx2 + WWdy2
    normR0 = np.linalg.norm(rk)

    # start the iteration
    eps = 1e-9
    k = 0
    phi = np.zeros((dx.shape[0], dy.shape[1]))
    while (~np.all(rk == 0.0)):
        zk = solvePoisson(rk)
        k += 1

        # equivalent to (rk*zk).sum()
        rkzksum = np.tensordot(rk, zk)
        if (k == 1):
            pk = zk
        else:
            betak = rkzksum / rkzkprevsum
            pk = zk + betak * pk

        # save the current value as the previous values
        rkzkprevsum = rkzksum

        # perform one scalar and two vectors update
        Qpk = applyQ(pk, WWx, WWy)
        alphak = rkzksum / np.tensordot(pk, Qpk)
        phi += alphak * pk
        rk -= alphak * Qpk

        # check the stopping conditions
        if ((k >= kmax) or (np.linalg.norm(rk) < eps * normR0)):
            break
    return phi


def phase_unwrap_prediff(dx, dy, weight=None, kmax=100):
    """dx, dy sized NxM-1 and N-1xM, weights sized NxM"""
    if weight is None:
        WWx = np.ones_like(dx)
        WWy = np.ones_like(dy)
        WWdx = dx
        WWdy = dy
    else:
        # multiply the vector b by weight square (W^T * W)
        WW = weight**2

        # See 3. Implementation issues: eq. 34 from Ghiglia et al.
        # Improves number of needed iterations. Different from matlab implementation
        WWx = np.minimum(WW[:, :-1], WW[:, 1:])
        WWy = np.minimum(WW[:-1, :], WW[1:, :])
        WWdx = WWx * dx
        WWdy = WWy * dy

    # applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
    WWdx2 = np.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = np.diff(WWdy, axis=0, prepend=0, append=0)

    rk = WWdx2 + WWdy2
    normR0 = np.linalg.norm(rk)

    # start the iteration
    eps = 1e-9
    k = 0
    phi = np.zeros((dx.shape[0], dy.shape[1]))
    scaling = precomp_Poissonscaling(rk)
    while (~np.all(rk == 0.0)):
        zk = solvePoisson_precomped(rk, scaling)
        k += 1

        # equivalent to (rk*zk).sum()
        rkzksum = np.tensordot(rk, zk)
        if (k == 1):
            pk = zk
        else:
            betak = rkzksum / rkzkprevsum
            pk = zk + betak * pk

        # save the current value as the previous values
        rkzkprevsum = rkzksum

        # perform one scalar and two vectors update
        Qpk = applyQ(pk, WWx, WWy)
        alphak = rkzksum / np.tensordot(pk, Qpk)
        phi += alphak * pk
        rk -= alphak * Qpk

        # check the stopping conditions
        if ((k >= kmax) or (np.linalg.norm(rk) < eps * normR0)):
            break
    return phi
