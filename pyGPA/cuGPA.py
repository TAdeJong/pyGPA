"""Module with cupy versions, i.e. GPU accelerated versions
of some geometric_phase_analysis functions."""

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cpndi

from .mathtools import wrapToPi


def cuGPA(image, kvec, sigma=22):
    """Perform spatial lock-in on an image

    GPU version of `optGPA()`.

    Parameters
    ----------
    image : np.array
        2D image input image
    kvec : 2-tuple or array of float
        components of the reference k-vector
    sigma : float, default=22
        standard deviation/ width of the Gaussian window

    Returns
    -------
    res : np.array, dtype complex
        Complex lock-in signal. Same shape as `image`.

    Notes
    -----
    This function should be a prime candidate to speed up using cupy.
    """
    xx, yy = cp.ogrid[0:image.shape[0], 0:image.shape[1]]
    multiplier = cp.exp(np.pi*2j * (xx*kvec[0] + yy*kvec[1]))
    X = cp.fft.fft2(cp.asarray(image) * multiplier)
    res = cp.fft.ifft2(cpndi.fourier_gaussian(X, sigma=sigma))
    return res


def wfr2_grad_opt(image, sigma, kx, ky, kw, kstep, grad=None):
    """Optimized version of wfr2_grad.

    In addition to returning the
    used k-vector and lock-in signal, return the gradient of the lock-in
    signal as well, for each pixel computed from the values of the surrounding pixels
    of the GPA of the best k-vector. Slightly more accurate, determination of this gradient,
    as boundary effects are mitigated.
    """
    xx, yy = cp.ogrid[0:image.shape[0],
                      0:image.shape[1]]
    c_image = cp.asarray(image)
    g = {'w': cp.zeros(image.shape + (2,)),
         'lockin': cp.zeros_like(c_image, dtype=np.complex128),
         'grad': cp.zeros(image.shape + (2,)),
         }
    gaussian = cpndi.fourier_gaussian(cp.ones_like(c_image), sigma=sigma)
    if grad == 'diff':
        def grad_func(phase):
            dbdx = cp.diff(phase, axis=0, append=np.nan)
            dbdy = cp.diff(phase, axis=1, append=np.nan)
            return dbdx, dbdy
    elif grad is None:
        def grad_func(phase):
            return cp.gradient(phase)
    else:
        grad_func = grad
    for wx in np.arange(kx-kw, kx+kw, kstep):
        for wy in np.arange(ky-kw, ky+kw, kstep):
            multiplier = cp.exp(np.pi*2j * (xx*wx + yy*wy))
            X = cp.fft.fft2(c_image * multiplier)
            X = X * gaussian
            sf = cp.fft.ifft2(X)
            t = cp.abs(sf) > cp.abs(g['lockin'])
            g['lockin'] = cp.where(t, sf * cp.exp(-2j*np.pi*((wx-kx)*xx + (wy-ky)*yy)), g['lockin'])
            g['w'] = cp.where(t[..., None], cp.array([wx, wy]), g['w'])

            angle = -cp.angle(sf)
            grad = grad_func(angle)
            grad = cp.stack(grad, axis=-1)
            # TODO: do outside forloop.
            g['grad'] = cp.where(t[..., None], grad + 2*np.pi * cp.array([(wx-kx), (wy-ky)]), g['grad'])
    for key in g.keys():
        g[key] = g[key].get()
    g['w'] = np.moveaxis(g['w'], -1, 0)
    g['grad'] = wrapToPi(2 * g['grad']) / 2
    return g


def wfr2_grad_single(image, sigma, kx, ky, kw, kstep, grad=None):
    """Optimized, single precision version of wfr2_grad.

    Single precision might be faster on some hardware.
    In addition to returning the
    used k-vector and lock-in signal, return the gradient of the lock-in
    signal as well, for each pixel computed from the values of the surrounding pixels
    of the GPA of the best k-vector. Slightly more accurate, determination of this gradient,
    as boundary effects are mitigated.
    """
    xx, yy = cp.ogrid[0:image.shape[0],
                      0:image.shape[1]]
    c_image = cp.asarray(image, dtype=np.float32)
    g = {'lockin': cp.zeros_like(c_image, dtype=np.complex64),
         'grad': cp.zeros(image.shape + (2,), dtype=np.float32),
         }
    gaussian = cpndi.fourier_gaussian(cp.ones_like(c_image, dtype=np.float32), sigma=sigma)
    if grad == 'diff':
        def grad_func(phase):
            dbdx = cp.diff(phase, axis=0, append=np.nan)
            dbdy = cp.diff(phase, axis=1, append=np.nan)
            return dbdx, dbdy
    elif grad is None:
        def grad_func(phase):
            return cp.gradient(phase)
    else:
        grad_func = grad
    for wx in np.arange(kx-kw, kx+kw, kstep):
        for wy in np.arange(ky-kw, ky+kw, kstep):
            multiplier = cp.exp(np.pi*2j * (xx*wx + yy*wy))
            X = cp.fft.fft2(c_image * multiplier)
            X = X * gaussian
            sf = cp.fft.ifft2(X)
            t = cp.abs(sf) > cp.abs(g['lockin'])
            angle = -cp.angle(sf)
            grad = grad_func(angle)
            grad = cp.stack(grad, axis=-1)
            g['lockin'] = cp.where(t, sf * cp.exp(-2j*np.pi*((wx-kx)*xx + (wy-ky)*yy)), g['lockin'])
            # TODO: do outside forloop.
            g['grad'] = cp.where(t[..., None], grad + 2*np.pi * cp.array([(wx-kx), (wy-ky)]), g['grad'])
    for key in g.keys():
        g[key] = g[key].get()
    g['grad'] = wrapToPi(2 * g['grad']) / 2
    return g


def wfr2_only_lockin(image, sigma, kvec, kw, kstep):
    """Optimized version of wfr2 calculating only the lock-in signal.
    Optimization in amount of computation done in each step by only
    computing updated values.
    """
    kx, ky = kvec
    xx, yy = cp.ogrid[0:image.shape[0],
                      0:image.shape[1]]
    c_image = cp.asarray(image)
    g = cp.zeros_like(c_image, dtype=np.complex)
    gaussian = cpndi.fourier_gaussian(cp.ones_like(c_image), sigma=sigma)
    for wx in np.arange(kx-kw, kx+kw, kstep):
        for wy in np.arange(ky-kw, ky+kw, kstep):
            multiplier = cp.exp(np.pi*2j * (xx*wx + yy*wy))
            X = cp.fft.fft2(c_image * multiplier)
            X = X * gaussian
            sf = cp.fft.ifft2(X)
            t = cp.abs(sf) > cp.abs(g)
            g = cp.where(t,
                         sf * cp.exp(-2j*np.pi*((wx-kx)*xx + (wy-ky)*yy)),
                         g)
    g = g.get()
    return g