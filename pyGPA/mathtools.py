import numpy as np
import scipy.optimize as spo

def periodic_average(X, period=2*np.pi, weights=1., **kwargs):
    """Take the periodic average of X, weighted by weights."""
    Y = weights * np.exp(1j * 2*np.pi / period * X)
    Y = np.angle(Y.mean(**kwargs))
    return Y * period / (2*np.pi)


def periodic_difference(X, Y, period=2*np.pi):
    """Take the periodic difference of X and Y"""
    Z =  np.exp(1j * 2*np.pi / period * (X-Y))
    Z = np.angle(Z)
    return Z * period / (2*np.pi)


def lfit_func(x, image, xx, yy):
    ax, ay, b = x
    return (image - (ax*xx + ay*yy + b)).flatten()


def lfit_func_mask(x, image, xx, yy, mask):
    ax, ay, b = x
    return (image - (ax*xx + ay*yy + b))[mask].flatten()


def fit_plane(image, verbose=False):
    lxx, lyy = np.meshgrid(np.arange(image.shape[0]),
                           np.arange(image.shape[1]),
                           indexing='ij')
    x0 = np.zeros(3)
    res = spo.least_squares(lfit_func, x0,
                            loss='huber', 
                            args=(image, lxx, lyy))
    if verbose:
        print(res.message)
    return res.x


def fit_plane_masked(image, verbose=False, mask=False):
    lxx, lyy = np.meshgrid(np.arange(image.shape[0]),
                           np.arange(image.shape[1]),
                           indexing='ij')
    x0 = np.zeros(3)
    if mask:
        res = spo.least_squares(lfit_func_mask, x0,
                                loss='huber',
                                args=(image, lxx, lyy, mask))
    else:
        res = spo.least_squares(lfit_func, x0,
                                loss='huber',
                                args=(image, lxx, lyy))
    if verbose:
        print(res.message)
    return res.x


def wrapToPi(x):
    """Wrap all values of x to the interval -pi,pi"""
    r = (x+np.pi)  % (2*np.pi) - np.pi
    return r


def remove_negative_duplicates(ks):
    """For a list of length 2 vectors ks (or Nx2 array),
    return the array of vectors without negative duplicates
    of the vectors (x-coord first, if zero non-negative y)
    """
    if ks.shape[0] == 0:
        return ks
    npks = []
    nonneg = np.where(np.sign(ks[:, [0]]) != 0, 
                      np.sign(ks[:, [0]]) * ks, 
                      np.sign(ks[:, [1]]) * ks)
    npks = [nonneg[0]]
    atol = 1e-3 * np.min(np.abs(nonneg), axis=1).mean()
    for k in nonneg[1:]:
        if not np.any(np.all(np.isclose(k, npks, atol=atol),axis=1)):
            npks.append(k)
    return np.array(npks)


def standardize_ks(kvecs):
    """standardize order and quadrant of lattice representation
    
    For a list `kvecs` of k-vectors representing a lattice,
    return the sorted vectors with positive angles i.e. without
    negative duplicates.
    """
    newvecs = remove_negative_duplicates(kvecs)
    symmetry = len(newvecs) * 2
    newvecs = np.concatenate([newvecs, -newvecs], axis=0)
    angles = np.arctan2(*newvecs.T[::-1])
    newvecs = newvecs[np.argsort(angles)]
    return newvecs[-symmetry//2:]