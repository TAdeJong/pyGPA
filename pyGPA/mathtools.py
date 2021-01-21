import numpy as np

def periodic_average(X, period=2*np.pi):
    Y = np.exp(1j * 2*np.pi / period * X)
    Y = np.angle(Y.mean())
    return Y * period / (2*np.pi)
    
def lfit_func(x, image, xx, yy):
    ax, ay, b = x
    return (image - (ax*xx + ay*yy + b)).flatten()

def lfit_func_mask(x, image, xx, yy, mask):
    ax, ay, b = x
    return (image - (ax*xx + ay*yy + b))[mask].flatten()

def fit_plane(image, verbose=False):
    lxx, lyy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    x0 = np.zeros(3)
    res = least_squares(lfit_func, x0, 
                        loss='huber', 
                        args=(image, lxx, lyy))
    if verbose:
        print(res.message)
    return res.x

def fit_plane_masked(image, verbose=False, mask=False):
    lxx, lyy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    x0 = np.zeros(3)
    if mask:
        res = least_squares(lfit_func_mask, x0, 
                            loss='huber', 
                            args=(image, lxx, lyy, mask))
    else:
        res = least_squares(lfit_func, x0, 
                            loss='huber', 
                            args=(image, lxx, lyy))
    if verbose:
        print(res.message)
    return res.x
    
def wrapToPi(x):
    """Wrap all values of x to the interval -pi,pi"""
    r = (x+np.pi)  % (2*np.pi) - np.pi
    return r
