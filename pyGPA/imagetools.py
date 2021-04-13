import collections

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.morphology import selem, disk

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def fftbounds(n, d=1):
    """Return the frequency edges for use with pcolormesh or similar"""
    r = np.fft.fftshift(np.fft.fftfreq(n, d))
    r = np.append(r, r[-1] + 1/(n*d))# - 1/(2*n*d)
    return r

def fftplot(fftim, d=1, pcolormesh=True, contour=False, levels=None, **kwargs):
    """Plot a Fourier transformed image with
    correct aspect ratio and axis.
    `d` is the pixel distance.
    A keyword ax= can be added to plot on an
    existing axis.
    Any other kwargs are forwarded to the call to 
    `pcolormesh`"""
    x,y = [fftbounds(n,d) for n in fftim.shape]
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else: 
        fig, ax = plt.subplots()
    if pcolormesh:
        X,Y = np.meshgrid(x,y, indexing='xy')
        im = ax.pcolormesh(X,Y, fftim.T, **kwargs)
    else:
        extent = [x[0], x[-1], y[0], y[-1]]
        im = ax.imshow(fftim.T, extent=extent, origin='lower', **kwargs)
        if contour:
            ax.contour(fftim.T,colors='white', origin='lower', 
                    extent=extent, alpha=0.3, levels=levels)
    ax.set_aspect('equal')
    return im

def indicate_k(pks, i, ax=None, inset=True, size="25%", origin='upper'):
    """Indicate the i-th vector in the list of vectors pks with an arrow
    and highlight in a scatterplot.
    If inset=True (default), create a new inset axis in ax.
    Return the axis in which the vectors have been drawn.
    """
    ks = pks.copy()
    if not ax:
        ax = plt.gca()
    if inset:
        ax = inset_axes(ax, width="25%", height="25%", loc=2)
        ax.tick_params(labelleft=False, labelbottom=False, 
                       direction='in', length=0)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_color(f"None")
        ax.patch.set_alpha(0.0)
    if origin == 'upper':
        ks[:,1] *= -1
    ax.scatter(*np.concatenate([ks, -ks, [[0,0]]]).T, color='gray', s=10)
    ax.scatter(ks[i,0], ks[i,1], color='red', s=30)
    if isinstance(i, collections.Iterable):
        for j in i:
            ax.arrow(0,0, ks[j,0], ks[j,1], length_includes_head=True)
    else:
        ax.arrow(0,0, ks[i,0], ks[i,1], length_includes_head=True)
    ax.set_aspect('equal')
    return ax


def gauss_homogenize2(image, mask, sigma, nan_scale=None):
    """Homogenize image by dividing by a 
    Gaussian filtered version of image, ignoring
    areas where mask is False.
    If nan_scale is given, scale all values not covered
    by the masked image by nan_scale.
    """
    VV = ndi.gaussian_filter(np.where(mask, image, 0), 
                             sigma=sigma)
    VV /= ndi.gaussian_filter(mask.astype(image.dtype), 
                              sigma=sigma)
    if nan_scale is not None:
        VV = np.nan_to_num(VV, nan=nan_scale)
    return image / VV

def gauss_homogenize3(image, mask, sigma):
    return gauss_homogenize2(image, mask, sigma, nan_scale=1)

def trim_nans(image):
    """Trim any rows and columns containing only nans from the image
    """
    xmask = np.all(np.isnan(image), axis=1)
    ymask = np.all(np.isnan(image), axis=0)
    if len(image.shape) >= 3:
        #Color channel handling
        if image.shape[-1] == 4:
            # alpha channel
            xmask = np.any(xmask[..., :3], axis=-1)
            ymask = np.any(ymask[..., :3], axis=-1)
        else:
            xmask = np.any(xmask, axis=-1)
            ymask = np.any(ymask, axis=-1)
    return image[~xmask][:,~ymask]

def trim_nans2(image, return_lims=False):
    """Trim all outer rows and columns containing nans,
    preserving as much area as possible.
    """
    timage = image.copy()
    xlims = [0, timage.shape[0]]
    ylims=[0, timage.shape[1]]
    stop = False
    while not stop:
        r = np.isnan(timage[[0,-1]]).sum(axis=1)
        c = np.isnan(timage[:, [0,-1]]).sum(axis=0)
        if r.sum() == 0 and c.sum() == 0:
            stop = True
            if return_lims:
                return timage, np.array([xlims, ylims])
            else:
                return timage
        elif r.sum() > c.sum():
            if r[0] > 0:
                timage = timage[1:]
                xlims[0] += 1
            if r[1] > 0:
                timage = timage[:-1]
                xlims[1] -= 1
        else:
            if c[0] > 0:
                timage = timage[:, 1:]
                ylims[0] += 1
            if c[1] > 0:
                timage = timage[:, :-1]
                ylims[1] -= 1


def generate_mask(dataset, mask_value, r=20):
    """Generate a boolean mask array covering everything that in 
    any image in dataset contains mask_value. Perform an
    erosion with radius r to create a safety margin."""
    mask = ~np.any(dataset == mask_value, axis=0).compute()
    mask = ndi.binary_erosion(mask, structure=disk(r))
    return mask


def to_KovesiRGB(image):
    """Convert to basis colors as suggested 
    by P. Kovesi in http://arxiv.org/abs/1509.03700
    """
    A = np.array([[0.90, 0.17, 0.00],
                 [0.00, 0.50, 0.00],
                 [0.10, 0.33, 1.00]])
    return np.dot(image, A)
