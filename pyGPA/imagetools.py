import numpy as np
import matplotlib.pyplot as plt

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
            ax.contour(fftim.T,colors='white', origin='lower', extent=extent, alpha=0.3, levels=levels)
    ax.set_aspect('equal')
    return im


def gauss_homogenize2(image, mask, sigma):
    VV = ndi.gaussian_filter(np.where(mask, image, 0), 
                             sigma=sigma)
    VV /= ndi.gaussian_filter(mask.astype(image.dtype), 
                              sigma=sigma)

    return image / VV

def gauss_homogenize3(image, mask, sigma):
    VV = ndi.gaussian_filter(np.where(mask, image, 0), 
                             sigma=sigma)
    VV /= ndi.gaussian_filter(mask.astype(image.dtype), 
                              sigma=sigma)
    VV = np.nan_to_num(VV, nan=1)
    return image / VV

def trim_nans(image):
    """Trim any rows and columns containing only nans from the image
    """
    xmask = np.all(np.isnan(image), axis=1)
    ymask = np.all(np.isnan(image), axis=0)
    if len(image.shape) >= 3:
        #Color channel handling
        xmask = np.any(xmask, axis=-1)
        ymask = np.any(ymask, axis=-1)
    return image[~xmask][:,~ymask]

def trim_nans2(image, return_lims=False):
    """Trim all outer rows and columns containing nans,
    preserving as much area as possible."""
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
