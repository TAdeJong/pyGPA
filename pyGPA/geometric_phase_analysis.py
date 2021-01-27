import numpy as np
import matplotlib.pyplot as plt
import os
import dask.array as da
import scipy.ndimage as ndi
from scipy.optimize import least_squares
from moisan2011 import per
from numba import njit, prange
from skimage.feature import peak_local_max
from .phase_unwrap import phase_unwrap, phase_unwrap_ref_prediff
#from .imagetools import gauss_homogenize2, fftplot, fftbounds trim_nans2
from .imagetools import gauss_homogenize2, fftbounds, fftplot
from .mathtools import fit_plane, periodic_average, wrapToPi
from itertools import combinations
from latticegen.transformations import rotate


def GPA(image, kx, ky, sigma=22):
    xx, yy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    multiplier = np.exp(np.pi*2*1j*(xx*kx + yy*ky))
    X = np.fft.fft2(image*multiplier)
    res = np.fft.ifft2(ndi.fourier_gaussian(X, sigma=sigma))
    return res

def optGPA(image, kvec, sigma=22):
    xx, yy = np.ogrid[0:image.shape[0],0:image.shape[1]]
    multiplier = np.exp(np.pi*2*1j*(xx*kvec[0] + yy*kvec[1]))
    X = np.fft.fft2(image*multiplier)
    res = np.fft.ifft2(ndi.fourier_gaussian(X, sigma=sigma))
    return res

def vecGPA(image, kvecs, sigma=22):
    """Vectorized GPA, taking a list of kvecs"""
    #TODO: maybe better make a gufunc out of this?
    xx, yy = np.ogrid[0:image.shape[-2],0:image.shape[-1]]
    multiplier = np.exp(np.pi*2*1j*(xx*kvecs.T[0,:, None, None] + yy*kvecs.T[1,:, None, None]))
    X = np.fft.fft2(image*multiplier)
    res = np.fft.ifft2(ndi.fourier_gaussian(X, sigma=sigma))
    return res



def fit_delta_k(phases):
    x_opt = fit_plane(phases)
    return x_opt[:2]/(2*np.pi)

@njit()
def myweighed_lstsq(b, K, w):
    res = np.empty((2,) + b.shape[1:])
    for i in prange(b.shape[1]):
        for j in prange(b.shape[2]):
            wloc = w[:, i, j]
            temp = np.linalg.lstsq((wloc * K.T).T, wloc*b[:,i,j])[0]
            res[:, i, j] = temp
    return res

def iterate_GPA(image, kvecs, sigma, edge=5, iters=3, kmax_iter=25, kmax=200, verbose=False):
    corr = np.zeros_like(kvecs)
    for i in range(iters + 1):
        rs = np.stack([GPA(image, *ks, sigma=sigma) for ks in kvecs + corr])
        if edge > 0:
            prs = [np.angle(r)[edge:-edge,edge:-edge] for r in rs]
            w = np.stack([np.abs(r)[edge:-edge,edge:-edge] for r in rs])
        else:
            prs = [np.angle(r) for r in rs]
            w = np.stack([np.abs(r) for r in rs])
        if i < iters:
            prs = [phase_unwrap(r, np.sqrt(we/we.max()), 
                                kmax=kmax_iter) 
                   for r,we in zip(prs,w)]
            delta_ks = np.stack([fit_delta_k(pr) for pr in prs])
            if verbose:
                print(delta_ks)
            corr -= delta_ks
        else:
            prs = [phase_unwrap(r, np.sqrt(we/we.max()), 
                                kmax=kmax) 
                   for r,we in zip(prs,w)]
            prs = np.stack(prs)
        
    return prs, w, corr

def reconstruct_u_inv(kvecs, b, weights=None, use_only_ks=None):
    """reconstruct the distortion field u from the phase shift
    along the kvecs."""
    K = 2*np.pi*(kvecs)
    b = b - b.mean(axis=(1,2), keepdims=True)
    if use_only_ks is None:
        if weights is None:
            lstsqres = np.linalg.lstsq(K, b.reshape((3,-1)), rcond=None)
            us = lstsqres[0].reshape((2,)+b[0].shape)
        else:
            us = myweighed_lstsq(b, K, weights)
    else:
        assert len(use_only_ks) == 2
        us = np.linalg.inv(K[use_only_ks]) @ b[use_only_ks].reshape((2,-1))
        us = us.reshape((2,)+b[0].shape)
    return us

def reconstruct_u_inv_from_phases(kvecs, phases, weights, weighted_unwrap=True):
    """Reconstruct the distortion field u from the wrapped phase shifts
    along the kvecs. By first projecting to cartesian vectors and only
    afterwards phase unwrapping.
    """
    K = 2*np.pi*(kvecs)
    #b = b - b.mean(axis=(1,2), keepdims=True)
    dbdx = wrapToPi(np.diff(phases, axis=2))
    dbdy = wrapToPi(np.diff(phases, axis=1))
    dudx = myweighed_lstsq(dbdx, K, weights)
    dudy = myweighed_lstsq(dbdy, K, weights)
    if weighted_unwrap:
        us = [phase_unwrap_ref_prediff(dudx[i], dudy[i], np.linalg.norm(weights, axis=0)) for i in range(2)]
    else:
        us = [phase_unwrap_ref_prediff(dudx[i], dudy[i]) for i in range(2)]
    return np.array(us)


def invert_u(us, iters=35, edge=0, mode='nearest'):
    """Find the inverse of the displacement u such that:
    \vec u_it(\vec r+\vec us(\vec r)) = \vec r
    i.e.:
    If an Image has been distorted by sampling at \vec r+ \vec us(\vec r),
    sampling that image at u_it will sample the original image.
    """
    xx, yy = np.mgrid[:us.shape[1], :us.shape[2]]
    u_it = [ndi.map_coordinates(u, [xx,yy], mode=mode) for u in us]
    for i in range(iters):
        u_it = [ndi.map_coordinates(u, [xx+u_it[0]-edge, yy+u_it[1]-edge], mode=mode) for u in us]
    return np.stack(u_it)

def invert_u_overlap(us, iters=35, edge=0, mode='nearest'):
    """Find the inverse of the displacement u such that:
    \vec u_it(\vec r+\vec us(\vec r)) = \vec r
    i.e.:
    If an Image has been distorted by sampling at \vec r+ \vec us(\vec r),
    sampling that image at u_it will sample the original image.
    """
    xx, yy = np.mgrid[-edge:us.shape[1]+edge, -edge:us.shape[2]+edge]
    u_it = [ndi.map_coordinates(u, [xx,yy], mode=mode) for u in us]
    for i in range(iters-1):
        u_it = [ndi.map_coordinates(u, [xx+u_it[0], yy+u_it[1]], mode=mode) for u in us]
    u_it = [ndi.map_coordinates(u, [xx+u_it[0], yy+u_it[1]], mode=mode, cval=np.nan) for u in us]
    return np.stack(u_it)


def average_lattice_vector(ks, symmetry=6):
    dt = periodic_average(np.arctan2(*ks.T), period=2*np.pi/symmetry)
    r = np.linalg.norm(ks, axis=1).mean()
    return r * np.array([np.sin(dt), np.cos(dt)])

def calc_diff_from_isotropic(ani_ks):
    k_hex = average_lattice_vector(ani_ks)
    ks_hex = np.array([rotate(k_hex, i*np.pi/3) for i in range(6)])
    alldiffs =  ks_hex - ani_ks[:,None]
    argmins = np.linalg.norm(alldiffs, axis=-1).argmin(axis=1)
    return alldiffs[np.arange(len(ani_ks)), argmins]


def prep_image(original, vlims=None, edges=None):
    """prep image for usage in GPA"""
    if vlims is None:
        vlims = np.quantile(original, [0.08, 0.999])
    if edges is not None:
        original = original[edges[0,0]:edges[0,1], edges[1,0]:edges[1,1]]
    else:
        original = trim_nans2(np.where(original==0, np.nan, original))
    original = np.clip(original, *vlims)
    mask = np.logical_and(original > np.quantile(original, 0.01), original < np.quantile(original,0.99))
    deformed1 = gauss_homogenize2(original, mask, sigma=5)

    mask2 = ndi.gaussian_filter(deformed1, sigma=5) > 0.995
    deformed2 = gauss_homogenize2(original, mask2, sigma=65)
    deformed = deformed2-deformed2.mean()
    xx, yy = np.meshgrid(np.arange(original.shape[0]), np.arange(original.shape[1]), indexing='ij')
    return deformed, xx, yy

def ratio2angle(R):
    """Given a ratio between unit cell sizes R < 1,
    return the corresponding angle in degrees for 
    $ \theta = 2 \arcsin(R/2) $"""
    return np.rad2deg(2*np.arcsin(R/2))

def f2angle(f, nmperpixel=1., a_0=0.246):
    """For a given lin frequency f (==2*pi*|k|),
    calculate the corresponding twist angle.
    """
    graphene_linespacing = 0.5*np.sqrt(3)*a_0
    linespacing = nmperpixel/f
    return ratio2angle(graphene_linespacing / linespacing)

def remove_negative_duplicates(ks):
    """For a list of length 2 vectors ks (or Nx2 array),
    return the array of vectors without negative duplicates
    of the vectors (x-coord first, if zero non-negative y)
    """
    if ks.shape[0] == 0:
        return ks
    npks = []
    nonneg = np.where(np.sign(ks[:,[0]]) != 0, np.sign(ks[:,[0]])*ks, np.sign(ks[:,[1]])*ks)
    npks = [nonneg[0]]
    atol = 1e-5*np.linalg.norm(nonneg, axis=1).mean()
    for k in nonneg[1:]:
        if not np.any(np.isclose(k, npks, atol=atol)):
            npks.append(k)
    return np.array(npks)

def _decrease_threshold(t):
    if t > 0.001:
        if t >= 0.2:
            t = t-0.1
        else:
            t = t/2
    return t

def extract_primary_ks(image, plot=False, threshold=0.7, pix_norm_range=(20,200), sigma=1.5, NMPERPIXEL=1.):
    """Attempt to extract primary k-vectors from an image from a smoothed
    version of the Fourier transform.
    """
    image = image - image.mean()
    pd, _ = per(image, inverse_dft=False)
    fftim = np.abs(np.fft.fftshift(pd))
    kxs, kys = [fftbounds(n) for n in fftim.shape]
    smooth = ndi.filters.gaussian_filter(fftim, sigma=sigma) - ndi.filters.gaussian_filter(fftim, sigma=50)

    center = np.array(smooth.shape)//2
    cindices = peak_local_max(smooth, threshold_rel=threshold)#min_distance=5, threshold_rel=np.quantile(smooth, threshold))
    coords = cindices - center
    selection = np.logical_and((np.linalg.norm(coords, axis=1) < pix_norm_range[1]), 
                                       (np.linalg.norm(coords, axis=1) > pix_norm_range[0]))
    cindices = cindices[selection]
    coords = coords[selection] #exclude low intensity edge area and center stigmated dirt spots
    
    #coords = np.vstack((coords, [0,0])) # reinclude center spot
    all_ks = np.array([kxs[cindices.T[0]], kys[cindices.T[1]]]).T
    # Select only one direction for each pair of k,-k
    all_ks = remove_negative_duplicates(all_ks)
    newparams = False    
    if len(all_ks) < 3:
        newparams = True
        if len(all_ks) == 0:
            print(f"no ks at: {threshold:.4f}")
            if threshold > _decrease_threshold(threshold):
                threshold = _decrease_threshold(threshold)
            else:
                print("No ks found at minimum threshold!")
                newparams = False
        else:
            coordsminlength = np.linalg.norm(coords, axis=1).min()
            print(f"cminlength {coordsminlength:.2f}, s {sigma:.1f}")
            if coordsminlength < 5 * sigma:
                sigma = coordsminlength / 6
                print(f"cminlength {coordsminlength:.2f}")
            elif threshold > 0.2*np.max([smooth[cindex[0],cindex[1]] for cindex in cindices]):
                threshold = 0.2*np.max([smooth[cindex[0],cindex[1]] for cindex in cindices])
            elif threshold > _decrease_threshold(threshold):
                    threshold = _decrease_threshold(threshold)
                    print(f"new thres: {threshold:.2f}")
            else:
                print("Can't find enough ks!")
                newparams = False
        if newparams:
            primary_ks, all_ks = extract_primary_ks(image, plot=False, threshold=threshold, 
                                                    sigma=sigma,
                                                    pix_norm_range=pix_norm_range)
        else:
            primary_ks = all_ks.copy()
            
    knorms = np.linalg.norm(all_ks, axis=1)
    if not newparams:    
        primary_ks = all_ks.copy()
    
    if len(primary_ks) != 3:
        if len(primary_ks) > 3:
            print(f"Too many primary ks {len(primary_ks)}")
            primary_ks = select_closest_to_triangle(all_ks)
        elif len(all_ks) > 6:
            #print("all_ks > 3, selecting 3 with most similar length")
            #primary_ks = all_ks[np.argpartition(np.abs(knorms-knorms.mean()), 3)[:3]]
            print("all_ks > 3 but not enough primary_ks, selecting closest to triangle")
            primary_ks = select_closest_to_triangle(all_ks)
        elif threshold > _decrease_threshold(threshold) and not newparams: 
            print(f"pks<3, all_ks < 6, decreasing threshold {threshold:.3f}")
            threshold = _decrease_threshold(threshold)
            primary_ks, all_ks = extract_primary_ks(image, plot=False, threshold=threshold, 
                                        sigma=sigma,
                                        pix_norm_range=pix_norm_range)
        else:
            print("pks < aks=3", len(all_ks), len(primary_ks))
            primary_ks = all_ks.copy()
    if plot:
        fig,ax = plt.subplots(ncols=2,figsize=[12,8])
        fftplot(smooth, d=NMPERPIXEL, ax=ax[0], levels=[smooth.max()*threshold*0.8], contour=False, pcolormesh=False) # , vmin=(smooth.max()*threshold*0.1)
        ax[0].set_xlabel('k (periods / nm)')
        ax[0].set_ylabel('k (periods / nm)')
        ax[0].scatter(*(all_ks/NMPERPIXEL).T, color='red', alpha=0.2, s=50)
        ax[0].scatter(*(primary_ks/NMPERPIXEL).T, color='black', alpha=0.7, s=50, marker='x')
        
        circle = plt.Circle((0, 0), 2.*knorms.min()/NMPERPIXEL, edgecolor='y', fill=False, alpha=0.6)
        ax[0].add_artist(circle)
        ax[1].imshow(image.T)
        for r in [kxs[center[0]+s] for s in pix_norm_range]:
            circle = plt.Circle((0, 0), r/NMPERPIXEL, edgecolor='w', fill=False, alpha=0.6)
            ax[0].add_artist(circle)
        plt.title(plot)
    return primary_ks, all_ks


def select_closest_to_triangle(ks):
    """Select the 3 ks that come closest to a triangle"""
    print(f"closest triangle: {ks}")
    combis = list(combinations(ks,3))
    sums = [np.linalg.norm(smallest_sum(combi)) for combi in combis]
    return np.array(combis[np.argmin(sums)])
        

def smallest_sum(ks):
    """For a set of 3 k-vectors, return
    the smallest possible sum of the three
    with one sign flipped.
    """
    if len(ks) != 3:
        return np.nan
    M = np.ones((3,3)) - 2*np.eye(3)
    sums = M@ks
    return sums[np.argmin(np.linalg.norm(sums,axis=1))]
    
    
def wff(image, sigma, threshold, wl, wu, verbose=False):
    """Windowed Fourier Filtering of image.
    with gaussian window width sigma,
    filter such that all frequencies above threshold
    between frequency boundaries wl and wu are retained
    The scheme is shown inFig. 3. A fringepattern is transforme
    """
    s = round(2*sigma)
    yy,xx = np.mgrid[-s:s,-s:s]
    w = np.exp(-(xx**2+yy**2)/(2*sigma**2))
    w = w/np.sqrt((w**2).sum())
    gs = np.stack([np.zeros_like(image)]*len(threshold))
    wi = 1/sigma
    for wx in np.arange(wl, wu+wi/2, wi):
        print("wx", wx)
        for wy in np.arange(wl, wu+wi/2, wi):
            wave = w * np.exp(1j*(wx*xx+wy*yy))
            sf = ndi.convolve(image, wave)
            if verbose:
                print(np.quantile(np.abs(sf), [0.9,0.99]), 
                      [(np.abs(sf)>=thri).sum()for thri in threshold])
            for i,g in enumerate(gs):
                sfi = np.where(np.abs(sf)>=threshold[i], sf, 0.)
                g += ndi.convolve(sfi, wave)
    gs *= wi*wi/(4*np.pi**2)
    return gs


def wfr(image, sigma, kx, ky, kw, kstep):
    """Adaptive GPA. Find the phase corresponding to 
    kx,ky k-vector by using the reference vector
    in the square 2*kw around it which has the highest lockin amplitude.
    return both lockin amplitude and phase as well as the used kvector
    in a dictionary structure.
    Based directly on original MATLAB algorithm in https://doi.org/10.1016/j.optlaseng.2005.10.012"""
    s = round(2*sigma)
    wyy, wxx = np.mgrid[-s:s,-s:s]
    xx, yy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    w = np.exp(-(wxx**2 + wyy**2) / (2*sigma**2))
    w = w / np.sqrt((w**2).sum())
    g = {'wx': np.zeros_like(image),
         'wy': np.zeros_like(image),
        'phase': np.zeros_like(image),
        'r': np.zeros_like(image),
        }
    for wx in np.arange(kx-kw,kx+kw, kstep):
        for wy in np.arange(ky-kw,ky+kw, kstep):
            sf = GPA(image, wx, wy, sigma)
            sf *= np.exp(-2j*np.pi*((wx-kx)*xx+(wy-ky)*yy))
            t = np.abs(sf) > g['r']
            g['r'][t] = np.abs(sf)[t]
            g['wx'][t] = wx
            g['wy'][t] = wy
            g['phase'][t] = np.angle(sf)[t]
    return g
    
def wfr2(image, sigma, kx, ky, kw, kstep):
    """Adaptive GPA. Find the phase corresponding to 
    kx,ky k-vector by using the reference vector
    in the square 2*kw around it which has the highest lockin amplitude.
    return the used k-vector as well as the complex lock-in signal.
    """
    s = round(2*sigma)
    wyy,wxx = np.mgrid[-s:s,-s:s]
    xx, yy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    w = np.exp(-(wxx**2+wyy**2)/(2*sigma**2))
    w = w/np.sqrt((w**2).sum())
    g = {'w': np.zeros(image.shape+(2,)), 
        'lockin': np.zeros_like(image, dtype=np.complex),
        }
    for wx in np.arange(kx-kw,kx+kw, kstep):
        for wy in np.arange(ky-kw,ky+kw, kstep):
            sf = GPA(image, wx, wy, sigma)
            sf *= np.exp(-2j*np.pi*((wx-kx)*xx+(wy-ky)*yy))
            t = np.abs(sf) > np.abs(g['lockin'])
            g['lockin'][t] = sf[t]
            g['w'][t] = np.array([wx,wy])
    g['w'] = np.moveaxis(g['w'],-1,0)
    return g

def wfr3(image, sigma, klist, kref):
    """Iterate over klist, calculate GPA of image for each k, with sigma width
    accept new value if lockin amplitude is larger.
    Compensate phase to be relative to kref"""  
    s = round(2*sigma)
    wyy,wxx = np.mgrid[-s:s,-s:s]
    xx, yy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    w = np.exp(-(wxx**2+wyy**2)/(2*sigma**2))
    w = w/np.sqrt((w**2).sum())
    g = {'w': np.zeros(image.shape+(2,)), 
        'lockin': np.zeros_like(image, dtype=np.complex),
        }
    for wx,wy in klist:
        sf = GPA(image, wx, wy, sigma)
        sf *= np.exp(-2j*np.pi*((wx-kref[0])*xx+(wy-kref[1])*yy))
        t = np.abs(sf) > np.abs(g['lockin'])
        g['lockin'][t] = sf[t]
        g['w'][t] = np.array([wx,wy])
    g['w'] = np.moveaxis(g['w'],-1,0)
    return g



def optwfr2(image, sigma, kx, ky, kw, kstep):
    """Optimized version of wfr2.
    Optimization in amount of computation done in each step by only
    computing updated values."""
    s = round(2*sigma)
    wyy,wxx = np.mgrid[-s:s,-s:s]
    xx, yy = np.ogrid[0:image.shape[0],0:image.shape[1]]
    w = np.exp(-(wxx**2+wyy**2)/(2*sigma**2))
    w = w/np.sqrt((w**2).sum())
    g = {'w': np.zeros(image.shape+(2,)), 
        'lockin': np.zeros_like(image, dtype=np.complex),
        }
    for wx in np.arange(kx-kw,kx+kw, kstep):
        for wy in np.arange(ky-kw,ky+kw, kstep):
            sf = optGPA(image, (wx, wy), sigma)
            t = np.abs(sf) > np.abs(g['lockin'])
            g['lockin'][t] = sf[t] * np.exp(-2j*np.pi*((wx-kx)*xx+(wy-ky)*yy)[t])
            g['w'][t] = np.array([wx,wy])
    g['w'] = np.moveaxis(g['w'],-1,0)
    return g




def wfr2_grad(image, sigma, kx, ky, kw, kstep):
    """Adapted version of wfr2. In addition to returning the 
    used k-vector and lock-in signal, return the gradient of the lock-in
    signal as well, for each pixel computed from the values of the surrounding pixels
    of the GPA of the best k-vector. Slightly more accurate, determination of this gradient,
    as boundary effects are mitigated"""
    s = round(2*sigma)
    wyy,wxx = np.mgrid[-s:s,-s:s]
    xx, yy = np.ogrid[0:image.shape[0],0:image.shape[1]]
    w = np.exp(-(wxx**2+wyy**2)/(2*sigma**2))
    w = w/np.sqrt((w**2).sum())
    g = {'w': np.zeros(image.shape+(2,)), 
        'lockin': np.zeros_like(image, dtype=np.complex),
         'grad': np.zeros(image.shape+(2,)), 
        }
    for wx in np.arange(kx-kw,kx+kw, kstep):
        for wy in np.arange(ky-kw,ky+kw, kstep):
            sf = optGPA(image, (wx, wy), sigma)
            sf *= np.exp(-2j*np.pi*((wx-kx)*xx+(wy-ky)*yy))
            grad = wrapToPi(np.stack(np.gradient(-np.angle(sf)), axis=-1)*2)/4/np.pi
            t = np.abs(sf) > np.abs(g['lockin'])
            g['lockin'][t] = sf[t]
            g['w'][t] = np.array([wx,wy])
            g['grad'][t] = grad[t]
    g['w'] = np.moveaxis(g['w'],-1,0)
    print(grad.shape, np.array(np.gradient(-np.angle(sf))).shape, g['grad'].shape)
    return g

def wfr2_grad_opt(image, sigma, kx, ky, kw, kstep):
    """Optimized version of wfr2_grad. In addition to returning the 
    used k-vector and lock-in signal, return the gradient of the lock-in
    signal as well, for each pixel computed from the values of the surrounding pixels
    of the GPA of the best k-vector. Slightly more accurate, determination of this gradient,
    as boundary effects are mitigated"""
    s = round(2*sigma)
    wyy,wxx = np.mgrid[-s:s,-s:s]
    xx, yy = np.ogrid[0:image.shape[0],0:image.shape[1]]
    w = np.exp(-(wxx**2+wyy**2)/(2*sigma**2))
    w = w/np.sqrt((w**2).sum())
    g = {'w': np.zeros(image.shape+(2,)), 
        'lockin': np.zeros_like(image, dtype=np.complex),
         'grad': np.zeros(image.shape+(2,)), 
        }
    for wx in np.arange(kx-kw,kx+kw, kstep):
        for wy in np.arange(ky-kw,ky+kw, kstep):
            sf = optGPA(image, (wx, wy), sigma)
            t = np.abs(sf) > np.abs(g['lockin'])
            grad =  np.stack(np.gradient(-np.angle(sf)), axis=-1)[t]
            g['lockin'][t] = sf[t] * np.exp(-2j*np.pi*((wx-kx)*xx+(wy-ky)*yy)[t])
            g['w'][t] = np.array([wx,wy])
            g['grad'][t] = grad + 2*np.pi*(np.stack([(wx-kx),(wy-ky)],axis=-1))
    g['w'] = np.moveaxis(g['w'],-1,0)
    g['grad'] = wrapToPi(g['grad']*2)/2
    return g

def wfr2_grad_vec(image, sigma, kx, ky, kw, kstep):
    """Vectorized version of wfr2_grad_opt. vectorize using dask"""
    s = round(2*sigma)
    wyy,wxx = np.mgrid[-s:s,-s:s]
    xx, yy = np.ogrid[0:image.shape[0],0:image.shape[1]]
    w = np.exp(-(wxx**2+wyy**2)/(2*sigma**2))
    w = w/np.sqrt((w**2).sum())
    g = {'w': np.zeros(image.shape+(2,)), 
        'lockin': np.zeros_like(image, dtype=np.complex),
         'grad': np.zeros(image.shape+(2,)), 
        }
    for wx in np.arange(kx-kw,kx+kw, kstep):
        wys = np.arange(ky-kw,ky+kw, kstep)
        wpairs = da.stack(([wx]*len(wys), wys),axis=-1).rechunk({0:3})
        sf = vecGPA(image, wpairs, sigma)
        for i,wy in enumerate(wys):
            t = np.abs(sf[i]) > np.abs(g['lockin'])
            grad =  np.stack(np.gradient(-np.angle(sf[i])), axis=-1)[t]
            g['lockin'][t] = sf[i][t] * np.exp(-2j*np.pi*((wx-kx)*xx+(wy-ky)*yy)[t])
            g['w'][t] = np.array([wx,wy])
            g['grad'][t] = grad + 2*np.pi*(np.stack([(wx-kx),(wy-ky)],axis=-1))
    g['w'] = np.moveaxis(g['w'],-1,0)
    g['grad'] = wrapToPi(g['grad']*2)/2
    return g

def wfr4(image, sigma, klist, kref, dk):
    """Iterate over klist, calculate GPA of image for each k, with sigma width
    accept new value if lockin amplitude is larger and new k is maximum 2 lattice positions
    dk away from old k. Only makes sense if klist is ordered.
    Compensate phase to be relative to kref"""    
    s = round(2*sigma)
    wyy,wxx = np.mgrid[-s:s,-s:s]
    xx, yy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    w = np.exp(-(wxx**2+wyy**2)/(2*sigma**2))
    w = w/np.sqrt((w**2).sum())
    g = {'w': np.zeros(image.shape+(2,)), 
        'lockin': np.zeros_like(image, dtype=np.complex),
        }
    g['w'][...,0] = klist[0,0]
    g['w'][...,1] = klist[0,1]
    for wx,wy in klist:
        sf = GPA(image, wx, wy, sigma)
        sf *= np.exp(-2j*np.pi*((wx-kref[0])*xx+(wy-kref[1])*yy))
        t = np.abs(sf) > np.abs(g['lockin'])
        t = t & (np.linalg.norm(g['w'] - np.array([wx,wy]),axis=-1) < 2*np.sqrt(2)*dk)
        g['lockin'][t] = sf[t]
        g['w'][t] = np.array([wx,wy])
    g['w'] = np.moveaxis(g['w'],-1,0)
    return g


def generate_klists(pks, dk=None, kmax=1.9, kmin=0.2, sort_list=False):
    """From a list of k-vectors pks, determine for each k-vector a list
    of k-vectors, spaced dk, with at max `kmax` times the max k-vector amplitude
    and at least `kmin` times that, where each vector in the list is not closer
    to any other kvec in pks.
    Used in conjunction with wfr3 or wfr4
    """
    doubleks = np.concatenate([pks,-pks])
    kmax = np.linalg.norm(pks, axis=1).max()*kmax
    kmin = np.linalg.norm(pks, axis=1).max()*kmin
    if dk is None:
        dk = np.linalg.norm(pks, axis=1).mean()/10
    kk = np.mgrid[-kmax:kmax:0.005,-kmax:kmax:0.005]
    dists = ((np.moveaxis(kk[...,None],0,-1)-doubleks)**2).sum(axis=-1)
    r = (kk**2).sum(axis=0)
    kmask0 = (r < kmax**2) & (r > kmin**2)
    klists = []
    for i,pk in enumerate(pks):
        kmask = kmask0 & (dists.min(axis=-1) == dists[...,i])
        klist = kk[:,kmask].T
        if sort_list:
            ampl = np.linalg.norm(klist-pks[i], axis=1)
            klist = klist[np.argsort(ampl.reshape((-1)))]
        klists.append(klist)
    return klists


def calc_props_eps(U, nmperpixel, Uinv=None, edge=0):
    if Uinv is None:
        Uinv = invert_u_overlap(uw, edge)
    J = np.stack(np.gradient(-U, nmperpixel, axis=(1,2)), axis=1)
    J = np.moveaxis(J,(0,1), (-2,-1))
    J = (np.eye(2) + J)
    u,s,v = np.linalg.svd(J)
    angle = (u@v)
    moireangle = phase_unwrap(np.arctan2(angle[...,1,0], angle[...,0,0])) #unwrap for proper interpolation
    aniangle = phase_unwrap(2*np.arctan2(v[...,1,0], v[...,0,0])) #unwrap for proper interpolation, factor 2 for degeneracy
    xxh, yyh = np.mgrid[-edge:U.shape[1]+edge, -edge:U.shape[2]+edge]
    print(xxh.shape, Uinv.shape, moireangle.shape)
    eps =  (s[...,0] - s[...,1])/(s[...,1] + s[...,0]*0.17)
    alpha = s[...,1] * (1+eps)
    props = [ndi.map_coordinates(p.T, [xxh+Uinv[0], yyh+Uinv[1]], mode='constant', cval=np.nan) 
             for p in [moireangle, aniangle, alpha, eps]]
    moireangle, aniangle, alpha, eps = props
    moireangle = np.rad2deg(moireangle )
    aniangle = np.rad2deg(aniangle )/2
    return moireangle, aniangle, alpha, eps
    
def calc_props(U, nmperpixel):
    """From the displacement field U, calculate the following properties:
    -local angle w.r.t. absolute reference
    -local direction of the anisotropy
    -local unit cell scaling factor
    -local anisotropy magnitude
    """
    J = np.stack(np.gradient(-U, nmperpixel, axis=(1,2)), axis=-1)
    J = np.moveaxis(J, 0, -2)
    J = (np.eye(2) + J)
    u,s,v = np.linalg.svd(J)
    angle = (u@v)
    moireangle = np.rad2deg(np.arctan2(angle[...,1,0], angle[...,0,0]))
    aniangle = np.rad2deg(np.arctan2(v[...,1,0], v[...,0,0])) % 180
    return moireangle, aniangle, np.sqrt(s[...,0]* s[...,1]), s[...,0] / s[...,1]

def calc_props_from_phases(kvecs, phases, weights, nmperpixel):
    """Calculate properties from phases."""
    K = 2*np.pi*(kvecs)
    dbdx , dbdy = wrapToPi(np.stack(np.gradient(phases, axis=(1,2)))*2)/2/nmperpixel
    #dbdy = wrapToPi(np.diff(phases, axis=1))
    dudx = myweighed_lstsq(dbdx, K, weights)
    dudy = myweighed_lstsq(dbdy, K, weights)
    J = -np.stack([dudx,dudy], axis=-1)
    J = np.moveaxis(J, 0, -2)
    J = (np.eye(2) + J)
    u,s,v = np.linalg.svd(J)
    angle = (u@v)
    moireangle = np.rad2deg(np.arctan2(angle[...,1,0], angle[...,0,0]))
    aniangle = np.rad2deg(np.arctan2(v[...,1,0], v[...,0,0])) % 180
    return moireangle, aniangle, np.sqrt(s[...,0]* s[...,1]), s[...,0] / s[...,1]

def calc_props_from_phasegradient(kvecs, grads, weights, nmperpixel):
    """Calculate properties directly from phase gradients.
    Might not yet properly take nmperpixel into account"""
    K = 2*np.pi*(kvecs)
    #b = b - b.mean(axis=(1,2), keepdims=True)
    #dbdx = wrapToPi(np.diff(phases, axis=2))
    #dbdy = wrapToPi(np.diff(phases, axis=1))
    #TODO: make a nice reshape for this call.
    dudx = myweighed_lstsq(grads[...,0], K, weights)
    dudy = myweighed_lstsq(grads[...,1], K, weights)
    #dudx = myweighed_lstsq(grad[:,0], K, weights)
    #dudy = myweighed_lstsq(grad[:,1], K, weights)
    J = np.stack([dudx,dudy], axis=-1) #*nmperpixel?
    J = np.moveaxis(J, 0, -2)
    J = (np.eye(2) + J)
    u,s,v = np.linalg.svd(J)
    angle = (u@v)
    moireangle = np.rad2deg(np.arctan2(angle[...,1,0], angle[...,0,0]))
    aniangle = np.rad2deg(np.arctan2(v[...,1,0], v[...,0,0])) % 180
    return moireangle, aniangle, np.sqrt(s[...,0]* s[...,1]), s[...,0] / s[...,1]


