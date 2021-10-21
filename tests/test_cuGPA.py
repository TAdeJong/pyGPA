import numpy as np
import scipy.ndimage as ndi
from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

import latticegen

import pyGPA.geometric_phase_analysis as GPA
import pyGPA.cuGPA as cuGPA

try:
    import cupy
except ImportError:
    pytest.skip("skipping cupy tests as cupy could not be imported", allow_module_level=True)
    


@pytest.fixture(scope='module')
def gaussiandeform(size=500):
    S = size // 2
    xp, yp = np.meshgrid(np.arange(-S, S), np.arange(-S, S), indexing='ij')
    xshift = 0.5*xp*np.exp(-0.5*((xp/(2*S/8))**2 + 1.2*(yp/(2*S/6))**2))
    return np.stack((xshift, np.zeros_like(xshift)), axis=0)




@pytest.fixture(scope='module')
def testset_gaussian(gaussiandeform):
    r_k = 0.1
    xi0 = 7.0
    psi = 0.0
    kappa = 1.001
    order = 2
    S = 500
    original = latticegen.hexlattice_gen(r_k, xi0, order, size=S,
                                         kappa=kappa, psi=psi).compute()
    deformed = latticegen.hexlattice_gen(r_k, xi0, order, S, kappa=kappa, psi=psi,
                                         shift=gaussiandeform
                                         ).compute()
    # TODO: fix numpy random seed in a way compatible with both new and old numpy random interface
    noise = ndi.filters.gaussian_filter(5*np.random.normal(size=deformed.shape), sigma=0.5)
    ori_ks = latticegen.generate_ks(r_k, xi0, kappa=kappa, psi=psi)[:-1]
    return original, deformed, noise, ori_ks

@pytest.mark.parametrize("wfr_func", [cuGPA.wfr2_grad_opt, cuGPA.wfr2_grad_single])
def test_displacement_field(testset_gaussian, gaussiandeform, wfr_func):
    original, deformed, noise, ori_ks = testset_gaussian
    u = -GPA.extract_displacement_field(deformed + noise, ori_ks[:3], wfr_func=wfr_func)
    assert u.shape == gaussiandeform.shape
    print(np.abs(u - gaussiandeform)[:, 20:-20, 20:-20].max())
    assert np.all(np.abs(u - gaussiandeform)[:, 20:-20, 20:-20] < 0.9)
    u2 = -GPA.extract_displacement_field(deformed, ori_ks[:3], deconvolve=True)
    assert u2.shape == gaussiandeform.shape
    print(np.abs(u2 - gaussiandeform)[:, 20:-20, 20:-20].max())
    assert np.all(np.abs(u2 - gaussiandeform)[:, 20:-20, 20:-20] < 0.05)


def test_reconstruction(testset_gaussian, gaussiandeform):
    original, deformed, noise, ori_ks = testset_gaussian
    u_inv = GPA.invert_u_overlap(-gaussiandeform)
    assert u_inv.shape == gaussiandeform.shape
    reconstructed = GPA.undistort_image(deformed, gaussiandeform)
    assert np.all(np.abs(reconstructed - original) / np.abs(original).max() < 0.02)
    # Add optical flow if feeling fancy, but parameters are hard to optimize.
