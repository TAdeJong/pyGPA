import numpy as np
import scipy.ndimage as ndi
from hypothesis import given, assume
import hypothesis.strategies as st
import pytest

import latticegen
from latticegen.transformations import rotation_matrix, scaling_matrix

import pyGPA.geometric_phase_analysis as GPA
from pyGPA.mathtools import periodic_difference, standardize_ks

@pytest.fixture(scope='module')
def gaussiandeform(size=500):
    S = size // 2
    xp, yp = np.meshgrid(np.arange(-S,S), np.arange(-S,S), indexing='ij')
    xshift = 0.5*xp*np.exp(-0.5*((xp/(2*S/8))**2 + 1.2*(yp/(2*S/6))**2))
    return np.stack((xshift, np.zeros_like(xshift)), axis=0)

# @pytest.fixture
# def randomdeform(size=500):
#     pass

@pytest.fixture(scope='module')
def testset_gaussian(gaussiandeform):
    r_k = 0.1
    xi0 = 7.0
    psi = 0.0
    kappa = 1.001
    order = 2
    S=500
    original = latticegen.hexlattice_gen(r_k, xi0, order, size=S, 
                                         kappa=kappa, psi=psi).compute()
    deformed = latticegen.hexlattice_gen(r_k, xi0, order, S, kappa=kappa, psi=psi,
                                         shift=gaussiandeform
                                         ).compute()
    noise = ndi.filters.gaussian_filter(10*np.random.normal(size=deformed.shape), sigma=0.5)
    ori_ks = latticegen.generate_ks(r_k, xi0, kappa=kappa, psi=psi)[:-1]
    return original, deformed, noise, ori_ks


@given(theta=st.floats(0., 60,),
       psi=st.floats(-90., 90.),
       kappa=st.floats(1.+1e-7, 2, exclude_min=True),
       r_k=st.floats(0.015, 0.24),
       )
def test_extract_primary_ks(r_k, theta, psi, kappa):
    size = 256
    ori_ks = latticegen.generate_ks(r_k, theta, kappa=kappa, psi=psi)[:-1]
    original = latticegen.hexlattice_gen(r_k, theta, order=1, size=size, kappa=kappa, psi=psi).compute()
    ext_ks, _ = GPA.extract_primary_ks(original, DoG=False)
    # rel_diffs = np.linalg.norm(standardize_ks(ext_ks) - standardize_ks(ori_ks[:3]), axis=1) / r_k
    # standardize ks has edge cases. Instead, compare each found k to the closed original k
    abs_diffs = np.linalg.norm((ext_ks[None] - ori_ks[:, None]), axis=-1).min(axis=0)
    rel_diffs =  abs_diffs / r_k
    assert np.all(abs_diffs < 1.5/size)
    #assert np.all(rel_diffs < 0.2)


def test_displacement_field(testset_gaussian, gaussiandeform):
    original, deformed, noise, ori_ks = testset_gaussian
    u = -GPA.extract_displacement_field(deformed+noise, ori_ks[:3])
    assert u.shape == gaussiandeform.shape
    print(np.abs(u - gaussiandeform)[:,20:-20,20:-20].max())
    assert np.all(np.abs(u - gaussiandeform)[:,20:-20,20:-20] < 0.9)
    u2 = -GPA.extract_displacement_field(deformed, ori_ks[:3], deconvolve=True)
    assert u2.shape == gaussiandeform.shape
    print(np.abs(u2 - gaussiandeform)[:,20:-20,20:-20].max())
    assert np.all(np.abs(u2 - gaussiandeform)[:,20:-20,20:-20] < 0.05)


def test_reconstruction(testset_gaussian, gaussiandeform):
    original, deformed, noise, ori_ks = testset_gaussian
    u_inv = GPA.invert_u_overlap(-gaussiandeform)
    assert u_inv.shape == gaussiandeform.shape
    reconstructed = GPA.undistort_image(deformed, gaussiandeform)
    assert np.all(np.abs(reconstructed - original) / np.abs(original).max() < 0.02)
    # Add optical flow if feeling fancy, but parameters are hard to optimize.