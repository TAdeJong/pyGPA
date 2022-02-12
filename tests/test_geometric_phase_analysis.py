import numpy as np
import scipy.ndimage as ndi
from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

import latticegen

import pyGPA.geometric_phase_analysis as GPA


@pytest.fixture(scope='module')
def gaussiandeform(size=500):
    S = size // 2
    xp, yp = np.meshgrid(np.arange(-S, S), np.arange(-S, S), indexing='ij')
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


@settings(deadline=350)
@given(theta=st.floats(0., 60,),
       psi=st.floats(-90., 90.),
       kappa=st.floats(1.+1e-7, 2, exclude_min=True),
       r_k=st.floats(0.03, 0.24),
       )
def test_extract_primary_ks(r_k, theta, psi, kappa):
    size = 128
    ori_ks = latticegen.generate_ks(r_k, theta, kappa=kappa, psi=psi)[:-1]
    original = latticegen.hexlattice_gen(r_k, theta, order=1, size=size, kappa=kappa, psi=psi).compute()
    ext_ks, _ = GPA.extract_primary_ks(original, DoG=False)
    # rel_diffs = np.linalg.norm(standardize_ks(ext_ks) - standardize_ks(ori_ks[:3]), axis=1) / r_k
    # standardize ks has edge cases. Instead, compare each found k to the closed original k
    abs_diffs = np.linalg.norm((ext_ks[None] - ori_ks[:, None]), axis=-1).min(axis=0)
    assert np.all(abs_diffs < 1.5/size)


def test_displacement_field(testset_gaussian, gaussiandeform):
    original, deformed, noise, ori_ks = testset_gaussian
    u = -GPA.extract_displacement_field(deformed + noise, ori_ks[:3])
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


@pytest.mark.parametrize("wfr_func1,wfr_func2", [(GPA.optwfr2, GPA.wfr2),
                                                 (GPA.optwfr2, GPA.wfr2_grad),
                                                (GPA.wfr2_grad_opt, GPA.wfr2_grad)])
def test_wfr2_variants_lockin(wfr_func1, wfr_func2, testset_gaussian):
    original, deformed, noise, ori_ks = testset_gaussian
    kw = np.linalg.norm(ori_ks, axis=1).mean() / 2.5
    sigma = int(np.ceil(1/np.linalg.norm(ori_ks, axis=1).min()))
    kstep = kw / 3
    gs = [wfr_func1(deformed - deformed.mean(), sigma,
                   pk[0], pk[1], kw=kw, kstep=kstep)
          for pk in ori_ks]
    gs2 = [wfr_func2(deformed - deformed.mean(), sigma,
                   pk[0], pk[1], kw=kw, kstep=kstep)
          for pk in ori_ks]
    for g1,g2 in zip(gs,gs2):
        assert np.allclose(g1['lockin'], g2['lockin'])
