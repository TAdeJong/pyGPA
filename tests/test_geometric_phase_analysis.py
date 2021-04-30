import numpy as np
from hypothesis import given, assume
import hypothesis.strategies as st

import latticegen
from latticegen.transformations import rotation_matrix, scaling_matrix

import pyGPA.geometric_phase_analysis as GPA
from pyGPA.mathtools import periodic_difference, standardize_ks

# def gaussiandeform(size=500):
#     S = size // 2
#     xp, yp = da.meshgrid(np.arange(-S,S), np.arange(-S,S), indexing='ij')
#     xshift = 0.6*xp*np.exp(-0.5*((xp/(2*S/6))**2+(yp/(2*S/6))**2))
#     return np.stack((xshift,np.zeros_like(xshift)), axis=0)


# def randomdeform(size=500):
#     pass


@given(theta=st.floats(0., 60,),
       psi=st.floats(-90., 90.),
       kappa=st.floats(1.+1e-7, 2, exclude_min=True),
       r_k=st.floats(0.015, 0.2),
       )
def test_extract_primary_ks(r_k, theta, psi, kappa):
    ori_ks = latticegen.generate_ks(r_k, theta, kappa=kappa, psi=psi)[:-1]
    original = latticegen.hexlattice_gen(r_k, theta, order=1, size=256, kappa=kappa, psi=psi).compute()
    ext_ks, _ = GPA.extract_primary_ks(original, DoG=False)
    # rel_diffs = np.linalg.norm(standardize_ks(ext_ks) - standardize_ks(ori_ks[:3]), axis=1) / r_k
    # standardize ks has edge cases. Instead, compare each found k to the closed original k
    rel_diffs = np.linalg.norm((ext_ks[None] - ori_ks[:, None]), axis=-1).min(axis=0) / r_k
    assert np.all(rel_diffs < 0.2)

# TODO: higher order lattice, with gaussian noise.
