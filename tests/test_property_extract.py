import numpy as np
from hypothesis import given
import hypothesis.strategies as st

import latticegen
from latticegen.transformations import rotation_matrix, scaling_matrix

import pyGPA.property_extract as pe
from pyGPA.mathtools import periodic_difference


@given(theta=st.floats(0., 360.),
       psi=st.floats(-90., 90.),
       kappa=st.floats(1.+1e-7, 1e10, exclude_min=True),
       a=st.floats(1e-10, 1e10, exclude_min=True),
       )
def test_props_from_J(theta, psi, kappa, a):
    W = rotation_matrix(np.deg2rad(theta))
    V = rotation_matrix(np.deg2rad(psi))
    D = scaling_matrix(kappa)*a
    J_ori = V.T @ D @ V @ W
    props = pe.props_from_J(J_ori)
    assert np.isclose(periodic_difference(props[0], theta, period=360), 0, atol=1e-6)
    assert np.isclose(periodic_difference(props[1], psi, period=180), 0, atol=1e-5)
    assert np.isclose(props[2], a)
    assert np.isclose(props[3], kappa)

@given(theta=st.floats(0., 360,),
       psi=st.floats(-90., 90.),
       kappa=st.floats(1.+1e-7, 1e10, exclude_min=True),
       a=st.floats(1e-5, 1e5, exclude_min=True),
       )
def test_svd_assumptions(theta, psi, kappa, a):
    W = rotation_matrix(np.deg2rad(theta))
    V = rotation_matrix(np.deg2rad(psi))
    D = scaling_matrix(kappa)*a
    J_ori = V.T @ D @ V @ W
    u,s,v, = np.linalg.svd(J_ori)
    v = (np.sign(np.diag(u))*v)
    u = (np.sign(np.diag(u))*u).T
    angle = (u@v).T
    assert np.allclose(angle, W)
    assert np.allclose(np.diag(s), D)
    assert np.allclose(V, u)
    assert np.isclose(np.linalg.det(D), np.linalg.det(J_ori))
    
@given(theta=st.floats(-180., 180., exclude_min=True),
       psi=st.floats(-90., 90.),
       kappa=st.floats(1.+1e-7, 1e4, exclude_min=True),
       a=st.floats(1e-10, 1e10, exclude_min=True),
       )
def test_calc_props_from_kvecs(theta, psi, kappa, a):
    kvecs = latticegen.generate_ks(a, theta, 
                                   kappa=kappa, 
                                   psi=psi)[:3]
    props = pe.calc_props_from_kvecs4(kvecs)
    assert np.isclose(periodic_difference(props[0], theta, period=60), 0, atol=1e-6)
    assert np.isclose(periodic_difference(props[1], psi, period=180), 0, atol=1e-2)
    assert np.isclose(props[2], a)
    assert np.isclose(props[3], kappa)
    
    
# @given(theta=st.floats(0., 360., exclude_min=True),
#        psi=st.floats(-90., 90.),
#        kappa=st.floats(1.+1e-7, 1e9, exclude_min=True),
#        a=st.floats(1e-9, 1e9, exclude_min=True),
#        )
# def test_moire2diff(theta, psi, kappa, a):
#     ks1 = latticegen.generate_ks(a, 5, kappa=kappa, psi=psi)
#     ks2 = latticegen.generate_ks(a, 5+theta, kappa=kappa, psi=psi)
#     J1 = pe.kvecs2J(ks1[:3])
#     J2 = pe.kvecs2J(ks2[:3])
#     J_moire = pe.kvecs2J(ks1[:3]-ks2[:3])
#     J_diff = pe.J_2_J_diff(J_moire, theta)
#     assert np.allclose(J_diff, J1-J2)