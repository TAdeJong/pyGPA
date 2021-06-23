import numpy as np
from hypothesis import given, example
import hypothesis.strategies as st

import latticegen
from latticegen.transformations import rotation_matrix, scaling_matrix, a_0_to_r_k, epsilon_to_kappa

import pyGPA.property_extract as pe
from pyGPA.mathtools import periodic_difference
from pyGPA.geometric_phase_analysis import f2angle


@given(theta=st.floats(0., 360.),
       psi=st.floats(-90., 90.),
       kappa=st.floats(1.+1e-7, 1e4, exclude_min=True),
       a=st.floats(1e-10, 1e10, exclude_min=True),
       )
def test_props_from_J(theta, psi, kappa, a):
    W = rotation_matrix(np.deg2rad(theta))
    V = rotation_matrix(np.deg2rad(psi))
    D = scaling_matrix(kappa)*a
    Jac_ori = V.T @ D @ V @ W
    props = pe.props_from_Jac(Jac_ori)
    assert np.isclose(periodic_difference(props[0], theta, period=360), 0, atol=1e-6)
    assert np.isclose(periodic_difference(props[1], psi, period=180), 0, atol=1e-5)
    assert np.isclose(props[2], a)
    assert np.isclose(props[3], kappa)
    props2 = pe.props_from_J(Jac_ori / a - np.eye(2), refscale=a)
    assert np.isclose(periodic_difference(props2[0], theta, period=360), 0, atol=1e-6)
    assert np.isclose(periodic_difference(props2[1], psi, period=180), 0, atol=1e-5)
    assert np.isclose(props2[2], a)
    assert np.isclose(props2[3], kappa)

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
    u, s, v, = np.linalg.svd(J_ori)
    v = (np.sign(np.diag(u))*v)
    u = (np.sign(np.diag(u))*u).T
    angle = (u@v).T
    assert np.allclose(angle, W)
    assert np.allclose(np.diag(s), D)
    assert np.allclose(V, u)
    assert np.isclose(np.linalg.det(D), np.linalg.det(J_ori))


@given(theta=st.floats(-180., 180., exclude_min=True),
       psi=st.floats(-90., 90.),
       kappa=st.floats(1.+1e-7, 1e3, exclude_min=True),
       a=st.floats(1e-9, 1e9, exclude_min=True),
       )
def test_calc_props_from_kvecs(theta, psi, kappa, a):
    kvecs = latticegen.generate_ks(a, theta,
                                   kappa=kappa,
                                   psi=psi)[:3]
    props = pe.calc_props_from_kvecs4(kvecs)
    assert np.isclose(periodic_difference(props[0], theta, period=60), 0, atol=1e-3) #atol=1e-6)
    assert np.isclose(periodic_difference(props[1], psi, period=180), 0, atol=1e-2)
    assert np.isclose(props[2], a) #, rtol=1e-6)
    assert np.isclose(props[3], kappa)


@given(theta=st.floats(1e-2, 60 - 1e-2, exclude_min=True),
       psi=st.floats(-90., 90.),
       kappa=st.floats(1.+1e-7, 1.1, exclude_min=True),
       a=st.floats(1e-9, 1e9, exclude_min=True),
       )
def test_kvecs2Jac(theta, psi, kappa, a):
    ks = latticegen.generate_ks(a, theta, kappa=kappa, psi=psi)[:3]
    Jac = pe.kvecs2Jac(ks, standardize=False)
    J = pe.kvecs2J(ks, standardize=False)
    assert np.allclose(Jac, J + np.eye(2))
    r_kl, theta_0, symmetry = pe.get_initial_props(ks)
    krefs = latticegen.generate_ks(r_kl, theta_0, sym=symmetry)[:-1]
    krefs2 = krefs @ Jac.T
    abs_diffs = np.linalg.norm((krefs2[None] - ks[:, None]), axis=-1).min(axis=1)
    rel_diffs =  abs_diffs / r_kl
    #rel_error = np.linalg.norm(krefs2 - ks, axis=1) / r_kl
    assert np.allclose(rel_diffs, 0, atol=1e-3)


@given(theta=st.floats(1e-1, 45 - 1e-1),#, exclude_min=True),
       psi=st.floats(-90., 90.),
       epsilon=st.floats(1e-5, 0.1, exclude_min=True),
       a=st.floats(1e-3, 1e3, exclude_min=True),
       xi=st.floats(-180., 180.),
       )
def test_kerelsky_plus(theta, psi, epsilon, a, xi):
    ks1 = latticegen.generate_ks(a_0_to_r_k(a), xi, kappa=1, psi=psi)
    r_k2, kappa = epsilon_to_kappa(a_0_to_r_k(a), epsilon)
    ks2 = latticegen.generate_ks(r_k2, xi+theta, kappa=kappa, psi=psi)
    props = pe.Kerelsky_plus(ks2[:3] - ks1[:3],
                               nmperpixel=1, a_0=a)
    assert np.isclose(periodic_difference(np.abs(props[0]), theta, period=60), 0, atol=1e-5)
    assert np.isclose(periodic_difference(props[1], psi, period=180), 0, atol=1e-3)
    assert np.isclose(props[2], epsilon, rtol=1e-3, atol=1e-6)
    assert np.isclose(periodic_difference(props[3], xi, period=360), 0, atol=1e-5)


@given(theta=st.floats(1e-6, 60 - 1e-6, exclude_min=True),
       nmperpixel=st.floats(1e-9, 1e9, exclude_min=True),
       a=st.floats(1e-9, 1e9, exclude_min=True),
       )  
def test_f2angle(theta, nmperpixel, a):
    # a is in nm / unit cell, so to convert to unit cells per pixel, we divide
    ks1 = latticegen.generate_ks(a_0_to_r_k(a/nmperpixel), 0)
    ks2 = latticegen.generate_ks(a_0_to_r_k(a/nmperpixel), theta)
    moire_ks = ks1[:3] - ks2[:3]
    r_k, theta_0, symmetry = pe.get_initial_props(moire_ks)
    theta_iso = f2angle(r_k, nmperpixel=nmperpixel, a_0=a)
    assert np.isclose(theta_iso, theta)