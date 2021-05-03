# -*- coding: utf-8 -*-
import numpy as np

import latticegen

from .mathtools import wrapToPi, standardize_ks, periodic_average
from .geometric_phase_analysis import calc_diff_from_isotropic, myweighed_lstsq, f2angle


def u2J(U, nmperpixel):
    """From the displacement field U, calculate J.
    """
    J = np.stack(np.gradient(-U, nmperpixel, axis=(1, 2)), axis=-1)
    J = np.moveaxis(J, 0, -2)
    J = (np.eye(2) + J)
    return J


def phases2J(kvecs, phases, weights, nmperpixel):
    """Calculate properties from phases directly.
    Does not take into account base values of properties
    as kvecs themselve describe, e.g. average twist angle.
    """
    K = 2*np.pi*(kvecs)
    dbdx, dbdy = wrapToPi(np.stack(np.gradient(phases, axis=(1, 2)))*2)/2/nmperpixel
    #dbdy = wrapToPi(np.diff(phases, axis=1))
    dudx = myweighed_lstsq(dbdx, K, weights)
    dudy = myweighed_lstsq(dbdy, K, weights)
    J = - np.stack([dudx, dudy], axis=-1)
    J = np.moveaxis(J, 0, -2)
    J = (np.eye(2) + J)
    return J


def phasegradient2J(kvecs, grads, weights, nmperpixel):
    """Calculate J directly from phase gradients.
    Using phase gradients calculated in wfr directly counters
    artefacts at reference k-vector boundaries.
    Include calculation of base values from used kvecs.
    Returns J_diff, the dislocation field gradient of the
    difference of both layers directly, based on the
    average twist angle.
    """
    dks = calc_diff_from_isotropic(kvecs)
    K = 2*np.pi*(kvecs + dks)
    iso_grads = np.stack([g - 2*np.pi*dk
                          for g, dk in zip(grads, dks)])
    iso_grads = wrapToPi(iso_grads)
    # TODO: make a nice reshape for this call?
    dudx = myweighed_lstsq(iso_grads[..., 0], K, weights)
    dudy = myweighed_lstsq(iso_grads[..., 1], K, weights)
    J = np.stack([dudx, dudy], axis=-1) / nmperpixel
    J = np.moveaxis(J, 0, -2)
    J = (np.eye(2) + J)
    return J


def kvecs2J(ks, standardize=True):
    if standardize:
        kvecs = standardize_ks(ks)
    else:
        kvecs = ks
    r_k, theta_0, symmetry = get_initial_props(kvecs)
    krefs = latticegen.generate_ks(r_k, theta_0, sym=symmetry)[:3]
    #krefs = latticegen.generate_ks(r_k, 0, sym=symmetry)[:3]
    if standardize:
        krefs = standardize_ks(krefs)
    dks = krefs - kvecs

    J = np.linalg.lstsq(krefs, -dks, rcond=None)[0]
    J = (np.eye(2) + J).T
    # This transpose is needed for tests to work? Checkout if needed for phases2J, u2J and phasegradient2J too...
    return J


def kvecs2T(ks):
    """T is the transformation from lattice with zero angle
    and unit size"""
    kvecs = ks  # standardize_ks(ks)
    r_k, theta_0, symmetry = get_initial_props(kvecs)
    krefs = latticegen.generate_ks(r_k, 0, sym=symmetry)[:3]
    krefs = standardize_ks(krefs)
    dks = krefs - kvecs

    J = np.linalg.lstsq(krefs, -dks, rcond=None)[0]
    J = r_k * (np.eye(2) + J).T
    return J


def props_from_J(J, refangle=0., refscale=1):
    """Calculate properties of a lattice
    from transformation J

    Returns
    -------
    angle  : float or ndarray
        local angle w.r.t. horizontal in degrees
    aniangle : float or ndarray
        local direction of the anisotropy in degrees
        also w.r.t. horizontal
    alpha : float or ndarray
        local unit cell scaling factor before anisotropy
    kappa : float or ndarray
        local anisotropy magnitude
    """
    u, s, v = np.linalg.svd(J)
    #u = np.sign(np.diag(v)) * u
    #v = (np.sign(np.diag(v)) * v)
    #u_p = u @ v
    v = (np.sign(np.diag(u))*v)
    u = (np.sign(np.diag(u))*u).T
    u_p = (u @ v).T
    angle = np.rad2deg(np.arctan2(u_p[..., 1, 0], u_p[..., 0, 0]))
    aniangle = np.rad2deg(np.arctan2(u[..., 1, 0], u[..., 0, 0])) % 180
    alpha = s[..., 1]
    kappa = s[..., 0] / s[..., 1]
    # return np.array([angle + refangle, aniangle + refangle, alpha * refscale, kappa])
    return np.array([angle + refangle, aniangle, alpha * refscale, kappa])


def calc_props_from_phasegradient(kvecs, grads, weights, nmperpixel):
    """Calculate properties directly from phase gradients.

    Using phase gradients calculated in wfr directly counters
    artefacts at reference k-vector boundaries.
    Include calculation of base values from used kvecs.
    Returns props, a tuple of arrays denoting properties
    as described in https://doi.org/10.1103/PhysRevResearch.3.013153:
    - local angle of the moire lattice w.r.t. horizontal (?)
    - local angle of the anisotropy w.r.t. horizontal (?)
    - local twist angle assuming a graphene lattice (TODO: make flexible)
    - local anisotropy magnitude.
    """
    dks = calc_diff_from_isotropic(kvecs)
    xi_iso = (np.rad2deg(np.arctan2((kvecs+dks)[..., 1],
                                    (kvecs+dks)[..., 0])) % 60).mean()
    J = phasegradient2J(kvecs, grads, weights, nmperpixel)
    props = np.array(props_from_J(J))
    props[2] = props[2]  # * theta_iso
    props[0] = props[0] + xi_iso
    return props


def calc_eps_from_phasegradient(kvecs, grads, weights, nmperpixel):
    """Calculate local lower bound of strain
    assuming uniaxial strain of a graphene lattice
    directly from phase gradients.
    Using phase gradients calculated in wfr directly counters
    artefacts at reference k-vector boundaries.
    """
    J_diff = J_diff_from_phasegradient(kvecs, grads, weights, nmperpixel)
    props = np.array(props_from_J(J_diff))
    kappa = props[3]
    delta = 0.16
    epsilon = (kappa - 1) / (1 + delta*kappa)
    return epsilon


def J_2_J_diff(J, theta_iso):
    t = np.deg2rad(theta_iso)
    J0 = np.array([[np.cos(t)-1, -np.sin(t)],
                   [np.sin(t), np.cos(t)-1]])
    J_diff = J - np.eye(2)
    # Should we use local J instead of J0 here?
    J_diff = J_diff @ J0
    J_diff = (np.eye(2) + J_diff)

    return J_diff


def T_2_T_diff(J, theta_iso):
    t = np.deg2rad(theta_iso)
    J0 = np.array([[np.cos(t)-1, -np.sin(t)],
                   [np.sin(t), np.cos(t)-1]])
    J_diff = J - np.eye(2)
    # Should we use local J instead of J0 here?
    J_diff = J_diff @ J0
    J_diff = (np.eye(2) + J_diff)

    return J_diff


def u_2_u_diff(u, theta_iso):
    t = np.deg2rad(theta_iso)
    J0 = np.array([[np.cos(t)-1, -np.sin(t)],
                   [np.sin(t), np.cos(t)-1]])
    # Should we use local J instead of J0 here?
    u_diff = u @ J0
    return u_diff


def J_diff_from_phasegradient(kvecs, grads, weights, nmperpixel):
    """Calculate J_diff directly from phase gradients.

    Shortcut to prevent + np.eye - np.eye when using J_2_J_diff
    """
    dks = calc_diff_from_isotropic(kvecs)
    theta_iso = f2angle(np.linalg.norm(kvecs + dks, axis=1),
                        nmperpixel=nmperpixel).mean()
    t = np.deg2rad(theta_iso)
    J0 = np.array([[np.cos(t)-1, -np.sin(t)],
                   [np.sin(t), np.cos(t)-1]])
    # xi_iso = (np.rad2deg(np.arctan2((kvecs+dks)[...,1],
    #                                (kvecs+dks)[...,0])) % 60).mean()
    K = 2*np.pi*(kvecs + dks)
    iso_grads = np.stack([g - 2*np.pi*dk
                          for g, dk in zip(grads, dks)])
    iso_grads = wrapToPi(iso_grads)
    # TODO: make a nice reshape for this call?
    dudx = myweighed_lstsq(iso_grads[..., 0], K, weights)
    dudy = myweighed_lstsq(iso_grads[..., 1], K, weights)
    J = np.stack([dudx, dudy], axis=-1) / nmperpixel
    J = np.moveaxis(J, 0, -2)
    # Should we use local J instead of J0 here?
    J_diff = J @ J0
    J_diff = (np.eye(2) + J_diff)
    return J_diff


def calc_props_from_phasegradient2(kvecs, grads, weights, nmperpixel):
    """Calculate properties directly from phase gradients.
    Using phase gradients calculated in wfr directly counters
    artefacts at reference k-vector boundaries.
    Include calculation of base values from used kvecs.
    Returns props, assuming uniaxial strain, angles in degrees.
    - local angle of the moire lattice w.r.t. horizontal (?)
    - local angle of the anisotropy w.r.t. horizontal (?)
    - local twist angle assuming graphene lattices
      of which one is strained by uniaxial strain (TODO: make flexible)
    - local strain epsilon
    """
    dks = calc_diff_from_isotropic(kvecs)
    theta_iso = f2angle(np.linalg.norm(kvecs + dks, axis=1),
                        nmperpixel=nmperpixel).mean()
    xi_iso = (np.rad2deg(np.arctan2((kvecs+dks)[..., 1],
                                    (kvecs+dks)[..., 0])) % 60).mean()
    J = phasegradient2J(kvecs, grads, weights, nmperpixel)
    J_diff = J_2_J_diff(J, theta_iso)
    #assert J_diff == J_diff_from_phasegradient(kvecs, grads, weights, nmperpixel)
    props = np.array(props_from_J(J_diff))
    props[2] = props[2] * theta_iso
    props[0] = props[0] + xi_iso
    return props


def uniaxial_props_from_J(J, delta=0.17):
    u, s, v = np.linalg.svd(J)
    angle = (u@v)
    moireangle = np.rad2deg(np.arctan2(angle[..., 1, 0], angle[..., 0, 0]))
    aniangle = np.rad2deg(np.arctan2(v[..., 1, 0], v[..., 0, 0])) % 180
    d1 = s[..., 0]
    d2 = s[..., 1]
    epsilon = (d1 - d2) / (d2 + delta*d1)
    alpha = (d1 + delta*d2) / (1 + delta)
    return moireangle, aniangle, alpha, epsilon


def calc_props_from_kvecs3(ks):
    """Calculate properties of a lattice directly from `ks`.

    Parameters
    ----------
    ks : ndarray (2 x `symmetry`/2)

    Returns
    -------
    theta : float
        twist angle with respect to
        horizontal in degrees
    psi : float
        orientation of anisotropy, between 0 and 180 degress
    r_k : float
        lattice constant before applying anisotropy
    kappa : float
        anisotropy magnitude
    """
    kvecs = ks  # standardize_ks(ks)
    r_k, theta_0, symmetry = get_initial_props(kvecs)
    krefs = latticegen.generate_ks(r_k, theta_0, sym=symmetry)[:3]
    #krefs = standardize_ks(krefs)
    dks = krefs - kvecs

    J = np.linalg.lstsq(krefs, -dks, rcond=None)[0]
    J = (np.eye(2) + J).T
    props = np.array(props_from_J(J))
    props[0] = props[0] + theta_0
    props[1] = props[1] + theta_0
    props[2] = props[2] * r_k
    return props


def calc_props_from_kvecs4(ks):
    """Calculate properties of a lattice directly from `ks`.

    Parameters
    ----------
    ks : ndarray (2 x `symmetry`/2)

    Returns
    -------
    theta : float
        twist angle with respect to
        horizontal in degrees
    psi : float
        orientation of anisotropy, between 0 and 180 degress
    r_k : float
        lattice constant before applying anisotropy
    kappa : float
        anisotropy magnitude
    """

    J = kvecs2J(ks)
    r_k, theta_0, symmetry = get_initial_props(ks)
    props = props_from_J(J)  # , **get_ref_prop_dict(ks))
    props[0] = props[0] + theta_0
    props[1] = props[1]  # + theta_0
    props[2] = props[2] * r_k
    return props


def get_initial_props(ks):
    kvecs = standardize_ks(ks)
    symmetry = 2 * len(kvecs)
    r_k = np.linalg.norm(kvecs, axis=1).mean()
    theta_0 = np.rad2deg(periodic_average(np.arctan2(*(kvecs).T[::-1]),
                                          2*np.pi / symmetry)
                         )
    return r_k, theta_0, symmetry


def get_ref_prop_dict(ks):
    r_k, theta_0, _ = get_initial_props(ks)
    return {'refangle': theta_0, 'refscale': r_k}


def calc_abcd(J, delta=0.16):
    """decompose J into symmetric and antisymmetric
    parts in both directions. Broadcasts over the first
    dimensions of J (assumes J = (NxMx)2x2)
    """
    a = (J[..., 0, 0] + J[..., 1, 1]) / (1 - delta)
    b = (J[..., 0, 1] + J[..., 1, 0]) / (1 + delta)
    c = (J[..., 1, 0] - J[..., 0, 1]) / (1 - delta)
    d = (J[..., 1, 1] - J[..., 0, 0]) / (1 + delta)
    return a, b, c, d


def double_strain_decomp(J, delta=0.16):
    a, b, c, d = calc_abcd(J, delta=delta)
    bd = b*b + d*d
    alpha = 4 / (1-delta)
    #taylored in 1/alpha
    #c0 = bd * (1+ c*c*(1/alpha**2 - 2/alpha**3))
    #c1 = c*c / (alpha**2) * (1 + 2*np.sqrt(bd)/alpha)
    # Dropping  terms smaller than eps^2/alpha
    ca = c*c/(alpha*alpha)
    #c0 = bd/(1-ca)
    #c1 = ca / (1-ca)
    # Renewed expansion
    c0 = bd * (1 + ca*(1 - 2*np.sqrt(bd) / alpha))
    c1 = ca * (2*np.sqrt(bd) / alpha - 1)
    btemp = bd + a*a*(1 - c1)
    epsminus = np.sqrt(0.5*(btemp + np.sqrt(btemp**2 + 4*a*a*c0)))
    assert np.all(epsminus >= 0.)
    epsplussquare = c0 + c1*epsminus*epsminus

    #phi = np.arccos(a / epsminus)
    # Two ways to compute epsplus, for debug purposes
    #assert np.all(np.sin(phi) >= 0)
    #epsplus = c / np.sin(phi) - alpha
    #assert np.all(epsplus >= 0)

    assert np.all(epsplussquare >= 0)

    epsplus = np.sqrt(epsplussquare)
    phi = np.arcsin(c/(alpha+epsplus))

    epsr = np.tan(phi) * epsminus / epsplus
    #theta = 0.5*np.arctan((b - d*epsr) / (b*epsr + d))
    # I don't know why this gives a seemingly correct answer?
    theta = 0.5*np.arctan2((b*epsr + d), (b - d*epsr))
    epsa = 0.5*(epsplus + epsminus)
    epsb = 0.5*(epsplus - epsminus)
    return theta, phi, epsa, epsb, epsplus, epsplussquare
