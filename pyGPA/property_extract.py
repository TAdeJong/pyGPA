# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import least_squares

import latticegen

from .mathtools import wrapToPi, standardize_ks, periodic_average, periodic_difference
from .geometric_phase_analysis import calc_diff_from_isotropic, myweighed_lstsq, f2angle


def u2J(U, nmperpixel):
    """From the displacement field U, calculate J.
    """
    J = np.stack(np.gradient(-U, nmperpixel, axis=(1, 2)), axis=-1)
    J = np.moveaxis(J, 0, -2)
    return J

def u2Jac(U, nmperpixel):
    """From the displacement field U, calculate Jac.
    """
    J = u2J(U)
    Jac = (np.eye(2) + J)
    return Jac


def phases2Jac(kvecs, phases, weights, nmperpixel):
    """Calculate properties from phases directly.
    Does not take into account base values of properties
    as kvecs themselve describe, e.g. average twist angle.
    """
    J = phases2J(kvecs, phases, weights, nmperpixel)
    Jac = np.eye(2) + J
    return Jac

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
    #suspicious minus sign
    J = - np.stack([dudx, dudy], axis=-1)
    J = np.moveaxis(J, 0, -2)
    return J


def phasegradient2Jac(kvecs, grads, weights, nmperpixel):
    """Calculate Jac directly from phase gradients.
    Using phase gradients calculated in wfr directly counters
    artefacts at reference k-vector boundaries.
    Include calculation of base values from used kvecs.
    Returns J_diff, the dislocation field gradient of the
    difference of both layers directly, based on the
    average twist angle.
    """
    J = phasegradient2J(kvecs, grads, weights, nmperpixel)
    Jac = (np.eye(2) + J)
    return Jac

def phasegradient2J(kvecs, grads, weights, nmperpixel):
    """Calculate J directly from phase gradients.
    Using phase gradients calculated in wfr directly counters
    artefacts at reference k-vector boundaries.
    Include calculation of base values from used kvecs.
    Returns J w.r.t. kvecs+calc_diff_from_isotropic(kvecs)
    based on the average twist angle.
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
    return J


def kvecs2J(ks, standardize=True):
    """Convert given ks to a J matrix
    with respect to the isotropic kvectors as
    given by get_initial_props(ks). 
    
    Parameters
    ----------
    ks : Nx2 array
        k-vectors to convert
    standardardize: bool, default True
        Whether to apply standardize_ks everywhere.
    """
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
    # This transpose is needed for tests to work? Checkout if needed for phases2J, u2J and phasegradient2J too...
    return J.T

def kvecs2Jac(ks, standardize=True):
    J = kvecs2J(ks, standardize=standardize)
    Jac = (np.eye(2) + J)
    return Jac


# def kvecs2T(ks):
#     """T is the transformation from lattice with zero angle
#     and unit size"""
#     kvecs = ks  # standardize_ks(ks)
#     r_k, theta_0, symmetry = get_initial_props(kvecs)
#     krefs = latticegen.generate_ks(r_k, 0, sym=symmetry)[:3]
#     krefs = standardize_ks(krefs)
#     dks = krefs - kvecs

#     J = np.linalg.lstsq(krefs, -dks, rcond=None)[0]
#     J = r_k * (np.eye(2) + J).T
#     return J


def props_from_Jac(Jac, refangle=0., refscale=1):
    """Calculate properties of a lattice
    from transformation Jac

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
    u, s, v = np.linalg.svd(Jac)
    #u = np.sign(np.diag(v)) * u
    #v = (np.sign(np.diag(v)) * v)
    #u_p = u @ v
    signs = np.sign(u[...,None,[0,1],[0,1]])
    #v = (np.sign(np.diag(u))*v)
    #u = (np.sign(np.diag(u))*u).T
    v = signs*v
    u = np.swapaxes(signs*u, -1, -2)
    u_p = np.swapaxes(u @ v, -1, -2)
    #u_p = (u @ v).T
    angle = np.rad2deg(np.arctan2(u_p[..., 1, 0], u_p[..., 0, 0]))
    aniangle = np.rad2deg(np.arctan2(u[..., 1, 0], u[..., 0, 0])) % 180
    alpha = s[..., 1]
    kappa = s[..., 0] / s[..., 1]
    # return np.array([angle + refangle, aniangle + refangle, alpha * refscale, kappa])
    return np.array([angle + refangle, aniangle, alpha * refscale, kappa])


def phys_props_from_Jac(Jac, refangle=0., refscale=1, poisson_ratio=0.16):
    """Calculate physical properties of a lattice
    from transformation Jac

    Returns
    -------
    angle  : float or ndarray
        local angle w.r.t. horizontal in degrees
    aniangle : float or ndarray
        local direction of the heterostrain in degrees
        also w.r.t. horizontal
    alpha : float or ndarray
        local unit cell scaling factor before anisotropy
    epsilon  : float or ndarray
        local heterostrain magnitude
    """
    u, s, v = np.linalg.svd(Jac)
    signs = np.sign(u[...,None,[0,1],[0,1]])
    v = signs*v
    u = np.swapaxes(signs*u, -1, -2)
    u_p = np.swapaxes(u @ v, -1, -2)
    angle = np.rad2deg(np.arctan2(u_p[..., 1, 0], u_p[..., 0, 0]))
    aniangle = np.rad2deg(np.arctan2(u[..., 1, 0], u[..., 0, 0])) % 180
    
    delta = poisson_ratio
    epsilon = (s[..., 0] - s[..., 1]) / (s[..., 0] + delta * s[..., 1])
    alpha = s[..., 0] / (1+epsilon)
    return np.array([angle + refangle, aniangle, alpha * refscale, epsilon])

def props_from_J(J, refangle=0., refscale=1):
    return props_from_Jac(J + np.eye(2), refangle=refangle, refscale=refscale)

def props_from_J_old(J):
    """Calculate properties of a lattice
    from J corresponding to u of that lattice"""
    u,s,v = np.linalg.svd(J)
    angle = (u @ v)
    moireangle = np.rad2deg(np.arctan2(angle[...,1,0], angle[...,0,0]))
    aniangle = np.rad2deg(np.arctan2(v[...,1,0], v[...,0,0])) % 180
    return [moireangle, aniangle, np.sqrt(s[...,0]* s[...,1]), s[...,0] / s[...,1]]


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
#     dks = calc_diff_from_isotropic(kvecs)
#     theta_iso = f2angle(np.linalg.norm(kvecs + dks, axis=1), 
#                         nmperpixel=nmperpixel).mean()
#     xi_iso = (np.rad2deg(np.arctan2((kvecs+dks)[..., 1],
#                                     (kvecs+dks)[..., 0])) % 60).mean()
#     J = phasegradient2J(kvecs, grads, weights, nmperpixel)
#     props = np.array(props_from_J(J))
#     props[2] = props[2] * theta_iso
#     props[0] = props[0] + xi_iso
    a_0 = 0.246
    Jac = phasegradient2Jac(kvecs, grads, weights, nmperpixel)
    r_k, theta_0, symmetry = get_initial_props(kvecs)
    theta_iso = f2angle(r_k, nmperpixel=nmperpixel)
    props = props_from_Jac(Jac)
    props[0] = props[0] + theta_0
    props[1] = props[1]  # + theta_0
    props[2] = f2angle(props[2] * r_k, nmperpixel=nmperpixel, a_0=a_0)
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


# def Jac_2_Jac_diff(Jac, theta_iso):
#     t = np.deg2rad(theta_iso)
#     J0 = np.array([[np.cos(t)-1, -np.sin(t)],
#                    [np.sin(t), np.cos(t)-1]])
#     J_diff = Jac - np.eye(2)
#     J_diff = J_diff @ J0
#     Jac_diff = (np.eye(2) + J_diff)

#     return Jac_diff

def Jac_2_Jac_diff(Jac, theta_iso):
    J_diff = J_2_J_diff(Jac - np.eye(2), theta_iso)
    Jac_diff = (np.eye(2) + J_diff)
    return Jac_diff

def J_2_J_diff(J, theta_iso):
    t = np.deg2rad(theta_iso)
    J0 = np.array([[np.cos(t)-1, -np.sin(t)],
                   [np.sin(t), np.cos(t)-1]])
    # Should use J0 corresponding to kvecs used to derive J!
    J_diff = J @ J0

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


def u_moire_2_u_diff(u, theta_iso):
    t = np.deg2rad(theta_iso)
    J0 = np.array([[np.cos(t)-1, -np.sin(t)],
                   [np.sin(t), np.cos(t)-1]])
    # Should we use local J instead of J0 here?
    u_diff = u @ J0
    return u_diff


def Jac_diff_from_phasegradient(kvecs, grads, weights, nmperpixel, a_0=0.246):
    """Calculate Jac_diff directly from phase gradients.

    Shortcut to prevent + np.eye - np.eye when using J_2_J_diff?
    """
    J = phasegradient2J(kvecs, grads, weights, nmperpixel)
    r_k, theta_0, symmetry = get_initial_props(kvecs)
    theta_iso = f2angle(r_k, nmperpixel=nmperpixel, a_0=a_0)
    J_diff = J_to_J_diff(J, theta_iso)
    Jac_diff = (np.eye(2) + J_diff)
    return Jac_diff




def calc_props_from_phasegradient2(kvecs, grads, weights, nmperpixel, a_0=0.246):
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


# def uniaxial_props_from_J(J, delta=0.17):
#     u, s, v = np.linalg.svd(J)
#     angle = (u@v)
#     moireangle = np.rad2deg(np.arctan2(angle[..., 1, 0], angle[..., 0, 0]))
#     aniangle = np.rad2deg(np.arctan2(v[..., 1, 0], v[..., 0, 0])) % 180
#     d1 = s[..., 0]
#     d2 = s[..., 1]
#     #TODO: check if applied in the right direction.
#     epsilon = (d1 - d2) / (d2 + delta*d1)
#     alpha = (d1 + delta*d2) / (1 + delta)
#     return moireangle, aniangle, alpha, epsilon



def calc_props_from_kvecs4(ks,
                          decomposition=None,
                          standardize=False):
    """Calculate properties of a lattice directly from `ks`.

    Parameters
    ----------
    ks : ndarray (2 x `symmetry`/2)
    decomposition : str or None

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

    Jac = kvecs2Jac(ks)
    r_k, theta_0, symmetry = get_initial_props(ks, 
                                               standardize=standardize)
    if decomposition == 'physical':
        props = phys_props_from_Jac(Jac)
    else:
        props = props_from_Jac(Jac)
    props[0] = props[0] + theta_0
    props[1] = props[1]  # + theta_0
    props[2] = props[2] * r_k
    return props

def calc_moire_props_from_kvecs(ks, nmperpixel=3.7, a_0=0.246,
                                decomposition='physical'):
    """Calculate properties of a moire lattice directly from `ks`.

    Parameters
    ----------
    ks : ndarray (2 x `symmetry`/2)

    Returns
    -------
    theta : float
        local twist angle
    psi : float
        orientation of anisotropy, between 0 and 180 degress
    alpha : float
        remaining unit cell scaling.
    epsilon : float
        heterostrain
    """

    Jac = kvecs2Jac(ks, standardize=False)
    props = moire_props_from_Jac(ks, Jac, nmperpixel, a_0, decomposition)
#     r_k, theta_0, symmetry = get_initial_props(ks)
#     theta_iso = f2angle(r_k, nmperpixel=nmperpixel, a_0=a_0)
#     Jac_moire = Jac_2_Jac_diff(Jac, theta_iso)
#     if decomposition == 'physical':
#         props = phys_props_from_Jac(Jac_moire)
#     else:
#         props = props_from_Jac(Jac_moire)
#     props[0] = props[0] + theta_iso
#     print(theta_iso)
#     props[1] = props[1]  #- ((90 + theta_0) % 60)
#     props[2] = f2angle(props[2] * r_k, nmperpixel=nmperpixel, a_0=a_0)
    return props

def moire_props_from_phasegradient(kvecs, grads, weights, nmperpixel, 
                                   a_0=0.246, decomposition=None):
    """Calculate properties directly from phase gradients.

    Using phase gradients calculated in wfr directly counters
    artefacts at reference k-vector boundaries.
    Include calculation of base values from used kvecs.
    Returns props, a tuple of arrays denoting properties
    as described in https://doi.org/10.1103/PhysRevResearch.3.013153:
    - local angle of the moire lattice w.r.t. horizontal (?)
    - local angle of the anisotropy w.r.t. horizontal (?)
    - local twist angle assuming a graphene lattice (TODO: make flexible)
    - heterostrain epsilon
    """
    Jac = phasegradient2Jac(kvecs, grads, weights, nmperpixel)
    props = moire_props_from_Jac(kvecs, Jac, nmperpixel, a_0, decomposition)
#     r_k, theta_0, symmetry = get_initial_props(kvecs)
#     theta_iso = f2angle(r_k, nmperpixel=nmperpixel, a_0=a_0)
#     Jac_moire = Jac_2_Jac_diff(Jac, theta_iso)
#     props = phys_props_from_Jac(Jac_moire)
#     props[0] = props[0] + theta_0
#     props[1] = props[1]  - ((90 + theta_0) % 60)
    
#     props[2] = f2angle(props[2] * r_k, nmperpixel=nmperpixel, a_0=a_0)
    return props

def moire_props_from_Jac(kvecs, Jac, nmperpixel, a_0=0.246, decomposition=None):
    r_k, theta_0, symmetry = get_initial_props(kvecs)
    theta_iso = f2angle(r_k, nmperpixel=nmperpixel, a_0=a_0)
    Jac_moire = Jac_2_Jac_diff(Jac, theta_iso)
    if decomposition == 'physical':
        props = phys_props_from_Jac(Jac_moire)
    else:
        props = props_from_Jac(Jac_moire)
    props[0] = props[0] + theta_iso
    props[1] = props[1] - theta_iso/2#- ((90 + theta_0) % 60)
    
    #props[2] = f2angle(props[2] * r_k, nmperpixel=nmperpixel, a_0=a_0)
    return props

def twist_matrix(angle):
    """Create a twist matrix

    an array corresponding to the transformation matrix
    of a lattice twist over `angle`.

    Parameters
    ----------
    angle : float
        rotation angle in degrees

    Returns
    -------
    ndarray (2x2)
        2D transformation matrix corresponding
        to the rotation
    """
    ha = np.deg2rad(angle/2)
    B0 = np.array([[np.cos(ha), -np.sin(ha)],
                     [np.sin(ha), np.cos(ha)]])
    B0 = B0 - np.array([[np.cos(-ha), -np.sin(-ha)],
                     [np.sin(-ha), np.cos(-ha)]])
    return B0


def moire_props_from_Jac_2_Kerelsky(kvecs, Jac, nmperpixel, a_0=0.246, decomposition=None):
    dks = calc_diff_from_isotropic(kvecs)
    iso_props = Kerelsky_plus(kvecs + dks, nmperpixel, a_0)
    assert iso_props[2] == 0
    B0 = twist_matrix(iso_props[0])
    props = double_strain_decomp(Jac @ B0)
    return props, isoprops


# def get_initial_props(ks, standardize=False):
#     if standardize:
#         kvecs = standardize_ks(ks)
#     else:
#         kvecs = ks 
#     symmetry = 2 * len(kvecs)
#     r_k = np.linalg.norm(kvecs, axis=1).mean()
#     theta_0 = np.rad2deg(periodic_average(np.arctan2(*(kvecs).T[::-1]),
#                                           2*np.pi / symmetry)
#                          )
#     return r_k, theta_0, symmetry

def get_initial_props(ks, standardize=False):
    if standardize:
        kvecs = standardize_ks(ks)
    else:
        kvecs = ks 
    symmetry = 2 * len(kvecs)
    r_k = np.linalg.norm(kvecs, axis=1).mean()
    theta_0 = np.rad2deg(periodic_average(np.arctan2(*(kvecs).T[::-1]),
                                          2*np.pi / symmetry)
                         )
    hexa = np.arange(-180, 180, 60)
    diffind = np.argmin(np.abs(theta_0 + hexa - np.rad2deg(np.arctan2(*(kvecs).T[::-1,0]))))
    return r_k, theta_0 + hexa[diffind], symmetry


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


def double_strain_decomp(Jac, delta=0.16):
    """Do a double strain decomposition on a Jac

    Returns
    -------
    psi : float
        relative twist angle in degrees
    theta : float
        strain angle in degrees
    epsa : float
    epsb : float
    """
    a, b, c, d = calc_abcd(Jac, delta=delta)
    bd = b*b + d*d
    alpha = 4 / (1-delta)
    #taylored in 1/alpha
    #c0 = bd * (1+ c*c*(1/alpha**2 - 2/alpha**3))
    #c1 = c*c / (alpha**2) * (1 + 2*np.sqrt(bd)/alpha)
    # Dropping  terms smaller than eps^2/alpha
    ca = c*c / (alpha*alpha)
    #c0 = bd/(1-ca)
    #c1 = ca / (1-ca)
    # Renewed expansion
    c0 = bd * (1 + ca*(1 - 2*np.sqrt(bd) / alpha))
    c1 = -ca * (1 - 2*np.sqrt(bd) / alpha)
    btemp = bd + a*a*(1 - c1)
    epsminus = np.sqrt(0.5*(btemp + np.sqrt(btemp**2 + 4*a*a*c0)))
    for i in range(2):
        epsplussquare = c0 + c1*epsminus*epsminus
        print('epsminus', epsminus, epsplussquare)
        epsminussquare = ((bd+a*a) + np.sqrt((bd+a*a)**2 + a*a*epsplussquare))/2
        epsminus = np.sqrt(epsminussquare)
        print('epsminus', epsminus)
    #phi = np.arccos(a / epsminus)
    # Two ways to compute epsplus, for debug purposes
    #assert np.all(np.sin(phi) >= 0)
    #epsplus = c / np.sin(phi) - alpha
    #assert np.all(epsplus >= 0)

    assert np.all(epsplussquare >= 0)

    epsplus = np.sqrt(epsplussquare)
    phi = np.arcsin(c / (alpha+epsplus))

    epsr = np.tan(phi) * epsminus / epsplus
    theta = 0.5*np.arctan((b - d*epsr) / (b*epsr + d))
    # I don't know why this gives a seemingly correct answer?
    #theta = 0.5*np.arctan2((b*epsr + d), (b - d*epsr))
    epsa = 0.5*(epsplus + epsminus)
    epsb = 0.5*(epsplus - epsminus)
    return np.array([2*np.rad2deg(phi),
                     np.rad2deg(theta),
                     epsa,
                     epsb]) #epsplus, epsplussquare


def moire_amplitudes(theta, psi, epsilon, a_0=0.246):
    ks1 = latticegen.generate_ks(latticegen.transformations.a_0_to_r_k(a_0), 0)[:3]
    W = latticegen.transformations.rotation_matrix(np.deg2rad(theta))
    V = latticegen.transformations.rotation_matrix(np.deg2rad(psi))
    D = latticegen.transformations.strain_matrix(epsilon)
    ks2 = latticegen.transformations.apply_transformation_matrix(ks1,  V.T @ D @ V @ W)
    return np.linalg.norm(ks1 - ks2, axis=1)



def Kerelsky(kvecs, nmperpixel=1., a_0=0.246):
    knorms = np.linalg.norm(kvecs, axis=1) * nmperpixel
    res = least_squares(lambda x: (moire_amplitudes(*x, a_0) - knorms)/knorms.mean(), [0.01, 0., 0.])
    if res.cost > 1e-20:
        res2 = least_squares(lambda x: (moire_amplitudes(*x, a_0) - knorms)/knorms.mean(), [.01, 90., 0.])
        if res2.cost < res.cost:
            res = res2
    if res.success:
        params = res.x
    else:
        params = np.full(4, np.nan)
    return params

def Kerelsky_plus(kvecs, nmperpixel=1., a_0=0.246,
                  reference=None,
                  debug=False):
    """From kvecs, compute properties using Kerelsky et al.

    In addition to what Kerelsky et al. describe,
    fit a reference angle `xi`.

    Parameters
    ----------
    kvecs : array_like, Nx2
        k-vectors in unit cells per pixel
    nmperpixel : float
        resolution, in nm per pixel
    a_0 : float
        real space size of the underlying unit cell
        in nm per unit cell
    reference : None or 'symmetric'
    debug : boolean

    Returns
    -------
    theta : float
        twist angle in degrees
    psi : float
        strain angle in degrees
    epsilon : float
        hetero strain
    xi : float
        angle of lattice with respect to horizontal
        in degrees

    References
    ----------
    [1] Kerelsky et al., https://www.nature.com/articles/s41586-019-1431-9
        Suppl. Note 1.
    """
    knorm = np.linalg.norm(kvecs, axis=1).mean() / nmperpixel
    angles = np.arctan2(*kvecs.T[::-1])
    lkvecs = kvecs[np.argsort(periodic_difference(angles, periodic_average(angles)))]
    #lkvecs = standardize_ks(kvecs)
    def moire_diffs(args):
        theta, psi, epsilon, xi = args
        ks1 = latticegen.generate_ks(latticegen.transformations.a_0_to_r_k(a_0), xi)[:3]
        W = latticegen.transformations.rotation_matrix(np.deg2rad(theta))
        V = latticegen.transformations.rotation_matrix(np.deg2rad(psi))
        D = latticegen.transformations.strain_matrix(epsilon)
        ks2 = latticegen.transformations.apply_transformation_matrix(ks1,  V.T @ D @ V @ W)
        return np.ravel(lkvecs / nmperpixel - (ks2 - ks1)) / knorm * 100
    bounds = np.full((2,4), np.inf)
    bounds[0,:] = -np.inf
    bounds[0,[0,2]] = 0
    gtol = 1e-8
    if a_0 < 1:
        gtol = gtol* a_0
    est = [.01, 0., 0., np.rad2deg(np.arctan2(lkvecs[0,1], lkvecs[0,0]))%360]
    res = least_squares(moire_diffs, est, bounds=bounds, gtol=gtol)
    if res.cost > 1e-20:
        est[1] = 90.
        res2 = least_squares(moire_diffs, est, bounds=bounds, gtol=gtol)
        if res2.cost < res.cost:
            res = res2
    if debug:
        print(res)
    if res.success:
        params = res.x
    else:
        params = np.full(4, np.nan)
    if reference == 'symmetric':
        params[3] = params[3] + params[0]/2
    return params