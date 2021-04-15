import numpy as np
import scipy.ndimage as ndi
#from moisan2011 import per
#from numba import njit, prange
#from skimage.feature import peak_local_max

from .phase_unwrap import phase_unwrap, phase_unwrap_prediff
from .mathtools import wrapToPi
from .geometric_phase_analysis import calc_diff_from_isotropic, myweighed_lstsq, f2angle


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
    return props_from_J(J)

def calc_props_from_phases(kvecs, phases, weights, nmperpixel):
    """Calculate properties from phases directly.
    Does not take into account base values of properties
    as kvecs themselve describe, e.g. average twist angle.
    """
    K = 2*np.pi*(kvecs)
    dbdx , dbdy = wrapToPi(np.stack(np.gradient(phases, axis=(1,2)))*2)/2/nmperpixel
    #dbdy = wrapToPi(np.diff(phases, axis=1))
    dudx = myweighed_lstsq(dbdx, K, weights)
    dudy = myweighed_lstsq(dbdy, K, weights)
    J = -np.stack([dudx,dudy], axis=-1)
    J = np.moveaxis(J, 0, -2)
    J = (np.eye(2) + J)
    return props_from_J(J)


def props_from_J(J):
    """Calculate properties of a lattice
    from J corresponding to u of that lattice"""
    u,s,v = np.linalg.svd(J)
    angle = (u@v)
    moireangle = np.rad2deg(np.arctan2(angle[...,1,0], angle[...,0,0]))
    aniangle = np.rad2deg(np.arctan2(v[...,1,0], v[...,0,0])) % 180
    return moireangle, aniangle, np.sqrt(s[...,0]* s[...,1]), s[...,0] / s[...,1]


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
    theta_iso = f2angle(np.linalg.norm(kvecs + dks, axis=1), 
                        nmperpixel=nmperpixel).mean()
    xi_iso = (np.rad2deg(np.arctan2((kvecs+dks)[...,1],
                                    (kvecs+dks)[...,0])) % 60).mean()
    K = 2*np.pi*(kvecs + dks)
    iso_grads = np.stack([g - 2*np.pi*np.array([dk[0], dk[1]])
                          for g, dk in zip(grads, dks)])
    iso_grads = wrapToPi(iso_grads)
    #TODO: make a nice reshape for this call?
    dudx = myweighed_lstsq(iso_grads[...,0], K, weights)
    dudy = myweighed_lstsq(iso_grads[...,1], K, weights)
    J = np.stack([dudx,dudy], axis=-1) / nmperpixel
    J = np.moveaxis(J, 0, -2)
    J = (np.eye(2) + J)
    props = np.array(props_from_J(J))
    props[2] = props[2] * theta_iso
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


def J_diff_from_phasegradient(kvecs, grads, weights, nmperpixel):
    """Calculate J_diff directly from phase gradients.
    Using phase gradients calculated in wfr directly counters
    artefacts at reference k-vector boundaries.
    Include calculation of base values from used kvecs.
    Returns J_diff, the dislocation field gradient of the
    difference of both layers directly, based on the
    average twist angle.
    """
    dks = calc_diff_from_isotropic(kvecs)
    theta_iso = f2angle(np.linalg.norm(kvecs + dks, axis=1),
                        nmperpixel=nmperpixel).mean()
    t = np.deg2rad(theta_iso)
    J0 = np.array([[np.cos(t)-1, -np.sin(t)],
                   [np.sin(t), np.cos(t)-1]])
    xi_iso = (np.rad2deg(np.arctan2((kvecs+dks)[...,1],
                                    (kvecs+dks)[...,0])) % 60).mean()
    K = 2*np.pi*(kvecs + dks)
    iso_grads = np.stack([g - 2*np.pi*dk
                          for g,dk in zip(grads, dks)])
    iso_grads = wrapToPi(iso_grads)
    #TODO: make a nice reshape for this call?
    dudx = myweighed_lstsq(iso_grads[...,0], K, weights)
    dudy = myweighed_lstsq(iso_grads[...,1], K, weights)
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
#     dks = calc_diff_from_isotropic(kvecs)
#     theta_iso = f2angle(np.linalg.norm(kvecs + dks, axis=1),
#                         nmperpixel=nmperpixel).mean()
#     t = np.deg2rad(theta_iso)
#     J0 = np.array([[np.cos(t)-1, -np.sin(t)],
#                    [np.sin(t), np.cos(t)-1]])
#     xi_iso = (np.rad2deg(np.arctan2((kvecs+dks)[...,1],
#                                     (kvecs+dks)[...,0])) % 60).mean()
#     K = 2*np.pi*(kvecs + dks)
#     iso_grads = np.stack([g - 2*np.pi*dk
#                           for g,dk in zip(grads, dks)])
#     iso_grads = wrapToPi(iso_grads)
#     #TODO: make a nice reshape for this call?
#     dudx = myweighed_lstsq(iso_grads[...,0], K, weights)
#     dudy = myweighed_lstsq(iso_grads[...,1], K, weights)
#     J = np.stack([dudx, dudy], axis=-1) / nmperpixel
#     J = np.moveaxis(J, 0, -2)
#     # Should we use local J instead of J0 here?
#     J_diff = J @ J0
#     J_diff = (np.eye(2) + J_diff)
    J_diff = J_diff_from_phasegradient(kvecs, grads, weights, nmperpixel)
    props = np.array(uniaxial_props_from_J(J_diff))
    props[2] = props[2] * theta_iso
    props[0] = props[0] + xi_iso
    return props

def uniaxial_props_from_J(J, delta=0.17):
    u,s,v = np.linalg.svd(J)
    angle = (u@v)
    moireangle = np.rad2deg(np.arctan2(angle[...,1,0], angle[...,0,0]))
    aniangle = np.rad2deg(np.arctan2(v[...,1,0], v[...,0,0])) % 180
    d1 = s[...,0]
    d2 = s[...,1]
    epsilon = (d1 - d2) / (d2 + delta*d1)
    alpha = (d1 + delta*d2) / (1 + delta)
    return moireangle, aniangle, alpha, epsilon


def calc_props_from_kvecs(kvecs, nmperpixel, decomposition='uniaxial'):
    """Calculate properties directly from kvecs.
    Include calculation of base values from used kvecs.
    Returns props, assuming uniaxial strain
    - local angle of the moire lattice w.r.t. horizontal (?)
    - local angle of the anisotropy w.r.t. horizontal (?)
    - local twist angle assuming graphene lattices
      of which one is strained by uniaxial strain (TODO: make flexible)
    - local strain magnitude epsilon
    """
    dks = calc_diff_from_isotropic(kvecs)
    theta_iso = f2angle(np.linalg.norm(kvecs + dks, axis=1),
                        nmperpixel=nmperpixel).mean()
    t = np.deg2rad(theta_iso)
    J0 = np.array([[np.cos(t)-1, -np.sin(t)],
                   [np.sin(t), np.cos(t)-1]])
    xi_iso = (np.rad2deg(np.arctan2((kvecs+dks)[...,1],
                                    (kvecs+dks)[...,0])) % 60).mean()
    K = 2*np.pi*(kvecs + dks)
    iso_grads = wrapToPi(-2*np.pi*dks)
    #TODO: make a nice reshape for this call?
    dudx = np.linalg.lstsq(K, iso_grads[...,0])[0]
    dudy = np.linalg.lstsq(K, iso_grads[...,1])[0]
    
    J = np.stack([dudx, dudy], axis=-1) / nmperpixel
    J = np.moveaxis(J, 0, -2)
    # Should we use local J instead of J0 here?
    J_diff = J @ J0
    J_diff = (np.eye(2) + J_diff)
    if decomposition =='uniaxial':
        props = np.array(uniaxial_props_from_J(J_diff))
    else:
        props = np.array(props_from_J(J_diff))
    props[2] = props[2] * theta_iso
    props[0] = props[0] + xi_iso
    return props


def calc_abcd(J, delta=0.16):
    """decompose J into symmetric and antisymmetric 
    parts in both directions. Broadcasts over the first
    dimensions of J (assumes J = (NxMx)2x2)
    """
    a = (J[...,0,0] + J[...,1,1]) / (1 - delta)
    b = (J[...,0,1] + J[...,1,0]) / (1 + delta)
    c = (J[...,1,0] - J[...,0,1]) / (1 - delta)
    d = (J[...,1,1] - J[...,0,0]) / (1 + delta)
    return a,b,c,d

def double_strain_decomp(J, delta=0.16):
    a,b,c,d = calc_abcd(J, delta=delta)
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
    #Two ways to compute epsplus, for debug purposes
    #assert np.all(np.sin(phi) >= 0)
    #epsplus = c / np.sin(phi) - alpha
    #assert np.all(epsplus >= 0)
    print("Boe!")
    
    assert np.all(epsplussquare >= 0)
    
    epsplus = np.sqrt(epsplussquare)
    phi = np.arcsin(c/(alpha+epsplus))
    
    epsr = np.tan(phi) * epsminus / epsplus
    #theta = 0.5*np.arctan((b - d*epsr) / (b*epsr + d))
    # I don't know why this gives a seemingly correct answer?
    theta = 0.5*np.arctan2( (b*epsr + d), (b - d*epsr))
    epsa = 0.5*(epsplus + epsminus)
    epsb = 0.5*(epsplus - epsminus)
    return theta, phi, epsa, epsb, epsplus, epsplussquare