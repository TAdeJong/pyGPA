"""Unit cell averaging of images."""
import numpy as np
import scipy.ndimage as ndi
from numba import njit


def forward_transform(vecs, ks):
    # A = 0.5*np.sqrt(3) * ks
    A = ks
    return vecs @ A.T


def backward_transform(vecs, ks):
    # A = 2/np.sqrt(3)*np.linalg.inv(ks)
    A = np.linalg.inv(ks)
    return vecs @ A.T


# @njit()
# def nb_cart_in_uc(vecs):
#     """Convert 2D vecs to cartesian coordinates within the unit cell,
#     i.e. within a first brillouin zone. (in real space)
#     Due to numba constraints only takes a one additional dimension,
#     i.e. a list of vectors.
#     """
#     #return nb_backward_transform((nb_forward_transform(vecs) - nb_forward_transform(np.array([55.,0.]))) % 1.) - rmin
#     return nb_backward_transform((nb_forward_transform(vecs) ) % 1.) - rmin

def cart_in_uc(vecs, ks, rmin=0):
    """Convert 2D vecs to cartesian coordinates within the unit cell,
    i.e. within a first brillouin zone. (in real space)
    vecs should reside in the last dimension, e.g.  vecs.shape = 100x100x2
    """
    return backward_transform(forward_transform(vecs, ks) % 1., ks) - rmin


@njit()
def float_overlap(f):
    """The overlap area of a square pixel shifted by `f`
    with its neighbours"""
    A = np.stack((1-f, f))
    return A[:, 0] * np.expand_dims(A[:, 1], 1)


def calc_ucell_parameters(ks, z):
    corners = np.array([[0., 0.],
                        [0., 1.],
                        [1., 0.],
                        [1., 1.]])
    cornervals = backward_transform(corners, ks)  # @ np.linalg.inv(ks).T
    rmin = cornervals.min(axis=0)
    rsize = tuple((z*np.ceil(cornervals.max(axis=0) - np.floor(rmin))).astype(int))
    return rmin, rsize


def unit_cell_average(image, ks, u=None, z=1):
    @njit()
    def nb_forward_transform(vecs):
        return vecs @ ks.T

    @njit()
    def nb_backward_transform(vecs):
        return vecs @ np.linalg.inv(ks).T

    rmin, rsize = calc_ucell_parameters(ks, z)
    if u is None:
        u = np.zeros(image.shape+(2,), dtype=np.float64)

    @njit()
    def nb_cart_in_uc(vecs):
        """Convert 2D vecs to cartesian coordinates within the unit cell,
        i.e. within a first brillouin zone. (in real space)
        Due to numba constraints only takes a one additional dimension,
        i.e. a list of vectors.
        """
        return nb_backward_transform((nb_forward_transform(vecs)) % 1.) - rmin

    @njit()
    def _unit_cell_average(image, u, z=1):
        """Average image with a distortion u over all it's unit cells
        using a drizzle like approach, scaling the unit cell
        up by a factor z.
        Return an array containing the unit cell
        and the corresponding weight distrbution."""
        res = np.zeros(rsize)
        weight = np.zeros(rsize)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if not np.isnan(image[i, j]):
                    R = np.array([i, j]).astype(np.float64)
                    R = nb_cart_in_uc(R + u[i, j]) * z
                    R_floor = np.floor(R)
                    overlap = float_overlap(R - R_floor)
                    R_int = R_floor.astype(np.int32)
                    for li in range(overlap.shape[0]):
                        for lj in range(overlap.shape[1]):
                            res[R_int[0]+li, R_int[1]+lj] += image[i, j] * overlap[li, lj]
                            weight[R_int[0]+li, R_int[1]+lj] += overlap[li, lj]
        return res/weight

    return _unit_cell_average(image, u, z)


def add_to_position(value, R, res, weight):
    R_floor = np.floor(R)
    overlap = float_overlap(R - R_floor)
    R_int = R_floor.astype(np.int32)
    for li in range(overlap.shape[0]):
        for lj in range(overlap.shape[1]):
            res[R_int[0]+li, R_int[1]+lj] += value * overlap[li, lj]
            weight[R_int[0]+li, R_int[1]+lj] += overlap[li, lj]

# def overlap_modulo(R_floor):
#     corners = np.array([[0., 0.],
#                         [0., 1.],
#                         [1., 0.],
#                         [1., 1.]])
#     neighbor_pos = nb_cart_in_uc(R_floor + corners) + rmin

#     handled = np.zeros(4, dtype=np.bool)
#     for i in len(neighbor_pos):
#         if not handled[i]:
#             new_pos = neigbor_pos[i] - corners[i]
#             add_to_position(image[i, j], newpos, res, weight)
#             dists = np.linalg.norm(neighbor_pos - newpos, axis=-1)
#             handled[dists <= np.sqrt(2)] = True


def expand_unitcell(unit_cell_image, ks, shape, z=1, z2=1, u=0):
    """Given a unit_cell_image as produced by a unit_cell_average function,
    zoomed in with factor `z`, recreate a full image of size `shape`.
    optionally scale the resulting image resolutinon up by a factor `z2`,
    optionally use a distortion `u`.
    """
    rr = np.mgrid[:shape[0], :shape[1]] / z2
    rrt = np.moveaxis(rr, 0, -1)
    rmin, rsize = calc_ucell_parameters(ks, z)
    X = cart_in_uc(rrt + u, ks, rmin) * z
    res = ndi.map_coordinates(np.nan_to_num(unit_cell_image),
                              np.moveaxis(X, -1, 0),
                              cval=0)
    return res

# @njit()
# def unit_cell_average_uccoords(image, ks, z=1):
#     """Average image over all it's unit cells
#     using a drizzle like approach, scaling the unit cell
#     up by a factor z.
#     Return an array containing the unit cell in lattice coordinates
#     and the corresponding weight distrbution.
#     calculating in lattice coordinates allows to use
#     wrapping for the interpolation, but generally performs worse
#     due to distortion. Might be fixable with a better float_overlap function.
#     (which for now assumes a euclidian lattice)."""
#     scale = int(np.ceil(z / np.linalg.norm(ks[0])))
#     res = np.zeros((scale, scale))
#     weight = np.zeros((scale, scale))
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             R = np.array([i, j]).astype(np.float64)
#             R = nb_cart_in_uc(R, ks) * z
#             R_floor = np.floor(R)
#             overlap = float_overlap(R - R_floor)
#             R_int = R_floor.astype(np.int32)
#             for li in range(overlap.shape[0]):
#                 for lj in range(overlap.shape[1]):
#                     res[(R_int[0]+li) % scale, (R_int[1]+lj) % scale] += image[i, j]*overlap[li, lj]
#                     weight[(R_int[0]+li) % scale, (R_int[1]+lj) % scale] += overlap[li, lj]
#     return res/weight, weight


# def expand_unitcell_uccoords(unit_cell_image, shape, z=1, z2=1, u=0):
#     scale = int(np.ceil(z / np.linalg.norm(ks[0])))
#     rr = np.mgrid[:shape[0], :shape[1]]/z2
#     rrt = np.moveaxis(rr, 0, -1)
#     X = (forward_transform(rrt + u, ks[:2]) % 1.) * scale
#     res = ndi.map_coordinates(unit_cell_image, np.moveaxis(X, -1, 0), modess='wrap')
#     return res
