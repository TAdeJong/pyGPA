import numpy as np
import pytest

import latticegen

import pyGPA.unit_cell_averaging as uc
from test_geometric_phase_analysis import gaussiandeform  # noqa F401


@pytest.mark.parametrize("z", [2, 3])
def test_project_and_expand(z):
    r_k = 0.02
    xi0 = 7.0
    psi = 0.0
    kappa = 1.05
    order = 2
    z = 2
    ori_ks = latticegen.generate_ks(r_k, xi0, kappa=kappa, psi=psi)[:2]
    original = latticegen.hexlattice_gen(r_k, xi0, order, kappa=kappa, psi=psi, size=200).compute()
    original = original / original.max()
    ucelorig = uc.unit_cell_average(original, ori_ks, z=z)
    uc_averaged = uc.expand_unitcell(ucelorig, ori_ks, original.shape, z=z)
    assert np.abs(original-uc_averaged).mean() < 3e-3
    assert np.abs(original-uc_averaged).max() < 0.1


@pytest.mark.parametrize("z", [2, 3])
def test_deformed_project_and_expand(z, gaussiandeform): # noqa F811
    r_k = 0.02
    xi0 = 7.0
    psi = 0.0
    kappa = 1.05
    order = 2
    z = 2
    ori_ks = latticegen.generate_ks(r_k, xi0, kappa=kappa, psi=psi)[:2]
    deformed = latticegen.hexlattice_gen(r_k, xi0, order, kappa=kappa, psi=psi, shift=gaussiandeform).compute()
    deformed = deformed / deformed.max()
    ucelorig = uc.unit_cell_average(deformed, ori_ks, z=z, u=gaussiandeform)
    uc_averaged = uc.expand_unitcell(ucelorig, ori_ks, deformed.shape, z=z, u=gaussiandeform)
    assert np.abs(deformed-uc_averaged).mean() < 3e-3
    assert np.abs(deformed-uc_averaged).max() < 0.1
