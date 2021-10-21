import numpy as np
from hypothesis import given, strategies as st
import pytest

import pyGPA.phase_unwrap as pu

# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.


@pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
@given(kmax=st.integers(1, 30))
def test_equivalent_phase_unwrap_ref_phase_unwrap(kmax):
    N = 256
    xx, yy = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    psi0 = (yy+xx) / (4*np.sqrt(2))
    psi = pu._wrapToPi(psi0)
    weight = np.ones_like(psi)
    result_phase_unwrap_ref = pu.phase_unwrap_ref(
        psi=psi, weight=weight, kmax=kmax
    )
    assert np.allclose(result_phase_unwrap_ref - result_phase_unwrap_ref.mean(), 
                       psi0 - psi0.mean())
    result_phase_unwrap = pu.phase_unwrap(
        psi=psi, weight=weight, kmax=kmax
    )
    assert np.allclose(result_phase_unwrap_ref, result_phase_unwrap)
    result_phase_unwrap = pu.phase_unwrap(
        psi=psi, weight=None, kmax=kmax
    )
    assert np.allclose(result_phase_unwrap_ref, result_phase_unwrap)


def test_equivalent_phase_unwrap_gaussian_weight():
    N = 256
    xx, yy = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    psi0 = (yy+xx) / (4*np.sqrt(2))
    psi = pu._wrapToPi(psi0)
    gaussian = np.exp(-((xx-N//2)**2+(yy-N//2)**2)/(0.3*N**2))
    result_phase_unwrap = pu.phase_unwrap(
        psi=psi, weight=gaussian
    )
    result_phase_unwrap_ref = pu.phase_unwrap(
        psi=psi, weight=None
    )
    assert np.allclose(result_phase_unwrap_ref, result_phase_unwrap)


@pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
@given(kmax=st.integers(1,30))
def test_equivalent_phase_unwrap_ref_prediff_phase_unwrap_prediff(kmax):
    N = 256
    xx, yy = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    psi0 = (yy+xx) / (4*np.sqrt(2))
    psi = pu._wrapToPi(psi0)
    dx = np.diff(psi, axis=1)
    dy = np.diff(psi, axis=0)
    weight = np.ones_like(psi)
    result_phase_unwrap_ref = pu.phase_unwrap_ref_prediff(
        dx=dx, dy=dy, weight=weight, kmax=kmax
    )
    assert np.allclose(result_phase_unwrap_ref - result_phase_unwrap_ref.mean(),
                       psi0 - psi0.mean())
    result_phase_unwrap = pu.phase_unwrap_prediff(
        dx=dx, dy=dy, weight=weight, kmax=kmax
    )
    assert np.allclose(result_phase_unwrap_ref, result_phase_unwrap)
    result_phase_unwrap = pu.phase_unwrap_prediff(
        dx=dx, dy=dy, weight=None, kmax=kmax
    )
    assert np.allclose(result_phase_unwrap_ref, result_phase_unwrap)
    result_phase_unwrap_ref = pu.phase_unwrap_ref(
        psi=psi, weight=weight, kmax=kmax
    )
    assert np.allclose(result_phase_unwrap_ref, result_phase_unwrap)
