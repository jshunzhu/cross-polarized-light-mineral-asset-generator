"""Real, nonabsorbing rank-two crystal optics, in right/up/beam coordinates.

B is dielectric impermeability (inverse squared principal indices), not a
projection of an index ellipsoid. Arrays may have arbitrary leading batch axes.
"""
from __future__ import annotations

import numpy as np


def rotation_matrix(azimuth_rad, tilt_rad):
    """Proper rotation Rz(azimuth) Ry(tilt); the third column is the c axis."""
    azimuth, tilt = np.broadcast_arrays(azimuth_rad, tilt_rad)
    ca, sa, ct, st = np.cos(azimuth), np.sin(azimuth), np.cos(tilt), np.sin(tilt)
    result = np.empty(azimuth.shape + (3, 3), dtype=np.float64)
    result[..., 0, :] = np.stack((ca*ct, -sa, ca*st), axis=-1)
    result[..., 1, :] = np.stack((sa*ct, ca, sa*st), axis=-1)
    result[..., 2, :] = np.stack((-st, np.zeros_like(st), ct), axis=-1)
    return result


def validate_frame(frame):
    if np.iscomplexobj(frame): raise ValueError('The real nonabsorbing core does not accept complex frames.')
    frame = np.asarray(frame, dtype=np.float64)
    if frame.shape[-2:] != (3, 3) or not np.isfinite(frame).all():
        raise ValueError('A finite 3x3 frame is required.')
    if (not np.allclose(np.swapaxes(frame, -1, -2) @ frame, np.eye(3), atol=1e-10, rtol=0)
            or not np.allclose(np.linalg.det(frame), 1., atol=1e-10, rtol=0)):
        raise ValueError('Frame must be orthonormal and right-handed (SO(3)).')
    return frame


def validate_tensor(tensor):
    if np.iscomplexobj(tensor): raise ValueError('The real nonabsorbing core does not accept complex tensors.')
    tensor = np.asarray(tensor, dtype=np.float64)
    if tensor.shape[-2:] != (3, 3) or not np.isfinite(tensor).all():
        raise ValueError('A finite 3x3 impermeability tensor is required.')
    if not np.allclose(tensor, np.swapaxes(tensor, -1, -2), atol=1e-12, rtol=0):
        raise ValueError('Impermeability must be symmetric.')
    if np.any(np.linalg.eigvalsh(tensor) <= 0):
        raise ValueError('Impermeability must be positive definite.')
    return tensor


def impermeability(principal_indices, orientation=None):
    if np.iscomplexobj(principal_indices): raise ValueError('Complex indices require a separate absorption model.')
    indices = np.asarray(principal_indices, dtype=np.float64)
    if indices.shape[-1:] != (3,) or not np.isfinite(indices).all() or np.any(indices <= 0):
        raise ValueError('Three finite positive principal refractive indices are required.')
    tensor = np.zeros(indices.shape[:-1] + (3, 3), dtype=np.float64)
    tensor[..., np.arange(3), np.arange(3)] = 1/indices**2
    return rotate(tensor, orientation) if orientation is not None else tensor


def rotate(tensor, orientation):
    tensor, orientation = validate_tensor(tensor), validate_frame(orientation)
    return orientation @ tensor @ np.swapaxes(orientation, -1, -2)


def solve_section(tensor, basis, thickness_um):
    """Central section normal to beam; basis columns are right, up, beam.

    The in-plane eigenvectors are D vibration directions. phi is clockwise
    from image right, modulo pi. At a degenerate section it is undefined and
    reported as zero, with axis_defined=False. Thickness is in micrometres.
    """
    tensor, basis = validate_tensor(tensor), validate_frame(basis)
    if np.iscomplexobj(thickness_um): raise ValueError('Thickness must be real.')
    thickness = np.asarray(thickness_um, dtype=np.float64)
    if not np.isfinite(thickness).all() or np.any(thickness <= 0):
        raise ValueError('Thickness must be finite and positive, in micrometres.')
    transverse = basis[..., :, :2]
    section = np.swapaxes(transverse, -1, -2) @ tensor @ transverse
    values, vectors = np.linalg.eigh(section)
    indices = 1/np.sqrt(values)
    fast = vectors[..., :, -1]
    delta = indices[..., 0] - indices[..., 1]
    defined = delta > 1e-12
    phi = np.where(defined, np.arctan2(-fast[..., 1], fast[..., 0]) % np.pi, 0.)
    return dict(phi_rad=phi, extinction_deg=np.rad2deg(phi) % 90,
                n_fast=indices[..., 1], n_slow=indices[..., 0],
                effective_birefringence=delta, gamma_nm=thickness*1000*delta,
                section_impermeability=section, axis_defined=defined)
