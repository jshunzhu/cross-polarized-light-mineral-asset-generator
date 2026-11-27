"""Versioned smooth 3-D quartz optic-axis fields; not measured strain fields."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import tensor_optics

MODEL = 'quartz-c-axis-v1'
from resources import REFERENCE_DIR

DATA = REFERENCE_DIR / 'optics' / 'quartz-reference.json'


def validate_settings(recipe):
    if recipe.get('orientation_model') != MODEL:
        raise ValueError('Unknown quartz orientation model; use '+MODEL)
    if recipe.get('mineral') != 'Quartz' or recipe.get('phase') not in ('uniform', 'undulose'):
        raise ValueError('The quartz c-axis model requires Quartz with uniform/undulose phase.')
    if 'birefringence' in recipe:
        raise ValueError('Tensor quartz derives birefringence; remove the independent scalar birefringence.')
    for key in ('optic_axis_tilt_deg', 'tilt_spread_deg', 'max_orientation_gradient_deg'):
        if key not in recipe or not np.isfinite(float(recipe[key])):
            raise ValueError('Missing or nonfinite quartz setting: '+key)
    tilt, spread = float(recipe['optic_axis_tilt_deg']), float(recipe['tilt_spread_deg'])
    if not 0 <= tilt <= 90 or not 0 <= spread <= 30 or not 0 < recipe['max_orientation_gradient_deg'] <= 360:
        raise ValueError('Invalid quartz tilt, tilt spread, or gradient bound.')
    if recipe['phase'] == 'uniform' and (recipe['spread_deg'] != 0 or spread != 0):
        raise ValueError('Uniform quartz cannot have orientation spreads.')
    # The normalized field below has |g| <= 1 inside the visible bounding box.
    if recipe['phase'] == 'undulose' and not spread <= tilt <= 90-spread:
        raise ValueError('Tilt field would cross the allowed 0–90 degree interval.')


def orientation_field(shape, alpha, recipe):
    validate_settings(recipe)
    visible = alpha > .5
    if not visible.any():
        raise ValueError('A visible grain is required for the quartz field scale.')
    ys, xs = np.nonzero(visible)
    yy, xx = np.indices(shape, dtype=np.float64)
    width, height = max(int(xs.max()-xs.min()), 1), max(int(ys.max()-ys.min()), 1)
    x = np.clip(2*(xx-(xs.min()+xs.max())/2)/width, -1, 1)
    y = np.clip(2*(yy-(ys.min()+ys.max())/2)/height, -1, 1)
    azimuth = np.full(shape, float(recipe['orientation_deg']))
    tilt = np.full(shape, float(recipe['optic_axis_tilt_deg']))
    if recipe['phase'] == 'undulose':
        azimuth += recipe['spread_deg']*(.65*x+.35*np.sin(2.5*y+.6*x))
        tilt += recipe['tilt_spread_deg']*(.6*y+.4*np.sin(1.7*x-.4*y))
    phi, tau = np.deg2rad(azimuth), np.deg2rad(tilt)
    # The third column of Rz(pi/2-phi) Ry(tau), without a 9-value frame per pixel.
    axis = np.stack((np.sin(phi)*np.sin(tau), np.cos(phi)*np.sin(tau), np.cos(tau)), axis=-1)
    max_gradient = 0.
    for dim, extent in ((0, height), (1, width)):
        a, b = (axis[:-1], axis[1:]) if dim == 0 else (axis[:, :-1], axis[:, 1:])
        support = (visible[:-1] & visible[1:]) if dim == 0 else (visible[:, :-1] & visible[:, 1:])
        if support.any():
            angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a*b, axis=-1)))
            max_gradient = max(max_gradient, float(angle[support].max())*extent)
    if max_gradient > recipe['max_orientation_gradient_deg'] + 1e-8:
        raise ValueError('Quartz orientation gradient exceeds the recipe bound: '+str(max_gradient))
    return axis, max_gradient


def fields(shape, alpha, recipe):
    axis, gradient = orientation_field(shape, alpha, recipe)
    data = json.loads(DATA.read_text(encoding='utf-8'))
    no, ne = data['n_ordinary'], data['n_extraordinary']
    mask = alpha > 0
    result = {key: np.zeros(shape, dtype=np.float32) for key in ('phi_rad', 'gamma_nm', 'n_fast', 'n_slow')}
    active = np.flatnonzero(mask)
    flat_axis = axis.reshape(-1, 3)
    # Solve bounded batches even for large export grids. Axial symmetry makes
    # R B0 R^T = I/no^2 + (1/ne^2-1/no^2) c c^T, without a rotation gauge.
    for start in range(0, len(active), 65536):
        selected = active[start:start+65536]
        local = flat_axis[selected]
        tensor = np.eye(3)/no**2 + (1/ne**2-1/no**2)*local[..., :, None]*local[..., None, :]
        solved = tensor_optics.solve_section(tensor, np.eye(3), recipe['thickness_um'])
        for key in result: result[key].ravel()[selected] = solved[key]
    phi, gamma = result['phi_rad'], result['gamma_nm']
    domains = mask.astype(np.uint16)
    extras = {key: np.where(mask, values, 0.).astype(np.float32) for key, values in
              [('optic_axis_'+name, axis[..., i]) for i, name in enumerate('xyz')]+
              [('n_fast', result['n_fast']), ('n_slow', result['n_slow'])]}
    info = dict(model=MODEL, material_data=data, material_data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                principal_indices=[no, no, ne], thickness_um=float(recipe['thickness_um']),
                coordinate_system='right-handed image right, image up, beam; phi clockwise in image',
                orientation_representation='unit c-axis; rotation about c is optically unobservable in this uniaxial model',
                max_orientation_gradient_deg=gradient, gradient_units='degrees per visible-grain width or height',
                assumptions=['Synthetic smooth c-axis field, not a measured orientation or strain reconstruction.',
                             'Nominal indices at 600 nm are held nondispersive across the visible spectrum.',
                             'Ideal nonabsorbing normal-incidence linear retarder; optical activity omitted.',
                             'Constant recipe thickness, no arbitrary texture-driven retardation.'])
    return phi, gamma, domains, extras, info
