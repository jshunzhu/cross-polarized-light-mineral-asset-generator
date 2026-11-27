"""Albite-law twins from published optical-axis data, not assigned extinction angles.

The Geneva source diagram uses y as the polar axis (rho), and phi is measured
from +z toward +x. Image angles are clockwise, with x right and y down.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import tensor_optics

from resources import REFERENCE_DIR

DATA = REFERENCE_DIR / 'optics' / 'burri1967-geneva.json'
TWIN = np.diag([-1., 1., -1.])  # 180 degrees around y, normal to (010)


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-10:
        raise ValueError('Degenerate direction.')
    return vector / length


def direction(phi_deg, rho_deg):
    phi, rho = np.deg2rad([phi_deg, rho_deg])
    return np.array([np.sin(rho)*np.sin(phi), np.cos(rho), np.sin(rho)*np.cos(phi)])


def indicatrix(an_percent):
    data = json.loads(DATA.read_text(encoding='utf-8'))
    rows = [r for r in data['rows'] if abs(r['an_percent']-an_percent) < 1e-8]
    if len(rows) != 1:
        raise ValueError('Select a tabulated An percentage; no unverified interpolation.')
    row = rows[0]
    a = direction(row['phi_a_deg'], row['rho_a_deg'])
    b = direction(row['phi_b_deg'], row['rho_b_deg'])
    z = unit(a+b)  # source explicitly identifies A+B with n_gamma
    x = unit(a-b)
    y = unit(np.cross(z, x))
    na, ng = row['n_alpha'], row['n_gamma']
    cos_v = float(np.dot(a, z))
    nb = 1/np.sqrt(cos_v**2/na**2 + (1-cos_v**2)/ng**2)
    basis = np.column_stack([x, y, z])
    tensor = basis @ np.diag(1/np.array([na, nb, ng])**2) @ basis.T
    return tensor, dict(row, n_beta_derived=float(nb), optic_axis_a=a.tolist(),
                        optic_axis_b=b.tolist(), principal_basis_xyz=basis.tolist(),
                        data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                        source_url=data['source_url'])


def section_basis(normal, trace_angle_deg):
    """Columns are physical screen-right, screen-up, and beam directions."""
    normal = unit(normal)
    trace = unit(np.cross([0., 1., 0.], normal))
    perpendicular = unit(np.cross(normal, trace))
    angle = np.deg2rad(-trace_angle_deg)
    right = np.cos(angle)*trace - np.sin(angle)*perpendicular
    up = np.sin(angle)*trace + np.cos(angle)*perpendicular
    return np.column_stack([right, up, normal])


def solve_section(tensor, basis, thickness_um):
    """Central section of the impermeability ellipsoid normal to the wave normal.

    Eigenvectors give transverse D vibration directions at normal incidence.
    No 3-D ellipsoid projection or empirical extinction-angle offset is used.
    """
    result = tensor_optics.solve_section(tensor, basis, thickness_um)
    # Preserve the historical JSON shape for exact replay of stored recipes.
    return {key: value.tolist() if key == 'section_impermeability' else float(value)
            for key, value in result.items() if key != 'axis_defined'}


def model(recipe):
    tensor, data = indicatrix(float(recipe['an_percent']))
    normal = direction(float(recipe['section_phi_deg']), float(recipe['section_rho_deg']))
    basis = section_basis(normal, float(recipe['trace_angle_deg']))
    other = TWIN @ tensor @ TWIN.T
    variants = [solve_section(t, basis, float(recipe['thickness_um'])) for t in (tensor, other)]
    return dict(twin_law='Albite (010), rotation pi about y = normal to (010)',
                coordinate_system='x in (010) perpendicular [001]; y normal (010); z parallel [001]',
                angle_convention='rho from +y; phi from +z toward +x; image x right, y down',
                optical_data=data, mother_impermeability=tensor.tolist(),
                twin_impermeability=other.tolist(), twin_operator=TWIN.tolist(),
                section_normal=normal.tolist(), image_basis_right_up_beam=basis.tolist(),
                variants=variants,
                assumptions=['Tabulated reference data, not a composition measurement of the generated grain.',
                             'Selected synthetic section and pose; geometry is reference-informed, not fitted.',
                             'Nondispersive indices, ideal nonabsorbing orthoscopic normal incidence.',
                             'Each pixel is a single homogeneous retarder through thickness; boundary-overlap ray paths are omitted.',
                             'Lamella widths and optional tips are synthetic morphology, not a growth simulation.'])


def validate_settings(recipe):
    required = ('an_percent', 'section_phi_deg', 'section_rho_deg', 'trace_angle_deg',
                'host_gap_fraction', 'twin_width_fraction', 'width_log_sigma',
                'termination_fraction')
    for key in required:
        if key not in recipe or not np.isfinite(float(recipe[key])):
            raise ValueError('Missing or nonfinite albite setting: '+key)
    if not 0 < recipe['section_rho_deg'] < 180:
        raise ValueError('Section must intersect (010); rho cannot be 0 or 180.')
    if not .002 <= recipe['twin_width_fraction'] <= .2 or not .004 <= recipe['host_gap_fraction'] <= .3:
        raise ValueError('Invalid lamella dimensions relative to grain transverse extent.')
    if not 0 <= recipe['width_log_sigma'] <= 1 or not 0 <= recipe['termination_fraction'] <= 1:
        raise ValueError('Invalid morphological variability.')
    model(recipe)


def fields(shape, alpha, recipe, seed):
    """Reference-informed narrow twin lamellae in a broader host.

    One variant per domain, with no independent random optical directions.
    The projected (010) plane trace fixes the lamella direction. Dimensions
    use visible alpha, so near-invisible image-model traces cannot set scale.
    """
    mask = alpha > 0
    yy, xx = np.indices(shape, dtype=np.float64)
    angle = np.deg2rad(recipe['trace_angle_deg'])
    along = xx*np.cos(angle) + yy*np.sin(angle)
    across = -xx*np.sin(angle) + yy*np.cos(angle)
    visible = alpha > .5
    if not visible.any():
        raise ValueError('No visible alpha support for twin geometry.')
    amin, amax = float(along[visible].min()), float(along[visible].max())
    vmin, vmax = float(across[visible].min()), float(across[visible].max())
    u = (along-amin)/max(amax-amin, 1)
    v = (across-vmin)/max(vmax-vmin, 1)
    rng = np.random.default_rng(seed)
    domains = np.where(mask, 1, 0).astype(np.uint16)
    variant = np.zeros(shape, dtype=np.uint8)
    cursor = -.1
    records = []
    domain = 2
    sigma = float(recipe['width_log_sigma'])
    while cursor < 1.1:
        gap = recipe['host_gap_fraction']*np.clip(rng.lognormal(-sigma*sigma/2, sigma), .3, 3)
        width = recipe['twin_width_fraction']*np.clip(rng.lognormal(-sigma*sigma/2, sigma), .25, 3)
        start = cursor+gap
        end = start+width
        finite = bool(rng.random() < recipe['termination_fraction'])
        lo, hi = (float(rng.uniform(-.12, .15)), float(rng.uniform(.8, 1.12))) if finite else (-1., 2.)
        taper = .06 if finite else 0.
        width_factor = np.clip(np.minimum((u-lo)/max(taper, 1e-6), (hi-u)/max(taper, 1e-6)), 0, 1) if finite else 1.
        member = mask & (np.abs(v-(start+end)/2) < width*width_factor/2) & (u>lo) & (u<hi)
        domains[member] = domain
        variant[member] = 1
        records.append(dict(domain_id=domain, variant=1, center=(start+end)/2,
                            width=width, along_start=lo, along_end=hi, taper=taper))
        domain += 1
        cursor = end
    info = model(recipe)
    phi = np.zeros(shape, dtype=np.float32)
    gamma = np.zeros(shape, dtype=np.float32)
    for index, values in enumerate(info['variants']):
        support = mask & (variant == index)
        phi[support] = values['phi_rad']
        gamma[support] = values['gamma_nm']
    info['geometry'] = dict(trace_angle_deg=recipe['trace_angle_deg'],
                            coordinates='normalized visible-grain extents',
                            host_domain_id=1, twin_lamellae=records)
    return phi, gamma, domains, info


def check_model(info):
    """Independent geometric and eigenproblem invariants for asset validation."""
    b = np.asarray(info['mother_impermeability'])
    bt = np.asarray(info['twin_impermeability'])
    basis = np.asarray(info['image_basis_right_up_beam'])
    checks = dict(twin_involution=np.allclose(TWIN@TWIN, np.eye(3)),
                  twin_preserves_principal_indices=np.allclose(np.linalg.eigvalsh(b), np.linalg.eigvalsh(bt)),
                  twin_tensor_obeys_law=np.allclose(bt, TWIN@b@TWIN.T, atol=1e-14),
                  section_basis_orthonormal=np.allclose(basis.T@basis, np.eye(3), atol=1e-12))
    for index, tensor in enumerate((b, bt)):
        result = solve_section(tensor, basis, 1.)
        checks[f'variant_{index}_extinction_from_tensor'] = abs(np.sin(2*(result['phi_rad']-info['variants'][index]['phi_rad']))) < 1e-10
        checks[f'variant_{index}_retardation_from_tensor'] = abs(result['effective_birefringence']-info['variants'][index]['effective_birefringence']) < 1e-12
    for name in ('optic_axis_a', 'optic_axis_b'):
        normal = np.asarray(info['optical_data'][name])
        right = unit(np.cross(normal, [0., 1., 0.]))
        axes = np.column_stack([right, np.cross(normal, right), normal])
        values = np.linalg.eigvalsh(axes[:, :2].T@b@axes[:, :2])
        checks[name+'_zero_birefringence'] = abs(values[1]-values[0]) < 1e-12
    return {key: bool(value) for key, value in checks.items()}
