"""Independent angular QA: analytic 2x2 tensors and direct spectral Jones optics.

This module deliberately imports none of xpl, tensor_optics, quartz_optics or
twin_optics. It does not use the production retardation colour lookup table.
The callable renderer is the implementation under test, not the reference.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def analytic_section(section, thickness_um):
    """Closed-form symmetric 2x2 eigenvalues, independent of the core eigensolver."""
    section = np.asarray(section, dtype=np.float64)
    if section.shape[-2:] != (2, 2) or not np.isfinite(section).all():
        raise ValueError('Invalid reference section.')
    if not np.allclose(section, np.swapaxes(section, -1, -2), atol=1e-12, rtol=0):
        raise ValueError('Reference section is not symmetric.')
    a, b, d = section[..., 0, 0], section[..., 0, 1], section[..., 1, 1]
    middle = (a+d)/2
    radius = np.hypot((a-d)/2, b)
    if np.any(middle-radius <= 0):
        raise ValueError('Reference section is not positive definite.')
    fast, slow = 1/np.sqrt(middle+radius), 1/np.sqrt(middle-radius)
    phi = np.where(slow-fast > 1e-12, (-.5*np.arctan2(2*b, a-d)) % np.pi, 0.)
    return phi, (slow-fast)*np.asarray(thickness_um)*1000, fast, slow


def jones_crossed(phi, gamma_nm, angles_deg, wavelengths_nm):
    """Unit input after polarizer, ideal crossed analyzer; no sin-squared shortcut."""
    phi, gamma = np.asarray(phi, dtype=np.float64), np.asarray(gamma_nm, dtype=np.float64)
    theta = np.deg2rad(np.asarray(angles_deg))[:, None]
    # Fast mode in right/up coordinates; analyzer is orthogonal to polarizer.
    fx, fy = np.cos(phi)[None, :], -np.sin(phi)[None, :]
    px, py = np.cos(theta), -np.sin(theta)
    ax, ay = np.sin(theta), np.cos(theta)
    fast_coefficient = (ax*fx+ay*fy)*(fx*px+fy*py)
    slow_coefficient = (ax*(-fy)+ay*fx)*((-fy)*px+fx*py)
    phase = np.exp(2j*np.pi*gamma[:, None]/np.asarray(wavelengths_nm)[None, :])
    amplitude = fast_coefficient[..., None] + slow_coefficient[..., None]*phase[None, ...]
    return np.abs(amplitude)**2


def spectral_weights(color_dir):
    color_dir = Path(color_dir)
    checksums = {'CIE_std_illum_D65.csv': '03d4eb9b837c60671627c946fb534deb',
                 'CIE_xyz_1931_2deg.csv': '17cca777db64b17170f06f67ce9d3ab7'}
    for name, expected in checksums.items():
        if hashlib.md5((color_dir/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Independent reference CIE checksum mismatch: '+name)
    light = np.loadtxt(color_dir/'CIE_std_illum_D65.csv', delimiter=',')
    observer = np.loadtxt(color_dir/'CIE_xyz_1931_2deg.csv', delimiter=',')
    wavelengths = np.arange(380., 781.)
    weights = np.interp(wavelengths, light[:, 0], light[:, 1])[:, None]*np.stack(
        [np.interp(wavelengths, observer[:, 0], observer[:, k]) for k in (1, 2, 3)], axis=-1)
    weights /= np.sum(weights[:, 1])
    return wavelengths, weights


def _direct_rgb(phi, gamma, angles, wavelengths, weights):
    spectrum = jones_crossed(phi, gamma, angles, wavelengths)
    xyz = spectrum @ weights
    # The production display contract clips negative linear-sRGB per component
    # retarder before incoherent area mixing. Use that same declared gamut rule.
    matrix = np.array([[3.2404542, -1.5371385, -.4985314],
                       [-.9692660, 1.8760108, .0415560], [.0556434, -.2040259, 1.0572252]])
    return np.maximum(xyz @ matrix.T, 0.)


def tensor_checks(fields, recipe, info):
    """Check complete spatial fields, not just the angularly sampled pixels."""
    mask = fields['alpha'] > 0
    checks = {}
    references = []
    thickness = recipe['thickness_um']
    if recipe.get('orientation_model') == 'quartz-c-axis-v1':
        if not info or info.get('model') != 'quartz-c-axis-v1':
            raise ValueError('Missing quartz tensor metadata.')
        axis = np.stack([fields['optic_axis_'+k][mask] for k in 'xyz'], axis=-1).astype(np.float64)
        checks['quartz_axis_unit_length'] = bool(np.allclose(np.sum(axis**2, axis=-1), 1., atol=2e-6, rtol=0))
        no, _, ne = info['principal_indices']
        xy = axis[:, :2]
        section = np.eye(2)/no**2 + (1/ne**2-1/no**2)*xy[..., :, None]*xy[..., None, :]
        phi, gamma, fast, slow = analytic_section(section, thickness)
        references.append(('', mask, phi, gamma))
        checks['quartz_indices_from_analytic_section'] = bool(
            np.allclose(fields['n_fast'][mask], fast, atol=3e-7, rtol=0)
            and np.allclose(fields['n_slow'][mask], slow, atol=3e-7, rtol=0))
    elif info and 'mother_impermeability' in info:
        mother = np.asarray(info['mother_impermeability'])
        operation = np.diag([-1., 1., -1.])
        transverse = np.asarray(info['image_basis_right_up_beam'])[:, :2]
        for index, tensor in enumerate((mother, operation@mother@operation.T)):
            phi, gamma, _, _ = analytic_section(transverse.T@tensor@transverse, thickness)
            if 'base_twin' in fields:
                references.append(('_twin' if index else '', mask, phi, gamma))
            else:
                support = mask & ((fields['domains'] > 1) if index else (fields['domains'] == 1))
                references.append(('', support, phi, gamma))
    for index, (suffix, support, phi, gamma) in enumerate(references):
        pkey = 'phi_twin_rad' if suffix else 'phi_rad'
        gkey = 'gamma_twin_nm' if suffix else 'gamma_nm'
        observed_phi, observed_gamma = fields[pkey][support], fields[gkey][support]
        checks[f'tensor_{index}_retardation_analytic'] = bool(np.allclose(observed_gamma, gamma, atol=2e-3, rtol=0))
        # Near an optical axis, the axis azimuth is not identifiable.
        defined = np.broadcast_to(np.asarray(gamma) > 1e-4, observed_gamma.shape)
        difference = np.sin(observed_phi-phi)
        checks[f'tensor_{index}_fast_axis_analytic'] = bool(np.all(np.abs(difference[defined]) < 3e-6))
    return checks


def verify(fields, base, recipe, info, color_dir, renderer, max_samples=257):
    """Compare linear output at 42 angles, with full-field tensor checks.

    Pixel sampling is deterministic and includes parameter extrema. This tests
    the numerical forward model; it is not validation against measured images.
    """
    if not 1 <= max_samples <= 4096:
        raise ValueError('Angular sample count must be between 1 and 4096.')
    mask = fields['alpha'] > 0
    active = np.flatnonzero(mask)
    if not len(active):
        raise ValueError('No active pixels for angular validation.')
    checks = tensor_checks(fields, recipe, info)
    selected = active[np.linspace(0, len(active)-1, min(max_samples, len(active)), dtype=int)]
    for key in ('gamma_nm', 'phi_rad', 'twin_fraction', 'twin_fraction_uncertainty'):
        if key in fields:
            values = fields[key].ravel()[active]
            selected = np.r_[selected, active[np.argmin(values)], active[np.argmax(values)]]
    selected = np.unique(selected)
    sampled = {key: value.reshape((-1, 3))[selected][None, ...] if key == 'base_twin'
               else value.ravel()[selected][None, ...] for key, value in fields.items()}
    sampled_base = base.reshape((-1, 3))[selected][None, ...]
    regular = np.arange(0., 180., 5.)
    angles = np.r_[regular, [7.3, 13.7, 22.5, 89.99, 103.7, 179.9]]
    observed = np.stack([renderer(sampled, sampled_base, float(angle))[0] for angle in angles])
    wavelengths, weights = spectral_weights(color_dir)
    phi = sampled['phi_rad'][0].astype(np.float64)
    gamma = sampled['gamma_nm'][0].astype(np.float64)
    transmission = sampled['transmission'][0].astype(np.float64)
    expected = _direct_rgb(phi, gamma, angles, wavelengths, weights)
    if 'base_twin' in fields:
        fraction = sampled['twin_fraction'][0].astype(np.float64)
        exposure = info['reference_decomposition']['fixed_exposure']
        other = _direct_rgb(sampled['phi_twin_rad'][0], sampled['gamma_twin_nm'][0], angles, wavelengths, weights)
        expected = exposure*transmission[None, :, None]*(expected*(1-fraction)[None, :, None]+other*fraction[None, :, None])
    else:
        expected *= transmission[None, :, None]
    error = float(np.max(np.abs(expected-observed)))
    checks['direct_spectral_jones_agreement'] = bool(np.allclose(observed, expected, atol=3e-5, rtol=3e-4))
    checks['nonnegative_linear_intensity'] = bool(np.min(observed) >= -1e-12)
    scale = max(1., float(np.max(np.abs(observed))))
    design = np.column_stack([np.ones(len(regular)), np.cos(np.deg2rad(4*regular)), np.sin(np.deg2rad(4*regular))])
    series = observed[:len(regular)].reshape(len(regular), -1)
    coefficients = np.linalg.lstsq(design, series, rcond=None)[0]
    residual = float(np.max(np.abs(series-design@coefficients)))
    checks['linear_zero_and_fourth_harmonic'] = residual < 2e-6*scale
    period_error = float(np.max(np.abs(observed[:18]-observed[18:36])))
    checks['full_cycle_ninety_degree_period'] = period_error < 2e-6*scale
    complementary_error = float(np.max(np.abs(observed[:18]+observed[9:27]-(observed[0]+observed[9]))))
    checks['forty_five_degree_complementarity'] = complementary_error < 2e-6*scale
    return dict(version=1, ok=all(checks.values()), checks=checks, sampled_pixels=len(selected),
                angles_deg=angles.tolist(), max_linear_rgb_error=error,
                max_harmonic_residual=residual, max_period_error=period_error,
                max_complementarity_error=complementary_error,
                tensor_fields_checked=bool(any(key.startswith('tensor_') for key in checks)),
                method='Closed-form tensor sections plus complex Jones amplitudes and direct 380–780 nm integration; no production optics imports or colour LUT.',
                scope='Numerical ideal nonabsorbing forward-model QA. No measured angular sequence was used. Legacy scalar recipes only receive angular/spectral checks.')
