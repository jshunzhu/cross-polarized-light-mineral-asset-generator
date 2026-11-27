"""Independent physical limits, sensitivity failures, and versioned replay tests.

All textures in this file are numerical test fixtures, not mineral assets.
"""
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.dont_write_bytecode = True
import numpy as np
from PIL import Image

import angle_reference
import image_domains
import quartz_optics
import tensor_optics
import xpl


def sample_fields(size=48, phase='undulose'):
    required = [quartz_optics.DATA, xpl.REFERENCE_DIR/'optics/CIE_xyz_1931_2deg.csv',
                xpl.REFERENCE_DIR/'optics/CIE_std_illum_D65.csv']
    if not all(path.is_file() for path in required):
        raise unittest.SkipTest('Requires local quartz and CIE reference tables.')
    yy, xx = np.indices((size, size))
    alpha = ((xx-(size-1)/2)**2+(yy-(size-1)/2)**2 < (size*.35)**2).astype(np.float32)
    recipe = dict(xpl.PROFILES['quartz-'+phase], thickness_um=30., seed=0)
    phi, gamma, domains, extra, info = quartz_optics.fields(alpha.shape, alpha, recipe)
    fields = dict(alpha=alpha, phi_rad=phi, gamma_nm=gamma, domains=domains,
                  transmission=alpha*.8, **extra)
    base = (xpl.physical_rgb(gamma)*fields['transmission'][..., None]).astype(np.float32)
    return fields, base, recipe, info


class TensorCoreTests(unittest.TestCase):
    def test_known_principal_plane(self):
        result = tensor_optics.solve_section(tensor_optics.impermeability([1.5, 1.6, 1.7]), np.eye(3), 30.)
        self.assertAlmostEqual(float(result['n_fast']), 1.5)
        self.assertAlmostEqual(float(result['n_slow']), 1.6)
        self.assertAlmostEqual(float(result['gamma_nm']), 3000.)
        self.assertAlmostEqual(float(result['phi_rad']), 0.)

    def test_uniaxial_exact_directional_formula(self):
        no, ne = 1.54421, 1.55333
        tau = np.deg2rad([0., 15., 30., 60., 90.])
        rotations = tensor_optics.rotation_matrix(np.zeros(5), tau)
        result = tensor_optics.solve_section(tensor_optics.impermeability([no, no, ne], rotations), np.eye(3), 30.)
        extraordinary = 1/np.sqrt(np.cos(tau)**2/no**2+np.sin(tau)**2/ne**2)
        np.testing.assert_allclose(result['n_fast'], no, atol=1e-14)
        np.testing.assert_allclose(result['n_slow'], extraordinary, atol=1e-14)
        np.testing.assert_allclose(result['gamma_nm'], 30000*(extraordinary-no), atol=2e-11)
        self.assertFalse(bool(result['axis_defined'][0]))

    def test_rotation_covariance(self):
        tensor = tensor_optics.impermeability([1.5, 1.6, 1.7])
        rotation = tensor_optics.rotation_matrix(.7, .9)
        original = tensor_optics.solve_section(tensor, np.eye(3), 25.)
        transformed = tensor_optics.solve_section(tensor_optics.rotate(tensor, rotation), rotation, 25.)
        self.assertAlmostEqual(float(original['gamma_nm']), float(transformed['gamma_nm']), places=9)
        self.assertLess(abs(np.sin(original['phi_rad']-transformed['phi_rad'])), 1e-12)

    def test_analytic_reference_on_random_biaxial_sections(self):
        rng = np.random.default_rng(71)
        indices = np.sort(rng.uniform(1.4, 1.9, (100, 3)), axis=-1)
        rotations = tensor_optics.rotation_matrix(rng.uniform(-np.pi, np.pi, 100), rng.uniform(0, np.pi, 100))
        tensors = tensor_optics.impermeability(indices, rotations)
        result = tensor_optics.solve_section(tensors, np.eye(3), 30.)
        phi, gamma, _, _ = angle_reference.analytic_section(tensors[:, :2, :2], 30.)
        np.testing.assert_allclose(result['gamma_nm'], gamma, atol=3e-11)
        self.assertLess(float(np.abs(np.sin(phi-result['phi_rad'])).max()), 1e-12)

    def test_isotropic_degeneracy(self):
        result = tensor_optics.solve_section(np.eye(3)/1.55**2, np.eye(3), 30.)
        self.assertFalse(bool(result['axis_defined']))
        self.assertEqual(float(result['gamma_nm']), 0.)
        self.assertEqual(float(result['phi_rad']), 0.)

    def test_reject_nonphysical_tensors(self):
        for tensor in (np.diag([1., 0., 1.]), np.diag([1., -1., 1.]),
                       np.array([[1., .1, 0], [0, 1, 0], [0, 0, 1]]), np.full((3, 3), np.nan)):
            with self.subTest(tensor=tensor), self.assertRaises(ValueError):
                tensor_optics.solve_section(tensor, np.eye(3), 30.)

    def test_reject_invalid_frames_and_indices(self):
        for frame in (np.diag([1., 1., -1.]), np.diag([1., 2., 1.]), np.ones((2, 2))):
            with self.assertRaises(ValueError): tensor_optics.rotate(np.eye(3), frame)
        for indices in ([1.5, 0, 1.6], [1.5, np.inf, 1.6], [1.5, 1.6]):
            with self.assertRaises(ValueError): tensor_optics.impermeability(indices)

    def test_reject_invalid_thickness(self):
        for thickness in (0., -2., float('nan')):
            with self.assertRaises(ValueError): tensor_optics.solve_section(np.eye(3), np.eye(3), thickness)

    def test_complex_optics_are_not_silently_discarded(self):
        with self.assertRaises(ValueError): tensor_optics.validate_tensor(np.eye(3, dtype=complex)*(1+.01j))
        with self.assertRaises(ValueError): tensor_optics.impermeability(np.array([1.5, 1.6, 1.7+.01j]))


class QuartzFieldTests(unittest.TestCase):
    def test_coupled_spatial_fields(self):
        fields, _, _, info = sample_fields()
        mask = fields['alpha'] > 0
        self.assertGreater(float(np.std(fields['gamma_nm'][mask])), 1.)
        self.assertGreater(float(np.std(fields['n_slow'][mask])), 1e-5)
        axis = np.stack([fields['optic_axis_'+k][mask] for k in 'xyz'], axis=-1)
        np.testing.assert_allclose(np.linalg.norm(axis, axis=-1), 1., atol=1e-7)
        self.assertLess(info['max_orientation_gradient_deg'], 45.)

    def test_uniform_field_and_fixed_support(self):
        fields, _, _, info = sample_fields(phase='uniform')
        mask = fields['alpha'] > 0
        self.assertEqual(len(np.unique(fields['gamma_nm'][mask])), 1)
        self.assertLess(info['max_orientation_gradient_deg'], 1e-8)
        for key in ('gamma_nm', 'optic_axis_x', 'optic_axis_y', 'optic_axis_z', 'n_fast', 'n_slow'):
            self.assertEqual(float(np.max(np.abs(fields[key][~mask]))), 0.)

    def test_gradient_bound_uses_grain_scale(self):
        results = [sample_fields(size)[3]['max_orientation_gradient_deg'] for size in (48, 96)]
        self.assertLess(abs(results[0]-results[1]), 1.)

    def test_gradient_bound_enforced(self):
        fields, _, recipe, _ = sample_fields()
        recipe['max_orientation_gradient_deg'] = .1
        with self.assertRaisesRegex(ValueError, 'gradient'):
            quartz_optics.fields(fields['alpha'].shape, fields['alpha'], recipe)

    def test_reject_independent_scalar_and_unknown_model(self):
        _, _, recipe, _ = sample_fields()
        for change in (dict(birefringence=.006), dict(orientation_model='future-unknown'),
                       dict(optic_axis_tilt_deg=89., tilt_spread_deg=5.), dict(mineral='Olivine')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                xpl.validate_recipe(dict(recipe, **change))

    def test_optic_axis_along_beam_is_dark(self):
        _, _, recipe, _ = sample_fields(phase='uniform')
        recipe['optic_axis_tilt_deg'] = 0.
        alpha = np.ones((32, 32))
        _, gamma, domains, _, _ = quartz_optics.fields(alpha.shape, alpha, recipe)
        self.assertEqual(float(gamma.max()), 0.)
        self.assertTrue(np.all(domains == 1))


class TwinStabilityTests(unittest.TestCase):
    def setUp(self):
        self.y, self.x = np.indices((48, 48))
        self.alpha = np.zeros((48, 48)); self.alpha[4:-4, 4:-4] = 1.
        self.grey = np.where((self.y//6) % 2, .15, .5)

    def test_clear_domains_are_stable(self):
        uncertainty, report = image_domains.stability(self.grey, self.alpha, 0., 13)
        self.assertTrue(report['stable'])
        self.assertEqual(len(report['perturbations']), 8)
        self.assertGreater(len({p['along_trace_filter_pixels'] for p in report['perturbations']}), 1)
        self.assertLess(report['fraction_std_p95'], .02)
        self.assertTrue(np.all(uncertainty[self.alpha == 0] == 0))

    def test_ambiguous_midtones_fail_sensitivity(self):
        grey = np.full((48, 48), .3); grey[:, 4:6] = .1; grey[:, 42:44] = .5
        _, report = image_domains.stability(grey, self.alpha, 0., 13)
        self.assertFalse(report['stable'])
        self.assertFalse(report['checks']['dominant_domains_stable'])

    def test_deterministic_seed_and_full_report(self):
        a, first = image_domains.stability(self.grey, self.alpha, 0., 17)
        b, second = image_domains.stability(self.grey, self.alpha, 0., 17)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(first, second)

    def test_small_trace_perturbation_reaches_the_estimator(self):
        _, report = image_domains.stability(self.grey, self.alpha, 0., 0)
        directional = [p for p in report['perturbations'] if p['name'].startswith('trace-')]
        self.assertTrue(all(p['mean_abs_fraction_change'] > 1e-5 for p in directional))

    def test_insufficient_contrast_is_rejected(self):
        with self.assertRaises(ValueError): image_domains.stability(np.full((48, 48), .3), self.alpha, 0., 0)

    def test_bounded_diagnostic_grid(self):
        yy, _ = np.indices((288, 288))
        alpha = np.zeros((288, 288)); alpha[8:-8, 8:-8] = 1.
        uncertainty, report = image_domains.stability(np.where((yy//24) % 2, .15, .5), alpha, 0., 0)
        self.assertEqual(report['diagnostic_shape'], [256, 256])
        self.assertEqual(uncertainty.shape, alpha.shape)
        self.assertTrue(np.isfinite(uncertainty).all())


class IndependentAngleTests(unittest.TestCase):
    def test_half_wave_plate_analytic_limit(self):
        result = angle_reference.jones_crossed([0.], [275.], [0., 22.5, 45., 90.], [550.])[:, 0, 0]
        np.testing.assert_allclose(result, [0., .5, 1., 0.], atol=1e-14)

    def test_zero_retardation_and_energy_bound(self):
        zero = angle_reference.jones_crossed([.3, 1.], [0., 0.], np.arange(180.), [400., 550., 700.])
        self.assertLess(float(zero.max()), 1e-28)
        result = angle_reference.jones_crossed([.3, 1.], [180., 1100.], np.arange(180.), [400., 550., 700.])
        self.assertLessEqual(float(result.max()), 1.+1e-14)

    def verify(self, renderer=xpl.response, fields=None, base=None):
        f, b, recipe, info = sample_fields()
        return angle_reference.verify(fields if fields is not None else f, base if base is not None else b,
                                      recipe, info, xpl.REFERENCE_DIR/'optics', renderer)

    def test_direct_jones_and_tensor_reference_pass(self):
        report = self.verify()
        self.assertTrue(report['ok'], report)
        self.assertTrue(report['tensor_fields_checked'])
        self.assertEqual(len(report['angles_deg']), 42)
        self.assertLess(report['max_linear_rgb_error'], 3e-5)

    def test_wrong_angular_frequency_detected(self):
        def wrong(f, b, angle):
            return b*np.sin(np.deg2rad(angle)-f['phi_rad'])[..., None]**2
        report = self.verify(wrong)
        self.assertFalse(report['ok'])
        self.assertFalse(report['checks']['linear_zero_and_fourth_harmonic'])

    def test_constant_gain_error_not_hidden_by_harmonics(self):
        report = self.verify(lambda f, b, a: 1.07*xpl.response(f, b, a))
        self.assertFalse(report['ok'])
        self.assertTrue(report['checks']['linear_zero_and_fourth_harmonic'])
        self.assertFalse(report['checks']['direct_spectral_jones_agreement'])

    def test_phase_offset_detected(self):
        report = self.verify(lambda f, b, a: xpl.response(f, b, a+5.))
        self.assertFalse(report['ok'])
        self.assertTrue(report['checks']['full_cycle_ninety_degree_period'])

    def test_inconsistent_optical_axis_detected(self):
        fields, base, _, _ = sample_fields()
        fields['phi_rad'] = fields['phi_rad']+.1*(fields['alpha'] > 0)
        report = self.verify(fields=fields, base=base)
        self.assertFalse(report['ok'])
        self.assertFalse(report['checks']['tensor_0_fast_axis_analytic'])


class PhysicsPipelineTests(unittest.TestCase):
    def setUp(self):
        required = ('optics/CIE_xyz_1931_2deg.csv', 'optics/CIE_std_illum_D65.csv')
        if any(not (xpl.REFERENCE_DIR / file).is_file() for file in required):
            self.skipTest('Requires the local CIE tables.')
        self.temp = tempfile.TemporaryDirectory(prefix='xpl-physics-v1-')
        self.root = Path(self.temp.name)
        self.texture = self.root/'numerical-fixture.png'
        rgba = np.full((40, 40, 4), 180, dtype=np.uint8)
        rgba[..., 3] = 0; rgba[4:-4, 4:-4, 3] = 255
        Image.fromarray(rgba).save(self.texture)

    def tearDown(self):
        self.temp.cleanup()

    @unittest.skipUnless(all((xpl.REFERENCE_DIR / f).is_file() for f in ('optics/quartz-reference.json', 'phenomena/visual/references.json')), 'Requires local model references.')
    def test_new_quartz_build_rebuild_and_readonly_validation(self):
        request = self.root/'request'; asset = self.root/'asset'
        xpl.prepare('quartz-undulose', request, size=40)
        xpl.build(request/'request.json', self.texture, asset)
        before = {p.name: xpl.sha(p) for p in asset.iterdir() if p.is_file()}
        self.assertTrue(xpl.verify_angles(asset)['ok'])
        self.assertTrue(xpl.validate(asset)['ok'])
        self.assertEqual(before, {p.name: xpl.sha(p) for p in asset.iterdir() if p.is_file()})
        result = xpl.rebuild(asset, 64, self.root/'larger')
        self.assertEqual(result['review_status'], 'pending')
        meta, fields, _ = xpl.load(self.root/'larger')
        self.assertEqual(fields['optic_axis_z'].shape, (64, 64))
        self.assertEqual(meta['source_image_sha256'], xpl.sha(self.texture))
        self.assertEqual(before, {p.name: xpl.sha(p) for p in asset.iterdir() if p.is_file()})

    def test_legacy_scalar_recipe_keeps_original_response(self):
        recipe = dict(mineral='Quartz', class_id=1, phase='undulose', birefringence=.006,
                      orientation_deg=0., spread_deg=9., morphology='Historical numerical fixture.', thickness_um=30., seed=0)
        request = self.root/'legacy.json'
        request.write_text(json.dumps(dict(profile='quartz-undulose', recipe=recipe, size=40, parameter_status='test fixture')))
        xpl.build(request, self.texture, self.root/'legacy')
        _, fields, _ = xpl.load(self.root/'legacy')
        self.assertNotIn('optic_axis_z', fields)
        np.testing.assert_array_equal(fields['gamma_nm'][fields['alpha'] > 0], np.float32(180.))
        self.assertFalse(xpl.verify_angles(self.root/'legacy')['tensor_fields_checked'])

    @unittest.skipUnless(all((xpl.REFERENCE_DIR / f).is_file() for f in ('optics/quartz-reference.json',)), 'Requires local model references.')
    def test_standalone_angle_command_and_report_no_overwrite(self):
        xpl.prepare('quartz-uniform', self.root/'request', size=40)
        xpl.build(self.root/'request/request.json', self.texture, self.root/'asset')
        report = self.root/'angles.json'
        command = [sys.executable, '-B', '-X', 'utf8', str(Path(xpl.__file__)), 'verify-angles',
                   '--asset', str(self.root/'asset'), '--out', str(report)]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(report.read_text())['ok'])
        original = report.read_bytes()
        repeated = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual(original, report.read_bytes())

    @unittest.skipUnless(all((xpl.REFERENCE_DIR / f).is_file() for f in ('optics/quartz-reference.json',)), 'Requires local model references.')
    def test_standalone_angle_failure_returns_nonzero(self):
        xpl.prepare('quartz-uniform', self.root/'request', size=40)
        xpl.build(self.root/'request/request.json', self.texture, self.root/'asset')
        path = self.root/'asset/base-linear.npy'
        base = np.load(path, allow_pickle=False)
        np.save(path, base*1.2, allow_pickle=False)
        result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(Path(xpl.__file__)),
                                 'verify-angles', '--asset', str(self.root/'asset')],
                                capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(json.loads(result.stdout)['ok'])

    @unittest.skipUnless(all((xpl.REFERENCE_DIR / f).is_file() for f in ('optics/burri1967-geneva.json', 'phenomena/visual/references.json')), 'Requires local model references.')
    def test_unstable_twin_build_retains_diagnostics_and_cannot_be_accepted(self):
        grey = np.full((40, 40), .3); grey[:, 4:6] = .1; grey[:, 34:36] = .5
        rgba = np.zeros((40, 40, 4), dtype=np.uint8)
        rgba[..., :3] = np.round(xpl.linear_to_srgb(grey[..., None])*255).astype(np.uint8)
        rgba[4:-4, 4:-4, 3] = 255
        Image.fromarray(rgba).save(self.texture)
        recipe = dict(xpl.PROFILES['plagioclase-twins'], trace_angle_deg=0.)
        path = self.root/'recipe.json'; path.write_text(json.dumps(recipe))
        xpl.prepare(None, self.root/'request', size=40, recipe_path=path)
        asset = self.root/'unstable'
        with self.assertRaisesRegex(ValueError, 'twin_decomposition_stable'):
            xpl.build(self.root/'request/request.json', self.texture, asset)
        report = json.loads((asset/'validation.json').read_text())
        self.assertTrue((asset/'preview.html').is_file())
        self.assertFalse(report['checks']['twin_decomposition_stable'])
        self.assertEqual(json.loads((asset/'asset.json').read_text())['review_status'], 'pending')
        with self.assertRaises(ValueError): xpl.review(asset, 'accepted', 'Numerical failure fixture')


if __name__ == '__main__':
    unittest.main(verbosity=2)
