"""Whole-mineral annotations must include dark twins and preserve the model image."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import mineral_masks


class MineralMasksTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.asset = self.root / 'original'
        self.asset.mkdir()
        rgba = np.zeros((8, 8, 4), np.uint8)
        rgba[2:6, 2:6, :3] = 210
        rgba[2:6, 2:6, 3] = 251
        rgba[3:5, 3:5, :3] = 0  # Dark twins remain mineral, not background.
        rgba[1, 1] = [255, 255, 255, 127]
        rgba[1, 2] = [0, 0, 0, 128]
        Image.fromarray(rgba).save(self.asset / 'source.png')
        self.metadata = {'mineral': 'Plagioclase Feldspar', 'source_image': 'source.png',
                         'source_image_sha256': mineral_masks.sha(self.asset / 'source.png')}
        self.write_metadata()
        self.out = self.root / 'labels'

    def write_metadata(self):
        (self.asset / 'asset.json').write_text(json.dumps(self.metadata), encoding='utf-8')

    def export(self, **kwargs):
        return mineral_masks.export(self.asset, self.out, 2, single_mineral=True, **kwargs)

    def test_dark_twins_and_alpha_edge_are_labeled_without_altering_original(self):
        before = {p.name: p.read_bytes() for p in self.asset.iterdir()}
        self.assertTrue(self.export()['ok'])
        with Image.open(self.out / 'mineral-mask.png') as image:
            labels = np.asarray(image)
        self.assertEqual(labels[3, 3], 2)
        self.assertEqual(labels[1, 1], 0)
        self.assertEqual(labels[1, 2], 2)
        self.assertEqual(labels[0, 0], 0)
        self.assertEqual(set(np.unique(labels)), {0, 2})
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.asset.iterdir()})
        self.assertEqual(before['source.png'], (self.out / 'source.png').read_bytes())

    def test_single_mineral_assertion_is_required_and_multiphase_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Confirm'):
            mineral_masks.export(self.asset, self.out, 2)
        self.metadata['phase_count'] = 2
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'phase_count'):
            self.export()
        self.assertFalse(self.out.exists())

    def test_existing_asset_and_bundle_are_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            mineral_masks.export(self.asset, self.asset, 2, single_mineral=True)
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()

    def test_tampered_mask_cannot_validate_even_if_hash_is_updated(self):
        self.export()
        mask_path = self.out / 'mineral-mask.png'
        with Image.open(mask_path) as image:
            labels = np.array(image)
        labels[3, 3] = 0
        Image.fromarray(labels).save(mask_path)
        info_path = self.out / 'annotation.json'
        info = json.loads(info_path.read_text(encoding='utf-8'))
        info['files']['mineral-mask.png'] = mineral_masks.sha(mask_path)
        info_path.write_text(json.dumps(info), encoding='utf-8')
        self.assertIn('whole_mineral_labels', mineral_masks.validate(self.out)['errors'])

    def test_class_ids_are_not_truncated_to_eight_bits(self):
        mineral_masks.export(self.asset, self.out, 1025, single_mineral=True)
        with Image.open(self.out / 'mineral-mask.png') as image:
            self.assertEqual(np.asarray(image)[3, 3], 1025)

    def test_modified_source_is_refused(self):
        with (self.asset / 'source.png').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.export()
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
