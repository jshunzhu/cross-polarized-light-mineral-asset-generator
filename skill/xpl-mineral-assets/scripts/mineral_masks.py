"""Export whole-mineral labels from a confirmed single-mineral RGBA asset."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
from PIL import Image, ImageDraw


KIND = 'single-mineral-semantic-mask'
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_alpha(path):
    with Image.open(path) as image:
        if image.mode != 'RGBA':
            raise ValueError('A source with an explicit RGBA boundary is required.')
        return np.asarray(image)[:, :, 3].copy()


@contextmanager
def staging_directory(parent, name):
    # Inherit the destination ACL: Windows temporary directories can hide exports
    # from the desktop viewer even after a rename into the user's output folder.
    temporary = parent / f'.{name}-{uuid.uuid4().hex}'
    temporary.mkdir()
    try:
        yield temporary
    finally:
        if temporary.exists():
            if temporary.resolve().parent != parent.resolve():
                raise ValueError('Staging cleanup must stay inside the export parent.')
            shutil.rmtree(temporary)


def preview(source, foreground, mineral, class_id, target):
    """A display copy only; neither source nor exported labels are resized."""
    height, width = foreground.shape
    scale = min(1., 640 / max(width, height))
    size = (round(width * scale), round(height * scale))
    y, x = np.indices((size[1], size[0]))
    checker = np.uint8(np.where((x // 16 + y // 16) % 2, 210, 240))
    background = Image.fromarray(np.repeat(checker[:, :, None], 3, axis=2)).convert('RGBA')
    with Image.open(source) as image:
        background.alpha_composite(image.resize(size, Image.Resampling.LANCZOS))
    binary = Image.fromarray(np.uint8(foreground) * 255).resize(size, Image.Resampling.NEAREST)
    canvas = Image.new('RGB', (size[0] * 2 + 24, size[1] + 48), 'white')
    canvas.paste(background.convert('RGB'), (8, 32))
    canvas.paste(binary.convert('RGB'), (size[0] + 16, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), 'Original image on checkerboard', fill='black')
    draw.text((size[0] + 16, 10), f'Whole mineral mask | {mineral} | class {class_id}', fill='black')
    canvas.save(target)


def export(asset, out, class_id, *, single_mineral=False, alpha_threshold=128):
    asset, out = Path(asset).resolve(), Path(out).resolve()
    if not single_mineral:
        raise ValueError('Confirm that all foreground belongs to one mineral with --single-mineral.')
    if isinstance(class_id, bool) or not isinstance(class_id, int) or not 1 <= class_id <= 65535:
        raise ValueError('class_id must be an integer from 1 to 65535; 0 is background.')
    if not isinstance(alpha_threshold, int) or not 1 <= alpha_threshold <= 255:
        raise ValueError('alpha_threshold must be an integer from 1 to 255.')
    if out.exists():
        raise FileExistsError(f'Output already exists: {out}')
    if out == ROOT or ROOT in out.parents or out == asset or asset in out.parents:
        raise ValueError('Export beside the original asset, outside the skill directory.')
    metadata_path = asset / 'asset.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    mineral = metadata['mineral']
    if not isinstance(mineral, str) or not mineral.strip():
        raise ValueError('The source asset must declare its target mineral.')
    request = asset / 'request.json'
    recipe = json.loads(request.read_text(encoding='utf-8')).get('recipe', {}) if request.exists() else {}
    for declaration in (metadata, recipe):
        for key in ('mineral_count', 'phase_count', 'material_phase_count'):
            if key in declaration and declaration[key] != 1:
                raise ValueError(f'{key} contradicts the single-mineral assertion.')
        if 'minerals' in declaration and len(declaration['minerals']) != 1:
            raise ValueError('A multi-mineral source requires separate mineral boundaries.')
    source = (asset / metadata['source_image']).resolve()
    if asset not in source.parents:
        raise ValueError('Source image must be inside the original asset.')
    source_hash = sha(source)
    if source_hash != metadata['source_image_sha256']:
        raise ValueError('Source image differs from the original asset record.')
    alpha = read_alpha(source)
    foreground = alpha >= alpha_threshold
    if alpha.min() != 0 or not foreground.any() or foreground.all():
        raise ValueError('A nonempty mineral and transparent exterior are required.')

    out.parent.mkdir(parents=True, exist_ok=True)
    with staging_directory(out.parent, out.name) as temporary:
        stage = Path(temporary) / 'bundle'
        stage.mkdir()
        source_name = 'source' + source.suffix.lower()
        shutil.copyfile(source, stage / source_name)
        if sha(stage / source_name) != source_hash:
            raise ValueError('Source changed while exporting.')
        Image.fromarray(np.where(foreground, class_id, 0).astype(np.uint16)).save(stage / 'mineral-mask.png')
        Image.fromarray(np.uint8(foreground) * 255).save(stage / 'binary-mask.png')
        Image.fromarray(alpha).save(stage / 'alpha.png')
        preview(stage / source_name, foreground, mineral, class_id, stage / 'preview.png')
        annotation = {
            'schema_version': 1, 'kind': KIND,
            'created_at': datetime.now(timezone.utc).isoformat(), 'review_status': 'pending',
            'mineral': mineral, 'class_id': class_id,
            'classes': {'0': 'background', str(class_id): mineral},
            'semantic_masks_provided': True, 'optical_domain_masks_provided': False,
            'physical_parameter_fields_provided': False, 'angular_response_validated': False,
            'single_mineral_asserted': True,
            'label_provenance': 'Declared synthetic mineral identity and RGBA foreground boundary.',
            'boundary_rule': 'alpha >= alpha_threshold; RGB intensity does not affect labels.',
            'alpha_threshold': alpha_threshold,
            'foreground_pixels': int(foreground.sum()), 'size': [alpha.shape[1], alpha.shape[0]],
            'original_asset': str(asset), 'original_metadata_sha256': sha(metadata_path),
            'source_image': source_name, 'source_image_sha256': source_hash,
            'limitations': [
                '整颗矿物的合成语义标注；双晶暗带与裂隙仍属于同一矿物类别。',
                '不标注内部双晶域、成分或光学参数；边界来自原图alpha及已记录阈值。',
            ],
            'files': {p.name: sha(p) for p in sorted(stage.iterdir())},
        }
        (stage / 'annotation.json').write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        report = validate(stage)
        if not report['ok']:
            raise ValueError(f'Mask export failed validation: {report["errors"]}')
        (stage / 'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        stage.rename(out)
    return {'bundle': str(out), 'semantic_masks_provided': True, 'class_id': class_id, **report}


def validate(bundle):
    bundle = Path(bundle)
    checks = {}
    try:
        info = json.loads((bundle / 'annotation.json').read_text(encoding='utf-8'))
        alpha = read_alpha(bundle / info['source_image'])
        foreground = alpha >= info['alpha_threshold']
        with Image.open(bundle / 'mineral-mask.png') as image:
            labels = np.asarray(image)
        with Image.open(bundle / 'binary-mask.png') as image:
            binary = np.asarray(image)
        with Image.open(bundle / 'alpha.png') as image:
            saved_alpha = np.asarray(image)
        checks['single_mineral_contract'] = (
            info['kind'] == KIND and info['single_mineral_asserted'] is True
            and info['semantic_masks_provided'] is True and 1 <= info['class_id'] <= 65535
            and 1 <= info['alpha_threshold'] <= 255)
        checks['nonempty_foreground_and_background'] = bool(foreground.any() and not foreground.all() and alpha.min() == 0)
        checks['source_preserved'] = sha(bundle / info['source_image']) == info['source_image_sha256']
        checks['dimensions'] = info['size'] == [alpha.shape[1], alpha.shape[0]]
        checks['whole_mineral_labels'] = np.array_equal(labels, np.where(foreground, info['class_id'], 0))
        checks['binary_mask'] = np.array_equal(binary, np.uint8(foreground) * 255)
        checks['alpha_preserved'] = np.array_equal(saved_alpha, alpha)
        checks['foreground_count'] = info['foreground_pixels'] == int(foreground.sum())
        required = {info['source_image'], 'mineral-mask.png', 'binary-mask.png', 'alpha.png', 'preview.png'}
        checks['file_hashes'] = required <= info['files'].keys() and all(
            (bundle / name).is_file() and sha(bundle / name) == digest for name, digest in info['files'].items())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'ok': False, 'errors': [str(exc)], 'checks': checks}
    return {'ok': all(checks.values()), 'errors': [k for k, v in checks.items() if not v],
            'checks': {k: bool(v) for k, v in checks.items()},
            'note': 'Checks label geometry and file integrity; does not validate mineral optical parameters.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    item = sub.add_parser('export')
    item.add_argument('--asset', type=Path, required=True)
    item.add_argument('--out', type=Path, required=True)
    item.add_argument('--class-id', type=int, required=True)
    item.add_argument('--alpha-threshold', type=int, default=128)
    item.add_argument('--single-mineral', action='store_true')
    item = sub.add_parser('validate')
    item.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    result = (export(args.asset, args.out, args.class_id, single_mineral=args.single_mineral,
                     alpha_threshold=args.alpha_threshold) if args.command == 'export' else validate(args.bundle))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
