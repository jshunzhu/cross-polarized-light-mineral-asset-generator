"""XPL appearance assets without semantic masks or synthetic optical labels."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

from PIL import Image

import atlas
import phenomena

ROOT = Path(__file__).resolve().parents[1]
MODE = 'appearance-only'
CONTRACT = dict(semantic_masks_provided=False, instance_masks_provided=False,
                optical_domain_masks_provided=False, physical_parameter_fields_provided=False,
                physical_model_validated=False, angular_response_validated=False)
NOTICE = phenomena.NO_MASK_NOTICE
RECIPE_KEYS = ('mineral', 'minerals', 'morphology', 'phenomena', 'requested_phenomena',
               'reference_ids', 'evidence', 'grain_count', 'mineral_count', 'phase_count',
               'material_phase_count', 'modality', 'scope')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def output_dir(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError('Generated requests and assets must be outside the skill directory.')
    path.mkdir(parents=True, exist_ok=False)
    return path


def check_contract(document):
    if document.get('mode') != MODE or any(document.get(k) is not v for k, v in CONTRACT.items()):
        raise ValueError('外观模式必须明确不提供语义掩膜及参数真值；请重新prepare，不能改写为物理标签资产。')
    if document.get('semantic_mask_notice') != NOTICE:
        raise ValueError('Missing explicit no-semantic-mask notice.')


def notice_text():
    return ('# 外观生成样本：不提供语义掩膜\n\n' + NOTICE + '\n\n'
            '这是按描述和参考图合成的XPL外观，矿物名称和结构描述为生成意图，须人工看图核查。'
            '不提供像素类别、实例或光学分区标签，也不提供厚度、取向、应变、成分或消光角参数场。'
            '图像中的颜色、结构和角度变化未经物理校准；静态图不能证明消光过程。'
            '人工接受外观不会使该资产获得语义掩膜或物理真值。\n')


def prepare(recipe, out, rock=None, size=1024, requested=None, reference_ids=None, source_profile=None):
    if not 32 <= size <= 4096:
        raise ValueError('Size must be 32–4096.')
    recipe = {key: value for key, value in recipe.items() if key in RECIPE_KEYS}
    if not isinstance(recipe.get('morphology'), str) or not recipe['morphology'].strip():
        raise ValueError('外观模式需要完整description或recipe.morphology，不能省略困难特征。')
    check = phenomena.check_request(None, recipe, requested, reference_ids, mode=MODE)
    if not check['allowed']:
        raise ValueError('不准备外观请求：' + '；'.join(check['errors']))
    declared = list(dict.fromkeys(check['phenomena'] + check['unverified_features']))
    request = dict(version=2, mode=MODE, **CONTRACT, semantic_mask_notice=NOTICE,
                   recipe=recipe, source_profile=source_profile, size=size, created_at=now(),
                   rock_context=rock, requested_phenomena=declared, phenomena_check=check,
                   reference_ids=check['reference_ids'],
                   reference_use_status='selected only; actual inspection and image-tool use must be recorded',
                   source_card=atlas.lookup(recipe['mineral'], atlas.LIBRARY, max_chars=9000),
                   rock_card=atlas.lookup(rock, atlas.LIBRARY, max_chars=5000) if rock else None,
                   parameter_status='appearance synthesis only; no measured or computed pixel labels')
    prompt = ('IMPORTANT: APPEARANCE ONLY. NO SEMANTIC MASKS, INSTANCE MASKS, OPTICAL-DOMAIN MASKS OR LABEL MAPS.\n'
              + NOTICE + '\n'
              'Create a realistic, flat transmitted-light XPL thin-section appearance. '
              'Render the requested interference colors and absorption appearance directly; do not use the grayscale-only optical-texture workflow.\n'
              'Visual priority: photographic credibility against the actual thin-section references. Let the image model generate the complete coherent appearance, including morphology, color, internal structure and microscope texture, at the relative scales seen in those references. '
              'Preserve material continuity across optical boundaries; avoid diagram-like regular patterns or separate planks in place of twin domains. The model output will be kept directly, without procedural striping or physical recoloring.\n'
              'Complete user description (preserve every requested feature and spatial relation):\n'
              + recipe['morphology'] + '\n'
              'Target mineral or assemblage: ' + recipe['mineral'] + '\n'
              'Respect the requested grain count and relationships, including requested inclusions, exsolution, alteration, zoning and deformation. '
              'Do not drop difficult features or add unrequested neighboring grains. '
              'Show the complete requested outline with margin and a true transparent RGBA background. '
              'No 3D facets, surface lighting, labels, arrows, legends, scale bars or segmentation overlays.\n'
              'This is a synthetic appearance candidate, not calibrated optical simulation or measured mineral identification. '
              'A single view cannot establish dynamic extinction, chemical composition or pixel truth.\n')
    if rock:
        prompt += 'Rock context: ' + rock + '\n'
    for feature, guidance in check.get('appearance_guidance', {}).items():
        prompt += ('\nPhenomenon-specific appearance guidance (' + feature + '):\n'
                   + json.dumps(guidance, ensure_ascii=False, indent=2) + '\n')
    if check['reference_images']:
        prompt += ('Inspect these actual reference files, then pass applicable files to image_gen.referenced_image_paths: '
                   + json.dumps(check['reference_images'], ensure_ascii=False) + '\n'
                   'Use only relevant requested features; do not copy annotations, frames or unrequested neighbors.\n')
    prompt += 'Evidence gaps and limits:\n' + '\n'.join(check['warnings']) + '\n'
    target = output_dir(out)
    save(target / 'request.json', request)
    (target / 'imagegen-prompt.txt').write_text(prompt, encoding='utf-8')
    (target / 'NOTICE.md').write_text(notice_text(), encoding='utf-8')
    return dict(request=str(target / 'request.json'), prompt=str(target / 'imagegen-prompt.txt'),
                mode=MODE, **CONTRACT, semantic_mask_notice=NOTICE, phenomena_check=check,
                next='Inspect references, generate the complete XPL appearance, record actual tool use, inspect the image, then build. No semantic masks are provided.')


def build(request_path, texture, out):
    request_path, texture = Path(request_path), Path(texture)
    request = json.loads(request_path.read_text(encoding='utf-8'))
    check_contract(request)
    previous = request.get('phenomena_check', {})
    check_contract(previous)
    if previous.get('allowed') is not True:
        raise ValueError('Appearance request was not approved by preparation.')
    check = phenomena.check_request(None, request['recipe'], request['requested_phenomena'],
                                    request['reference_ids'], mode=MODE)
    if not check['allowed'] or any(check.get(key) != previous.get(key) for key in
                                  ('phenomena', 'unverified_features', 'reference_ids')):
        raise ValueError('Appearance declarations or references changed; prepare a new request.')
    with Image.open(texture) as picture:
        if picture.mode != 'RGBA' or picture.format not in ('PNG', 'WEBP'):
            raise ValueError('Appearance source must be a PNG/WebP with a real RGBA alpha channel.')
        alpha = picture.getchannel('A')
        if alpha.getextrema() == (0, 0) or alpha.getextrema()[0] > 0:
            raise ValueError('Need visible requested material and a transparent background.')
        visible = alpha.point(lambda value: 255 if value > 127 else 0)
        box = visible.getbbox()
        if box is None or box[0] == 0 or box[1] == 0 or box[2] == picture.width or box[3] == picture.height:
            raise ValueError('Visible requested outline must be complete and have canvas margin.')
        source_size = list(picture.size)
        extension = '.png' if picture.format == 'PNG' else '.webp'
    record = request_path.parent / 'reference-use.json'
    if record.is_file() and not isinstance(json.loads(record.read_text(encoding='utf-8')), dict):
        raise ValueError('reference-use.json must be an actual operation record object.')
    target = output_dir(out)
    source_name = 'source' + extension
    shutil.copyfile(texture, target / source_name)
    shutil.copyfile(request_path, target / 'request.json')
    for name in ('imagegen-prompt.txt', 'generation-prompt.txt', 'reference-use.json'):
        if (request_path.parent / name).is_file():
            shutil.copyfile(request_path.parent / name, target / name)
    (target / 'NOTICE.md').write_text(notice_text(), encoding='utf-8')
    metadata = dict(schema_version=2, mode=MODE, **CONTRACT, semantic_mask_notice=NOTICE,
                    asset_id=target.name, mineral=request['recipe']['mineral'], created_at=now(),
                    review_status='pending', automated_checks='structural only; not physical or semantic validation',
                    requested_description=request['recipe']['morphology'],
                    requested_phenomena=request['requested_phenomena'],
                    reference_ids=check['reference_ids'], reference_use_status=request['reference_use_status'],
                    phenomena_check=check, source_image=source_name, source_image_sha256=sha(texture),
                    size=source_size, color_handling='Original source RGB and embedded alpha preserved byte-for-byte.',
                    alpha_semantics='Display transparency only; not semantic labels or a separate mask output.',
                    angle_convention=None, parameter_status=request['parameter_status'],
                    limitations=check['warnings'], files={})
    if record.is_file():
        metadata['reference_use_record'] = record.name
    metadata['files'] = {file.name: sha(file) for file in target.iterdir() if file.is_file()}
    save(target / 'asset.json', metadata)
    preview(target, target / 'preview.html')
    report = validate(target)
    save(target / 'validation.json', report)
    if not report['ok']:
        raise ValueError('Appearance structural validation failed: ' + str(report['errors']))
    return dict(asset=str(target), preview=str(target / 'preview.html'), review_status='pending',
                mode=MODE, **CONTRACT, semantic_mask_notice=NOTICE, validation=report)


def validate(asset):
    asset = Path(asset)
    checks = {}
    try:
        metadata = json.loads((asset / 'asset.json').read_text(encoding='utf-8'))
        request = json.loads((asset / 'request.json').read_text(encoding='utf-8'))
        check_contract(metadata)
        check_contract(request)
        source = phenomena.confined(asset, metadata['source_image'])
        checks['no_semantic_or_optical_outputs'] = not any(
            file.is_file() and (re.search(r'mask|labels?|domains?|fields', file.stem, re.I)
                               or file.suffix.lower() in ('.npz', '.npy')
                               or file.name in ('alpha.png', 'optical-model.json'))
            for file in asset.rglob('*'))
        checks['no_class_id'] = 'class_id' not in metadata and 'class_id' not in request['recipe']
        checks['notice_visible_in_delivery'] = NOTICE in (asset / 'NOTICE.md').read_text(encoding='utf-8')
        checks['preview_emphasizes_no_masks'] = '不提供语义掩膜' in (asset / 'preview.html').read_text(encoding='utf-8')
        checks['source_preserved'] = sha(source) == metadata['source_image_sha256']
        checks['file_hashes'] = all(phenomena.confined(asset, name).is_file() and
                                   sha(phenomena.confined(asset, name)) == digest
                                   for name, digest in metadata['files'].items())
        with Image.open(source) as picture:
            checks['rgba_source'] = picture.mode == 'RGBA' and list(picture.size) == metadata['size']
        errors = [key for key, passed in checks.items() if not passed]
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors = [str(error)]
    return dict(ok=not errors, errors=errors, checks=checks, mode=MODE, **CONTRACT,
                semantic_mask_notice=NOTICE, note='Only file integrity and the no-mask contract are checked; visual and physical truth are not certified.')


def preview(asset, out):
    asset, out = Path(asset), Path(out)
    metadata = json.loads((asset / 'asset.json').read_text(encoding='utf-8'))
    check_contract(metadata)
    source = phenomena.confined(asset, metadata['source_image'])
    with Image.open(source) as picture:
        mime = Image.MIME[picture.format]
    payload = base64.b64encode(source.read_bytes()).decode('ascii')
    text = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>' + html.escape(metadata['asset_id']) + '</title>'
            '<style>body{margin:0;background:#f4f4ef;color:#202922;font:16px/1.7 system-ui,sans-serif}'
            'main{max-width:1060px;margin:auto;padding:28px}h1{font-size:24px;margin:0 0 12px}'
            '.notice{background:#fff0b8;border-left:6px solid #9b6800;padding:18px 22px;border-radius:8px}'
            '.grain{margin:24px 0;background:repeating-conic-gradient(#e1e3de 0% 25%,#fafbf8 0% 50%) 0/24px 24px;'
            'border:1px solid #cbd1c7;border-radius:10px;text-align:center}'
            'img{display:block;max-width:100%;max-height:75vh;margin:auto;object-fit:contain}p{overflow-wrap:anywhere}</style>'
            '<main><div class="notice"><h1>外观生成样本 · 不提供语义掩膜</h1><strong>' + html.escape(NOTICE) + '</strong>'
            '<p>不提供像素类别、实例、光学分区或物理参数场；颜色与角度响应未经物理校准。</p></div>'
            '<p>' + html.escape(metadata['asset_id']) + ' · ' + html.escape(metadata['review_status']) + '</p>'
            '<div class="grain"><img alt="XPL外观图，无语义掩膜" src="data:' + mime + ';base64,' + payload + '"></div>'
            '<h2>生成意图，须看图核查</h2><p>' + html.escape(metadata['requested_description']) + '</p></main></html>')
    with out.open('x', encoding='utf-8') as stream:
        stream.write(text)
    return dict(preview=str(out), mode=MODE, semantic_masks_provided=False, semantic_mask_notice=NOTICE)
