"""Selective phenomenon lookup and generation preflight; no optical rendering."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

sys.dont_write_bytecode = True
from resources import ROOT, REFERENCE_DIR

RULES = ROOT / 'rules'
DATA_ROOT = REFERENCE_DIR / 'phenomena' if (REFERENCE_DIR / 'phenomena').is_dir() else RULES
REFERENCE_MANIFEST = 'visual/references.json'
STATUSES = {'supported', 'unsupported', 'out_of_scope'}
MODES = ('physical', 'appearance-only')
NO_MASK_NOTICE = '外观生成样本：不提供语义掩膜、实例掩膜或光学分区掩膜；RGBA透明度仅用于显示，不是语义标签。'


def normalize(value):
    value = unicodedata.normalize('NFKC', value).casefold().strip()
    return re.sub(r'[\s_\-\u2010-\u2015]+', ' ', value)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def catalog(library=DATA_ROOT):
    return read_json(Path(library) / 'catalog.json')


def manifest_path(library):
    return confined(library, 'references.json' if Path(library).resolve() == RULES else REFERENCE_MANIFEST)


def reference_records(library=DATA_ROOT):
    return read_json(manifest_path(library))['references']


def evidence_path(relative):
    parts = Path(relative).parts
    if not parts or parts[0] != 'references':
        raise ValueError('Expected an evidence path under references/')
    return confined(REFERENCE_DIR, Path(*parts[1:]))


def confined(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError('Reference path escapes its library: ' + str(relative))
    return path


def matches(query, entries):
    key = normalize(query)
    if not key:
        return []
    exact = [e for e in entries if key in {normalize(a) for a in
             [e['id'], e['name_zh'], *e.get('aliases', [])]}]
    return exact or [e for e in entries if any(key in normalize(a) for a in
                    [e['id'], e['name_zh'], *e.get('aliases', [])])]


def lookup(query, library=DATA_ROOT):
    """Return one small record, or explicit choices; never choose a fallback model."""
    library = Path(library)
    found = matches(query, catalog(library)['entries'])
    if not found:
        return {'query': query, 'status': 'unsupported', 'matches': [],
                'reason': '未收录或名称无法确认；先查证。物理模式不回退通用纹理；外观模式须保留证据缺口并明确不提供语义掩膜。'}
    if len(found) != 1:
        return {'query': query, 'status': 'unsupported', 'reason': '名称不明确，必须先选明现象。',
                'choose_entry': [{k: e[k] for k in ('id', 'name_zh', 'status')} for e in found]}
    result = dict(found[0])
    result['status_scope'] = 'physical model and label eligibility only'
    result['appearance_only'] = dict(eligible=result['scope'] != 'other_modality',
                                    semantic_masks_provided=False, notice=NO_MASK_NOTICE)
    if 'appearance' in result:
        result['appearance_only']['guidance'] = result['appearance']
    if 'card' in result:
        result['card_file'] = str(confined(library, result['card']))
    refs = {r['id']: r for r in reference_records(library)}
    result['references'] = [dict(id=rid, image=str(confined(library, refs[rid]['image'])),
                                 figure_label=refs[rid]['figure_label'],
                                 mode=refs[rid].get('mode'),
                                 generation_reference=refs[rid].get('generation_reference', True),
                                 reference_role=refs[rid].get('reference_role'),
                                 color_reference=refs[rid].get('color_reference'),
                                 source_url=refs[rid].get('source_url'),
                                 evidence_scope='appearance and source caption only; no measured angle/composition')
                            for rid in result.get('reference_ids', []) if rid in refs]
    return result


def string_list(value, label):
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError(label + ' 必须是非空名称组成的列表。')
    return list(value)


def is_negative_mention(text, start):
    prefix = re.split(r'[.;。；!?！？\n]|\b(?:but|however|instead)\b|但是|但要|而要', text[:start])[-1]
    # A negative modifier for one property must not suppress a later affirmative
    # predicate/count: "without inclusions, showing zoning"; "没有包裹体的两颗...".
    affirmations = r'\b(?:with|showing|shows|having|has|contains|containing|displaying|exhibiting|composed of|consisting of)\b|具有|呈现|显示|含有|包含|要求|需要|添加'
    reset = 0
    for match in re.finditer(affirmations, prefix):
        preceding = prefix[:match.start()]
        if not re.search(r'(?:\b(?:not|never|avoid)|do not|不|未|没有|不要)\s*$', preceding):
            reset = match.end()
    for match in re.finditer(r'[,，]\s*(?=(?:one|two|three|four|five|six|seven|eight|nine|ten|several|many|multiple|[2-9]\d*|1\d+)\b)', prefix):
        reset = max(reset, match.end())
    if re.search(r'[,，]\s*$', prefix) and re.match(r'(?:two|three|four|five|six|seven|eight|nine|ten|several|many|multiple|[2-9]\d*|1\d+)\b', text[start:]):
        reset = len(prefix)
    negative_modifier = r'(?:不含|没有|无)[^,，。;；\n]{0,32}(?:包裹体|包体|裂隙|解理|邻晶|背景|阴影|双晶|消光|干涉色)的'
    for match in re.finditer(negative_modifier, prefix):
        reset = max(reset, match.end())
    prefix = prefix[reset:]
    return bool(re.search(r'\b(?:no|without|avoid|exclude|not|never)\b|不含|没有|不要|不得|避免|不加入|禁止|无', prefix))


def explicit_scope_issues(text):
    """Recognize explicit counts even when there is no named texture alias."""
    text = unicodedata.normalize('NFKC', text or '').casefold()
    patterns = (
        r'\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|several|many|multiple|[2-9]\d*|1\d+)\s+(?:[a-z-]+\s+){0,5}(?:grains|crystals|mineral\s+phases|phases)\b',
        r'多个(?:矿物|颗粒|晶粒|晶体|物相)|多颗粒|多晶粒|多矿物|多物相|两相|多相',
        r'(?:[两二三四五六七八九十百千万]+|[2-9]\d*|1\d+)(?:个|颗|种)(?:独立的?|同种的?|相同的?|不同的?)?(?:石英|斜长石|矿物)?(?:颗粒|晶粒|晶体|矿物|物相)',
    )
    for pattern in patterns:
        if any(not is_negative_mention(text, m.start()) for m in re.finditer(pattern, text)):
            return ['文本明确要求多个矿物、颗粒或物相；同一种矿物和同一外轮廓都不能放行。']
    return []


def positive_mentions(text, entries):
    """Catch known explicit terms; this is an aid, not a natural-language proof.

    Whole English words and longest overlapping names avoid perthite/antiperthite
    collisions. Negative morphology instructions must not become requests.
    """
    if not text:
        return []
    text = unicodedata.normalize('NFKC', text).casefold()
    spans = []
    for entry in entries:
        for alias in {entry['name_zh'], *entry.get('aliases', [])}:
            alias = unicodedata.normalize('NFKC', alias).casefold()
            if not alias:
                continue
            pattern = re.escape(alias)
            if re.search('[a-z]', alias):
                pattern = r'(?<![a-z0-9])' + pattern + r'(?![a-z0-9])'
            for match in re.finditer(pattern, text):
                # An entire "no cleavage, twins, inclusions or neighboring grains"
                # enumeration is negative. Do not reinterpret it as positive.
                if is_negative_mention(text, match.start()):
                    continue
                spans.append((match.start(), match.end(), entry['id']))
    selected = []
    occupied = []
    for start, end, feature in sorted(spans, key=lambda m: (-(m[1] - m[0]), m[0])):
        if any(start < b and a < end for a, b in occupied):
            continue
        occupied.append((start, end))
        selected.append((start, feature))
    return list(dict.fromkeys(feature for _, feature in sorted(selected)))


def check_appearance(recipe, requested=None, reference_ids=None, text=None, library=DATA_ROOT):
    """Qualify XPL appearances without claiming optical models or pixel labels."""
    library = Path(library)
    entries = catalog(library)['entries']
    errors, warnings, features, unresolved = [], [], [], []
    if not isinstance(recipe.get('mineral'), str) or not recipe['mineral'].strip():
        errors.append('外观请求须明确目标矿物或矿物组合。')
    modality = normalize(recipe.get('modality', 'XPL'))
    if modality not in {'xpl', '正交偏光', 'cross polarized light', 'crossed polars'}:
        errors.append('外观模式当前生成透射XPL样本；其他成像方式须单独处理。')
    declared = string_list(recipe.get('phenomena'), 'recipe.phenomena')
    declared += string_list(recipe.get('requested_phenomena'), 'recipe.requested_phenomena')
    declared += string_list(requested, '--phenomenon')
    declared += positive_mentions(recipe.get('morphology', ''), entries)
    declared += positive_mentions(text, entries)
    for query in declared:
        found = [entry for entry in matches(query, entries) if normalize(query) in
                 {normalize(a) for a in [entry['id'], entry['name_zh'], *entry.get('aliases', [])]}]
        if len(found) != 1:
            if query not in unresolved:
                unresolved.append(query)
            continue
        entry = found[0]
        if entry['id'] not in features:
            features.append(entry['id'])
        if entry['scope'] == 'other_modality':
            errors.append(entry['name_zh'] + ' 不属于本外观路径的透射XPL观察方式。')
    if unresolved:
        warnings.append('以下特征未在索引精确匹配，须保留完整要求并另行查证，不作为已核实标签：' + '；'.join(unresolved))
    by_id = {entry['id']: entry for entry in entries}
    eligible_refs = {rid for feature in features for rid in by_id[feature].get('reference_ids', [])}
    selected = string_list(recipe.get('reference_ids'), 'recipe.reference_ids')
    selected += string_list(reference_ids, '--reference')
    if not selected:
        selected = [rid for feature in features for rid in by_id[feature].get('reference_ids', [])]
    selected = list(dict.fromkeys(selected))
    refs = {ref['id']: ref for ref in reference_records(library)}
    images = []
    for rid in selected:
        if rid not in refs:
            errors.append('参考图 ID 未收录：' + rid + '。请用 XPL_REFERENCE_DIR 指定含该图的本地参考库。')
            continue
        if rid not in eligible_refs:
            errors.append('参考图与本次已识别的现象不相符：' + rid)
            continue
        path = confined(library, refs[rid]['image'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != refs[rid]['sha256']:
            errors.append('参考图缺失或字节哈希不符：' + rid)
        elif refs[rid].get('generation_reference') is False:
            warnings.append('仅作形态辅助，不作为XPL干涉色目标或自动生图输入：' + rid)
        else:
            images.append(str(path))
    warnings += [NO_MASK_NOTICE, '成分、矿物身份、诊断机制及角度响应均未经物理或像素真值验证；需人工看图。']
    appearance_guidance = {feature: by_id[feature]['appearance'] for feature in features
                           if 'appearance' in by_id[feature]}
    for guidance in appearance_guidance.values():
        warnings.extend(guidance.get('limitations', []))
    if not images:
        warnings.append('本请求未选到可传入的精选真实图；需补查可追溯参考，不能声称已经传图。')
    return dict(policy_version=2, mode='appearance-only', allowed=not errors,
                generation_eligible=not errors, semantic_masks_provided=False,
                instance_masks_provided=False, optical_domain_masks_provided=False,
                physical_parameter_fields_provided=False,
                semantic_mask_notice=NO_MASK_NOTICE, physical_model_validated=False,
                angular_response_validated=False, errors=list(dict.fromkeys(errors)),
                warnings=warnings, phenomena=features, unverified_features=unresolved,
                reference_ids=selected, reference_images=images,
                 appearance_guidance=appearance_guidance,
                physical_support=[dict(id=f, status=by_id[f]['status'], reason=by_id[f]['reason']) for f in features],
                catalog_sha256=hashlib.sha256((library/'catalog.json').read_bytes()).hexdigest(),
                references_sha256=hashlib.sha256(manifest_path(library).read_bytes()).hexdigest(),
                checks=dict(semantic_completeness='must be checked by the skill; preserve the complete request',
                            reference_use='selected only; actual inspection and image-tool use must be recorded',
                            validation_scope='appearance eligibility and reference integrity; no physical or label validation'))


def check_request(profile, recipe, requested=None, reference_ids=None, text=None, library=DATA_ROOT, mode='physical'):
    """Check all declarations together before a request directory is created."""
    if mode not in MODES:
        raise ValueError('Unknown generation mode: ' + str(mode))
    if mode == 'appearance-only':
        return check_appearance(recipe, requested, reference_ids, text, library)
    library = Path(library)
    data = catalog(library)
    entries = data['entries']
    by_id = {e['id']: e for e in entries}
    errors = []
    checks = {}
    phase = recipe.get('phase')
    mineral = recipe.get('mineral', '')
    if not isinstance(mineral, str) or re.search(r'[+|/、]|\band\b|和|与', mineral, re.I):
        errors.append('矿物名称含混合或并列标记，不能据此确认单一物相；请选明单矿物或明确连续固溶体。')
    numerical_demo = phase == 'albite' and profile == 'plagioclase-law-demo'
    declared = string_list(recipe.get('phenomena'), 'recipe.phenomena')
    declared += string_list(recipe.get('requested_phenomena'), 'recipe.requested_phenomena')
    declared += string_list(requested, '--phenomenon')
    selected_refs = string_list(recipe.get('reference_ids'), 'recipe.reference_ids')
    selected_refs += string_list(reference_ids, '--reference')
    errors += explicit_scope_issues(recipe.get('morphology', ''))
    errors += explicit_scope_issues(text)

    for key in ('grain_count', 'mineral_count', 'phase_count', 'material_phase_count'):
        if key in recipe:
            count = recipe[key]
            if type(count) is not int or count < 1:
                errors.append(key + ' 必须是正整数。')
            elif count > 1:
                errors.append(key + ' 大于 1：多矿物、多颗粒或多物相关系超出范围，同一外轮廓也不能放行。')
    if recipe.get('scope') in ('multiple_grains', 'multiple_phases', 'other_modality'):
        errors.append('recipe.scope 超出单矿物单颗粒 XPL 资产范围。')
    if phase == 'twins':
        errors.append('旧 phase=twins 只供历史资产重放，不能准备新的真实双晶资产。')
    if phase == 'albite' and not numerical_demo:
        errors.append('程序条带 albite 只允许 plagioclase-law-demo 数值演示；真实双晶需 albite-reference。')

    if not numerical_demo:
        defaults = data.get('profile_defaults', {}).get(profile)
        if defaults is None:
            defaults = {'uniform': ['uniform-extinction'], 'undulose': ['undulose-extinction'],
                        'isotropic': ['ideal-isotropic'], 'albite-reference': ['albite-twinning']}.get(phase, [])
        declared += defaults
        # Existing pilot silhouettes are part of their established approximate
        # recipes, not new claims that every named habit is independently
        # supported. Scan changed/custom prose; explicit flags are always checked.
        from xpl import PROFILES
        baseline_morphologies = {p['morphology'] for p in PROFILES.values()}
        if recipe.get('morphology', '') not in baseline_morphologies:
            declared += positive_mentions(recipe.get('morphology', ''), entries)
        if text is not None:
            declared += positive_mentions(text, entries)
    elif declared or selected_refs or text:
        errors.append('数值双晶演示没有真实图像生成资格，不能声明为已支持的自然现象或采用生成参考。')

    feature_ids = []
    for query in declared:
        found = matches(query, entries)
        # Only exact identifiers/aliases are admissible for declarations.
        exact = [e for e in found if normalize(query) in {normalize(a) for a in
                 [e['id'], e['name_zh'], *e.get('aliases', [])]}]
        if len(exact) != 1:
            errors.append('现象未确认或存在歧义：' + query + '；不回退到通用生成。')
        elif exact[0]['id'] not in feature_ids:
            feature_ids.append(exact[0]['id'])
    if not feature_ids and not numerical_demo:
        errors.append('没有可确认的现象与模型对应关系；须先查证并声明，不生成外观候选。')

    allowed_refs = set()
    for feature in feature_ids:
        entry = by_id[feature]
        state = entry['status']
        if state != 'supported':
            errors.append(entry['name_zh'] + '：' + entry['reason'])
        if entry.get('compatible_phases') and phase not in entry['compatible_phases']:
            errors.append(entry['name_zh'] + ' 与当前 phase=' + str(phase) + ' 不相容。')
        minerals = entry.get('compatible_minerals', [])
        if minerals and normalize(recipe.get('mineral', '')) not in {normalize(m) for m in minerals}:
            errors.append(entry['name_zh'] + ' 与当前矿物 ' + str(recipe.get('mineral')) + ' 不相容。')
        for other in entry.get('incompatible_with', []):
            if other in feature_ids:
                errors.append(entry['name_zh'] + ' 与 ' + by_id[other]['name_zh'] + ' 的组合不受支持。')
        if state == 'supported':
            allowed_refs.update(entry.get('reference_ids', []))
    refs = {r['id']: r for r in reference_records(library)}
    if not selected_refs:
        selected_refs = [rid for f in feature_ids if by_id[f]['status'] == 'supported'
                         for rid in by_id[f].get('reference_ids', [])]
    selected_refs = list(dict.fromkeys(selected_refs))
    images = []
    for rid in selected_refs:
        if rid not in refs:
            errors.append('参考图 ID 未收录：' + rid + '。请用 XPL_REFERENCE_DIR 指定含该图的本地参考库。')
            continue
        if rid not in allowed_refs:
            errors.append('参考图与本次已支持现象不相符：' + rid)
            continue
        ref = refs[rid]
        path = confined(library, ref['image'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
            errors.append('参考图缺失或字节哈希不符：' + rid)
        elif not ref['generation_reference'] or ref['mode'] != 'XPL':
            errors.append('本图仅作形态辅助，不是XPL生成参考：' + rid)
        else:
            images.append(str(path))
    checks.update(all_declared_features_checked=True, known_text_mentions_checked=True,
                  validation_scope='capability, feature compatibility and reference integrity; xpl.prepare also validates numeric recipe fields',
                  semantic_completeness='must be checked by the skill; keyword matching is not complete',
                  reference_use='selected only; inspect pixels and pass actual files to image_gen before generation',
                  static_images_are_not_angular_or_compositional_measurements=True,
                  numerical_demo_only=numerical_demo)
    return dict(policy_version=1,
                catalog_sha256=hashlib.sha256((library/'catalog.json').read_bytes()).hexdigest(),
                references_sha256=hashlib.sha256(manifest_path(library).read_bytes()).hexdigest(),
                allowed=not errors, generation_eligible=not errors and not numerical_demo,
                errors=list(dict.fromkeys(errors)), phenomena=feature_ids, reference_ids=selected_refs,
                reference_images=images, checks=checks)


def validate_library(library=DATA_ROOT):
    """Verify portable files, hashes, source anchors, and catalog links."""
    library = Path(library)
    if library.resolve() == RULES:
        return dict(ok=False, errors=['Set XPL_REFERENCE_DIR to a local reference library for source and image validation.'])
    errors = []
    try:
        data = catalog(library)
        reference_manifest = read_json(confined(library, REFERENCE_MANIFEST))
        refs = reference_manifest['references']
        if reference_manifest.get('version') != 3:
            raise ValueError('Reference manifest must use version 3.')
        if reference_manifest.get('source_files_bundled') is not False:
            raise ValueError('Reference sources must be recorded as metadata, not bundled copies.')
    except (OSError, ValueError, KeyError) as error:
        return dict(ok=False, errors=[str(error)])
    entries = data.get('entries', [])
    ids = [e.get('id') for e in entries]
    ref_ids = [r.get('id') for r in refs]
    if len(ids) != len(set(ids)) or len(ref_ids) != len(set(ref_ids)):
        errors.append('Duplicate feature or reference IDs.')
    aliases = {}
    card_paths = set()
    for entry in entries:
        try:
            if entry['status'] not in STATUSES:
                errors.append('Unknown status: ' + entry['id'])
            if entry['status'] == 'supported' and entry['scope'] != 'single_grain':
                errors.append('Supported feature has an invalid scope: ' + entry['id'])
            card_path = confined(library, entry['card'])
            if not card_path.is_file():
                errors.append('Missing card: ' + entry['id'])
            else:
                card_paths.add(card_path)
            for alias in [entry['id'], entry['name_zh'], *entry.get('aliases', [])]:
                key = normalize(alias)
                if key in aliases and aliases[key] != entry['id']:
                    errors.append('Ambiguous alias: ' + alias)
                aliases[key] = entry['id']
            for rid in entry.get('reference_ids', []):
                if rid not in ref_ids:
                    errors.append('Unknown reference ' + rid + ' in ' + entry['id'])
            for other in entry.get('incompatible_with', []):
                if other not in ids:
                    errors.append('Unknown incompatibility: ' + other)
            for evidence in entry.get('evidence', []):
                if evidence.get('scope') not in ('specific', 'contextual'):
                    errors.append('Evidence scope missing: ' + entry['id'])
                if 'path' in evidence and not evidence_path(evidence['path']).is_file():
                    errors.append('Missing evidence: ' + evidence['path'])
        except (ValueError, KeyError, OSError) as error:
            errors.append(str(error))
    for name, defaults in data.get('profile_defaults', {}).items():
        if not set(defaults).issubset(ids):
            errors.append('Invalid profile defaults: ' + name)
    for card_path in card_paths:
        for target in re.findall(r'\]\(([^)]+)\)', card_path.read_text(encoding='utf-8')):
            if target.startswith(('http:', 'https:', '#')):
                continue
            relative = target.split('#', 1)[0].strip('<>')
            linked = (card_path.parent/relative).resolve()
            library_root = library.resolve()
            if library_root not in linked.parents and library_root != linked:
                # An alternate data directory (e.g. an integrity-test copy) still
                # uses the installed skill's sibling Atlas/source references.
                linked = ((DATA_ROOT/card_path.relative_to(library_root)).parent/relative).resolve()
            if not linked.is_file() or not any(root == linked or root in linked.parents for root in (REFERENCE_DIR, library_root)):
                errors.append('Broken card link: ' + card_path.name + ' -> ' + target)
    provenance = read_json(REFERENCE_DIR / 'source' / 'provenance.json')
    source = REFERENCE_DIR / 'source' / provenance['filename'] if 'filename' in provenance else REFERENCE_DIR / 'source' / 'Atlas of Minerals and Igneous and Metamorphic Rocks in Thin-Section.md'
    raw = source.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    if source_sha != provenance['sha256']:
        errors.append('Original Atlas source hash mismatch.')
    source_lines = raw.decode('utf-8-sig').splitlines()
    from PIL import Image
    registered_images = set()
    for ref in refs:
        try:
            rid = ref['id']
            path = confined(library, ref['image'])
            registered_images.add(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
                errors.append('Reference image hash mismatch: ' + rid)
            for field in ('original_sha256', 'source_sha256'):
                if not re.fullmatch(r'[0-9a-f]{64}', ref[field]):
                    errors.append('Invalid source fingerprint ' + field + ': ' + rid)
            if not ref['original_filename'] or Path(ref['original_filename']).name != ref['original_filename']:
                errors.append('Invalid original source filename: ' + rid)
            with Image.open(path) as picture:
                if picture.format != 'PNG':
                    errors.append('Reference image must be lossless PNG: ' + rid)
                if picture.size != (ref['width'], ref['height']):
                    errors.append('Reference image dimensions mismatch: ' + rid)
                if picture.mode != ref['pixel_mode']:
                    errors.append('Reference image pixel mode mismatch: ' + rid)
                original_width, original_height = ref['original_width'], ref['original_height']
                if (type(original_width) is not int or type(original_height) is not int or
                        min(original_width, original_height) < 1):
                    raise ValueError('Invalid original image dimensions: ' + rid)
                box = ref['crop_box_xyxy']
                if (not isinstance(box, list) or len(box) != 4 or
                        any(type(value) is not int for value in box) or
                        not (0 <= box[0] < box[2] <= original_width and
                             0 <= box[1] < box[3] <= original_height)):
                    raise ValueError('Invalid source crop bounds: ' + rid)
                if picture.size != (box[2] - box[0], box[3] - box[1]):
                    errors.append('Reference crop dimensions mismatch: ' + rid)
            if ref['mode'] not in ('XPL', 'PPL'):
                errors.append('Activity library contains a non-thin-section modality: ' + rid)
            eligible = ref['mode'] == 'XPL'
            if (ref['generation_reference'] is not eligible or
                    ref['reference_role'] != ('xpl_appearance' if eligible else 'morphology_only')):
                errors.append('Reference modality/role routing mismatch: ' + rid)
            if ref['color_reference'] is not (eligible and ref['pixel_mode'] == 'RGB'):
                errors.append('Reference incorrectly claims XPL color evidence: ' + rid)
            if ref['angular_measurement'] or ref['composition_measurement']:
                errors.append('Static reference falsely marked as a measurement: ' + rid)
            if ref.get('source_kind') == 'external_document':
                evidence = read_json(confined(library, ref['source_evidence']))
                if evidence != ref:
                    errors.append('External source evidence mismatch: ' + rid)
                if not ref.get('source_url', '').startswith('https://'):
                    errors.append('External reference lacks a traceable source URL: ' + rid)
                filename = ref['source_document_filename']
                if not filename or Path(filename).name != filename:
                    errors.append('Invalid source document filename: ' + rid)
                continue
            if ref['source_kind'] != 'atlas' or ref['source_sha256'] != source_sha:
                errors.append('Reference source hashes mismatch: ' + rid)
            evidence = evidence_path(ref['source_evidence']).read_text(encoding='utf-8')
            if ref['caption'] not in evidence or ref['original_filename'] not in evidence:
                errors.append('Evidence/caption/image ID mismatch: ' + rid)
            line = ref['caption_source_line']
            if type(line) is not int or not 1 <= line <= len(source_lines) or source_lines[line - 1].strip() != ref['caption'].strip():
                errors.append('Caption source line mismatch: ' + rid)
            line = ref['image_source_line']
            if type(line) is not int or not 1 <= line <= len(source_lines) or ref['original_filename'] not in source_lines[line - 1]:
                errors.append('Image source line mismatch: ' + rid)
        except (ValueError, KeyError, OSError) as error:
            errors.append(str(error))
    image_directory = confined(library, 'visual/images')
    actual_images = {path.resolve() for path in image_directory.rglob('*') if path.is_file()}
    if actual_images != registered_images:
        errors.append('Active reference images and manifest do not match.')
    return dict(ok=not errors, errors=errors, phenomena=len(entries), references=len(refs),
                categories=dict(Counter(e.get('category') for e in entries)),
                statuses=dict(Counter(e.get('status') for e in entries)), source_sha256=source_sha,
                reference_sources_bundled=False,
                reference_validation_scope='PNG integrity, crop metadata and source citations; '
                                           'original reference files are not bundled or pixel-compared')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    query = sub.add_parser('lookup')
    query.add_argument('query')
    query.add_argument('--max-chars', type=int, default=8000)
    query.add_argument('--library', type=Path, default=DATA_ROOT)
    check = sub.add_parser('check')
    check.add_argument('--mode', choices=MODES, default='physical')
    check.add_argument('--profile')
    check.add_argument('--recipe', type=Path)
    check.add_argument('--phenomenon', action='append', default=[])
    check.add_argument('--reference', action='append', default=[])
    check.add_argument('--text')
    sub.add_parser('validate').add_argument('--library', type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    if args.command == 'lookup':
        result = lookup(args.query, args.library)
        output = json.dumps(result, ensure_ascii=False, indent=2)
        if len(output) > args.max_chars:
            result.pop('references', None)
            output = json.dumps(result, ensure_ascii=False, indent=2)
        print(output)
    elif args.command == 'check':
        if args.recipe:
            recipe = read_json(args.recipe)
        else:
            import xpl
            recipe = dict(xpl.PROFILES.get(args.profile, {}))
        result = check_request(args.profile, recipe, args.phenomenon, args.reference, args.text, mode=args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result['allowed']:
            raise SystemExit(2)
    else:
        result = validate_library(args.library)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result['ok']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
