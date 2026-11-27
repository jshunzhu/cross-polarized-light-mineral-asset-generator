"""Build and query the text-only Atlas library. Uses only the Python standard library."""
from __future__ import annotations

import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import unicodedata

from resources import ROOT, REFERENCE_DIR

LIBRARY = REFERENCE_DIR / 'atlas'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(content)


def dump(path, obj):
    write(path, json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def normalize(text):
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9\u3400-\u9fff]+', ' ', text.casefold()).strip()


def slug(text):
    return normalize(text).replace(' ', '-')


def title_of(text):
    aliases = json.loads((ROOT / 'rules' / 'atlas-aliases.json').read_text(encoding='utf-8'))
    for name in sorted(aliases, key=len, reverse=True):
        if text == name or text.startswith(name + ' '): return name
    text = re.split(r'\s+(?=\$|(?:[A-Z][a-z]?)(?:[0-9<(]|\[))', text, maxsplit=1)[0]
    return re.sub(r'\s+(?:Ca|Na|K|Mg|Fe|Be|Zr)$', '', text).strip()


class Rows(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows = []; self.row = []; self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr': self.row = []
        if tag in ('td', 'th'): self.cell = []

    def handle_data(self, data):
        if self.cell is not None: self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(''.join(self.cell).strip()); self.cell = None
        if tag == 'tr' and self.row: self.rows.append(self.row)


def plain(text):
    return html.unescape(re.sub(r'<[^>]*>', '', text)).strip()


def text_only(text):
    return re.sub(r'!\[[^\]]*\]\(([^\n]+?)\)',
                  lambda m: '[Illustration omitted; source image ID: ' + Path(m[1]).name + ']', text)


def compact_card(record, raw):
    """Extract complete source lines/paragraphs; never invent a summary or truncate a sentence."""
    props = []
    for table in re.findall(r'<table>.*?</table>', raw, re.S):
        if 'Optical Properties' not in table: continue
        parser = Rows(); parser.feed(table)
        for row in parser.rows:
            if len(row) > 1 and row[0].endswith(':'):
                props.append((row[0], row[1]))
    labels = ('Color in thin-section:', 'Pleochroism:', 'Birefringence:', 'Relief:', 'Cleavage:',
              'Twinning:', 'Crystal system/optic sign:', 'Other features:', 'Mineralogy:',
              'IUGS classification:', 'Texture:')
    for line in raw.splitlines():
        if line.startswith(labels): props.append(tuple(line.split(':', 1)))
    seen = set(); body = []
    for key, value in props:
        key = key.rstrip(':'); value = value.strip()
        if key not in seen:
            body.append(f'- **{key}:** {value}'); seen.add(key)
    excerpts = []; section = 'Overview'; used = 0; in_math = False
    for line in raw.splitlines():
        if line.strip() == '$$': in_math = not in_math; continue
        if in_math: continue
        if line.startswith('#'):
            section = re.sub(r'^#+\s*', '', line); continue
        if not line or line.startswith(('<', '!', '**Figure', '**Table', '**', '[Illustration')): continue
        if line.startswith(labels) or len(line.split()) < 13: continue
        if re.match(r'^[A-H]\s', line) and ('light' in line.lower() or 'image' in line.lower()): continue
        if used + len(line) <= 1900:
            excerpts.append(f'**{section}:** {line}'); used += len(line)
    src = f"Atlas DOI 10.1017/9781009110020; source lines {record['start_line']}–{record['end_line']}."
    notes = '\n'.join('- ' + x for x in record['notes']) or '- Extracted source excerpts, not an independently verified interpretation.'
    return (f"# {record['title']}\n\n{src}\n\n"
            + '\n'.join(body) + '\n\n' + '\n\n'.join(excerpts)
            + f"\n\n## Limits and evidence\n\n{notes}\n\n"
            + f"Full evidence: `{record['evidence']}`. This card is selective; read evidence for conditions, figures, reactions, and exceptions.\n")


def build(source, out):
    if out.exists(): raise ValueError('Output exists; build into a new directory and compare before replacing it.')
    original = source.read_bytes(); text = original.decode('utf-8-sig').replace('\r\n', '\n'); lines = text.splitlines(keepends=True)
    aliases = json.loads((ROOT / 'rules' / 'atlas-aliases.json').read_text(encoding='utf-8'))
    heads = []
    for i, line in enumerate(lines):
        m = re.match(r'^(#{2,4})\s+(.+)', line)
        if m: heads.append({'start': i, 'level': len(m[1]), 'heading': m[2].strip()})
    # Known shared original entries: do not assign a common property table only to the last mineral.
    group_pairs = {'Sodalite': ['Nosean', 'Haüyne'], 'Anthophyllite': ['Gedrite'],
                   'Richterite–Ferrorichterite': ['Katophorite'], 'Gehlenite': ['Åkermanite']}
    merged = []; skip = set()
    for i, h in enumerate(heads):
        if i in skip: continue
        h['members'] = [title_of(h['heading'])]
        for first, following in group_pairs.items():
            if title_of(h['heading']) != first: continue
            for offset, name in enumerate(following, 1):
                nxt = heads[i + offset]
                if title_of(nxt['heading']) != name: raise ValueError('Shared-entry boundary changed: ' + first)
                h['members'].append(name); skip.add(i + offset)
        merged.append(h)
    heads = merged
    # Confirmed conversion layout: these property tables precede their own headings.
    for i, h in enumerate(heads):
        h['notes'] = []
        if title_of(h['heading']) in ('Orthoclase','Beryl','Xenotime') or h['heading']=='BOX 3.1 Pseudomorphs':
            prev = h['start'] - 1
            while prev >= 0 and not lines[prev].strip(): prev -= 1
            if ('Optical Properties' in lines[prev] or 'Common pseudomorph reactions' in lines[prev]) and lines[prev].startswith('<table>'):
                h['start'] = prev
                h['notes'].append('Conversion layout: the immediately preceding table is assigned to '+title_of(h['heading'])+'; check original layout before quantitative use.')
    bounds = [{'start': 0, 'level': 0, 'heading': 'Front matter', 'members': ['Front matter'], 'notes': []}] + heads
    records = []; chapter = 'front'; section = ''; ids = set()
    for i, h in enumerate(bounds):
        start = h['start']; end = bounds[i + 1]['start'] if i + 1 < len(bounds) else len(lines)
        raw = ''.join(lines[start:end]); heading = h['heading']
        if h['level'] == 2:
            match = re.match(r'([123]) ', heading)
            chapter = match[1] if match else 'front'; section = ''
        if h['level'] == 3: section = heading
        name = ' / '.join(h['members']) if len(h['members']) > 1 else title_of(heading)
        if chapter == 'front': category = 'front'
        elif section == 'Bibliography': category = 'bibliography'
        elif heading.startswith(('BOX ', 'APPLICATION ')): category = 'topics'
        elif h['level'] < 4: category = 'overviews'
        elif chapter == '1': category = 'minerals'
        elif chapter == '2' and section.startswith(('2.6 ', '2.7 ')): category = 'igneous-rocks'
        elif chapter == '3' and section.startswith('3.4 '): category = 'metamorphic-rocks'
        elif chapter == '3' and section.startswith('3.3 '): category = 'facies'
        else: category = 'textures'
        base = slug(name) or f'entry-{start+1}'
        eid = f'{category}/{base}'
        if eid in ids: eid += f'-ch{chapter}-{start+1}'
        ids.add(eid)
        known = []
        for member in h['members']:
            known += aliases.get(member, [])
            known += [member]
        printed = []
        for line in lines[86:347]:
            m = re.match(r'- (.+?)\s+(\d+)\s*$', line)
            if m and normalize(m[1]) in [normalize(x) for x in h['members']]: printed.append(int(m[2]))
        notes = list(h.get('notes', []))
        if len(h['members']) > 1: notes.append('Shared source entry: common and member-specific statements must remain distinct.')
        if len(raw.split()) < 20: notes.append('Heading-only or unusually short source block; inspect adjacent evidence before using it.')
        record = dict(id=eid, title=name, aliases=list(dict.fromkeys(known)), category=category,
                      chapter=chapter, section=section, start_line=start+1, end_line=end,
                      printed_start_pages=sorted(set(printed)), notes=notes,
                      raw_sha256=digest(raw.encode('utf-8')), evidence=f'evidence/{eid}.md', card=f'cards/{eid}.md')
        cooked = text_only(raw)
        header = f"<!-- Source: Atlas DOI 10.1017/9781009110020; lines {start+1}-{end}; source sha256 {digest(original)} -->\n\n"
        evidence = header + cooked
        record['evidence_sha256'] = digest(evidence.encode('utf-8'))
        write(out / record['evidence'], evidence)
        write(out / record['card'], compact_card(record, raw))
        records.append(record)
    # Family navigation is editorial metadata, not a new claim about all members' optical behaviour.
    for family, members in {'Garnet': ['Almandine', 'Andradite', 'Grossular–Hydrogrossular', 'Pyrope', 'Spessartine', 'Uvarovite'],
                            'Feldspar': ['Plagioclase Feldspar', 'Anorthoclase', 'Microcline', 'Orthoclase', 'Sanidine']}.items():
        targets = [r for r in records if r['title'] in members]
        eid = 'families/' + slug(family)
        card = '# ' + family + '\n\nThis is a navigation group, not a single optical or compositional model. Choose a member for generation.\n\n'
        card += '\n'.join(f"- {r['title']}: `{r['card']}`" for r in targets)
        if family == 'Garnet': card += '\n\nPilot: almandine under an explicitly ideal isotropic model. Some other garnet members have anomalous birefringence; do not generalize the pilot to all garnets.\n'
        write(out / f'cards/{eid}.md', card)
        records.append(dict(id=eid, title=family, aliases=[family] + aliases.get(family, []), category='families',
                            card=f'cards/{eid}.md', members=[r['id'] for r in targets]))
    dump(out / 'catalog.json', {'version': 1, 'entries': records})
    dump(out / 'source-map.json', {'source_filename': source.name, 'source_sha256': digest(original),
                                 'source_lines': len(lines), 'doi': '10.1017/9781009110020',
                                 'entries': [{k: r[k] for k in ('id','start_line','end_line','raw_sha256','evidence_sha256','printed_start_pages')} for r in records if 'start_line' in r]})
    write(out / 'INDEX.md', '# Atlas: selective lookup\n\n'
          'Do not load the complete catalog or original book into context. Use `scripts/atlas.py lookup NAME`; default output is a small source-excerpt card. '
          'Use `--mode evidence --section Paragenesis` for source details. A rock context is a separate lookup, not a probability table. '
          'Images are excluded. Captions remain in evidence; original source material stays outside the skill.\n\n'
          'Categories: minerals, families, igneous-rocks, metamorphic-rocks, facies, textures, topics, overviews, bibliography, front. '
          'Chinese aliases cover the mineral and rock entries; English titles are available throughout. '
          'The book primarily covers igneous and metamorphic rocks; missing sedimentary coverage is not evidence of incompatibility.\n\n'
          'Cards contain selective verbatim excerpts. Read full evidence for exceptions and caption context. '
          'Source line ranges are exact; printed start pages come only from explicit contents matches. PDF page numbers are not inferred. '
          'Conversion may displace tables or captions: unusual boundaries are recorded in catalog notes.\n')
    if source.read_bytes() != original: raise RuntimeError('Source changed during build.')
    return validate(source, out)


def validate(source, library):
    manifest = json.loads((library / 'source-map.json').read_text(encoding='utf-8'))
    original = source.read_bytes(); lines = original.decode('utf-8-sig').replace('\r\n', '\n').splitlines(keepends=True)
    errors = []; next_line = 1
    if digest(original) != manifest['source_sha256']: errors.append('Source hash mismatch')
    for r in manifest['entries']:
        if r['start_line'] != next_line: errors.append('Gap or overlap: ' + r['id'])
        raw = ''.join(lines[r['start_line']-1:r['end_line']])
        if digest(raw.encode('utf-8')) != r['raw_sha256']: errors.append('Source slice mismatch: ' + r['id'])
        path = library / f"evidence/{r['id']}.md"
        evidence = path.read_text(encoding='utf-8')
        if digest(evidence.encode('utf-8')) != r['evidence_sha256']: errors.append('Evidence hash mismatch: ' + r['id'])
        # Reconstruct the expected text independently of generated metadata: no text other than image embeds changes.
        if evidence.split('\n\n', 1)[1] != text_only(raw).replace('\r\n', '\n'): errors.append('Evidence differs: ' + r['id'])
        next_line = r['end_line'] + 1
    if next_line != len(lines)+1: errors.append('Incomplete coverage')
    for path in library.rglob('*'):
        if path.is_file() and path.suffix not in ('.md', '.json'): errors.append('Non-text file: ' + str(path))
    return {'ok': not errors, 'errors': errors, 'source_lines': len(lines), 'source_entries':len(manifest['entries']),
            'source_sha256':digest(original), 'image_files_in_library':0}


def lookup(query, library, mode='card', section=None, category=None, max_chars=6500):
    if library == LIBRARY and not (library / 'catalog.json').is_file():
        return 'No local Atlas index installed. Consult a source before interpreting this mineral or rock.'
    entries = json.loads((library / 'catalog.json').read_text(encoding='utf-8'))['entries']
    q = normalize(query)
    pool = [r for r in entries if category is None or r['category'] == category]
    exact = [r for r in pool if q in [normalize(a) for a in r['aliases']+[r['title'],r['id']]]]
    matches = exact or [r for r in pool if any(q in normalize(a) for a in r['aliases']+[r['title'],r['id']])]
    if not q: raise ValueError('Query cannot be empty.')
    if not matches: return 'No matching entry; absence is not evidence of geological incompatibility.'
    if len(matches) > 1:
        return json.dumps({'choose_entry':[{k:r[k] for k in ('id','title','category')} for r in matches[:20]], 'total':len(matches)},ensure_ascii=False,indent=2)
    r = matches[0]
    if mode == 'evidence' and 'evidence' not in r: return 'Family navigation only; choose a specific member.\n' + (library/r['card']).read_text(encoding='utf-8')
    body = (library / r[mode]).read_text(encoding='utf-8')
    if section:
        lines = body.splitlines(); chunks = []; take = False; level = 99
        for line in lines:
            m = re.match(r'^(#{1,6})\s+(.+)',line)
            if m:
                if normalize(section) in normalize(m[2]): take=True; level=len(m[1])
                elif len(m[1]) <= level: take=False
            if take: chunks.append(line)
        if not chunks: return 'Section not found; read the card or choose another section. Available evidence: ' + r.get('evidence',r['card'])
        body = '\n'.join(chunks)
    prefix = f"Entry: {r['id']}\nFile: {library / r[mode]}\n"
    if len(body) > max_chars:
        cut = body.rfind('\n',0,max_chars)
        body = body[:cut if cut>0 else max_chars] + '\n[Output limited; use --section or a larger --max-chars to continue.]'
    return prefix + body


def main():
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build'); b.add_argument('--source',type=Path,required=True); b.add_argument('--out',type=Path,default=LIBRARY)
    v=sub.add_parser('validate'); v.add_argument('--source',type=Path,required=True); v.add_argument('--library',type=Path,default=LIBRARY)
    q=sub.add_parser('lookup'); q.add_argument('query'); q.add_argument('--library',type=Path,default=LIBRARY)
    q.add_argument('--mode',choices=['card','evidence'],default='card'); q.add_argument('--section'); q.add_argument('--category'); q.add_argument('--max-chars',type=int,default=6500)
    args=p.parse_args()
    if args.command=='build': result=build(args.source,args.out)
    elif args.command=='validate': result=validate(args.source,args.library)
    else: print(lookup(args.query,args.library,args.mode,args.section,args.category,args.max_chars)); return
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['ok']: raise SystemExit(1)


if __name__=='__main__': main()
