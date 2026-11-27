"""Portable single-grain XPL assets: prepare, build, render, preview, validate, review."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True

import numpy as np
from PIL import Image

from atlas import LIBRARY, ROOT, lookup
from resources import REFERENCE_DIR
import twin_optics
import image_domains
import phenomena as phenomenon_policy
import appearance as appearance_assets
import quartz_optics
import angle_reference

PROFILES = {
    'quartz-uniform': dict(mineral='Quartz', class_id=1, phase='uniform', birefringence=0.006,
                           orientation_deg=0., spread_deg=0., morphology='One anhedral, irregular quartz thin-section grain; subtle conchoidal fractures; no cleavage, twins, inclusions or neighboring grains.'),
    'quartz-undulose': dict(mineral='Quartz', class_id=1, phase='undulose', birefringence=0.006,
                            orientation_deg=0., spread_deg=9., morphology='One irregular anhedral quartz thin-section grain; subtle curved fractures; no cleavage, twins, inclusions or neighboring grains. Optical undulose extinction will be added by the renderer, not painted.'),
    'plagioclase-twins': dict(mineral='Plagioclase Feldspar', class_id=2, phase='albite', birefringence=0.0077,
                             an_percent=60., section_phi_deg=130., section_rho_deg=85., trace_angle_deg=40.,
                             host_gap_fraction=.055, twin_width_fraction=.018, width_log_sigma=.55,
                             termination_fraction=.25, orientation_deg=0., spread_deg=0.,
                             morphology='One subhedral tabular plagioclase thin-section grain; restrained fine microtexture; no inclusions or neighboring grains. Do not paint twin bands: albite-law optical lamellae will be computed from published optical-axis data.'),
    'almandine-isotropic': dict(mineral='Almandine', class_id=3, phase='isotropic', birefringence=0.,
                               orientation_deg=0., spread_deg=0., morphology='One rounded to subhedral six-sided almandine garnet thin-section grain; subtle irregular fractures; no cleavage, inclusions or neighboring grains.')
}

PROFILES['plagioclase-reference-twins'] = dict(PROFILES['plagioclase-twins'],phase='albite-reference',
    trace_angle_deg=-17.,morphology='One isolated plagioclase grain in XPL. Generate reference-guided natural albite twin morphology integrated with fine microscope texture, not painted graphic stripes. This is a reference-angle appearance candidate whose optical-domain estimates need visual review.')
PROFILES['plagioclase-law-demo'] = dict(PROFILES['plagioclase-twins'])
PROFILES['plagioclase-twins'] = dict(PROFILES['plagioclase-reference-twins'])

# Explicit recipe versions keep stored assets on their original forward model.
for _name in ('quartz-uniform', 'quartz-undulose'):
    PROFILES[_name] = dict(PROFILES[_name], orientation_model=quartz_optics.MODEL,
                          optic_axis_tilt_deg=55., tilt_spread_deg=4.5 if _name.endswith('undulose') else 0.,
                          max_orientation_gradient_deg=45.)
    PROFILES[_name].pop('birefringence')
for _name in ('plagioclase-twins', 'plagioclase-reference-twins'):
    PROFILES[_name]['twin_stability_version'] = 1


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value, replace=False):
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if replace:
        temp=path.with_suffix(path.suffix+'.tmp')
        with temp.open('x',encoding='utf-8',newline='\n') as f: f.write(text)
        temp.replace(path)
    else:
        with path.open('x',encoding='utf-8',newline='\n') as f: f.write(text)


def output_dir(path):
    resolved=path.resolve()
    if resolved==ROOT or ROOT in resolved.parents:
        raise ValueError('Generated assets must be outside the installed skill directory.')
    resolved.mkdir(parents=True,exist_ok=False)
    return resolved


def srgb_to_linear(rgb):
    return np.where(rgb<=0.04045,rgb/12.92,((rgb+0.055)/1.055)**2.4)


def linear_to_srgb(rgb):
    rgb=np.maximum(rgb,0.)
    return np.where(rgb<=0.0031308,12.92*rgb,1.055*rgb**(1/2.4)-0.055)


def color_table(max_nm=1200., step_nm=0.5):
    """D65/CIE 1931 integration, fixed white normalization; no per-frame normalization."""
    data=REFERENCE_DIR/'optics'
    files={'CIE_std_illum_D65.csv':'03d4eb9b837c60671627c946fb534deb',
           'CIE_xyz_1931_2deg.csv':'17cca777db64b17170f06f67ce9d3ab7'}
    for name,expected in files.items():
        if hashlib.md5((data/name).read_bytes()).hexdigest()!=expected: raise ValueError('CIE data checksum mismatch: '+name)
    d65=np.loadtxt(data/'CIE_std_illum_D65.csv',delimiter=',')
    cmf=np.loadtxt(data/'CIE_xyz_1931_2deg.csv',delimiter=',')
    wave=np.arange(380.,781.,1.)
    light=np.interp(wave,d65[:,0],d65[:,1])
    xyz=np.stack([np.interp(wave,cmf[:,0],cmf[:,k]) for k in (1,2,3)],axis=1)
    weighted=light[:,None]*xyz
    weighted/=weighted[:,1].sum()
    delay=np.arange(0.,max_nm+step_nm,step_nm)
    spectrum=np.sin(np.pi*delay[:,None]/wave[None,:])**2
    values=spectrum@weighted
    matrix=np.array([[3.2404542,-1.5371385,-0.4985314],[-0.9692660,1.8760108,0.0415560],[0.0556434,-0.2040259,1.0572252]])
    return delay,np.maximum(values@matrix.T,0.)


def physical_rgb(gamma):
    delay,table=color_table(float(max(1200.,np.max(gamma)+1.)))
    return np.stack([np.interp(gamma,delay,table[:,k]) for k in range(3)],axis=-1)


def phase_fields(shape,mask,recipe,seed):
    ys,xs=np.nonzero(mask); x0,x1=xs.min(),xs.max(); y0,y1=ys.min(),ys.max()
    yy,xx=np.indices(shape); x=2*(xx-(x0+x1)/2)/max(x1-x0,1); y=2*(yy-(y0+y1)/2)/max(y1-y0,1)
    phase=recipe['phase']; base=float(recipe['orientation_deg']); spread=float(recipe['spread_deg'])
    domains=np.ones(shape,dtype=np.uint16)
    phi=np.full(shape,base,dtype=np.float64)
    if phase=='undulose': phi += spread*(0.65*x+0.35*np.sin(2.5*y+0.6*x))
    elif phase=='twins':
        rng=np.random.default_rng(seed)
        widths=rng.uniform(.12,.22,24); edges=np.cumsum(widths); edges=edges/edges[-1]*4-2
        u=x*np.cos(np.deg2rad(18.))+y*np.sin(np.deg2rad(18.))
        domains=np.searchsorted(edges,u).astype(np.uint16)+1
        phi += np.where(domains%2==0,spread,-spread)
    elif phase not in ('uniform','isotropic'): raise ValueError('Unsupported phase model')
    domains[~mask]=0; phi[~mask]=0
    return np.deg2rad(phi).astype(np.float32),domains


def validate_recipe(recipe):
    required={'mineral','class_id','phase','birefringence','orientation_deg','spread_deg','morphology','thickness_um','seed'}
    tensor_quartz = 'orientation_model' in recipe
    if tensor_quartz: required.remove('birefringence')
    if not required <= recipe.keys(): raise ValueError('Missing recipe fields: '+str(sorted(required-recipe.keys())))
    for key in (('orientation_deg','spread_deg','thickness_um') if tensor_quartz else ('birefringence','orientation_deg','spread_deg','thickness_um')):
        if not np.isfinite(float(recipe[key])): raise ValueError('Nonfinite recipe parameter: '+key)
    if recipe['phase'] not in ('uniform','undulose','twins','albite','albite-reference','isotropic'): raise ValueError('Unsupported phase model.')
    if not 0<recipe['thickness_um']<=100 or (not tensor_quartz and not 0<=recipe['birefringence']<=.2):
        raise ValueError('Invalid birefringence or thickness.')
    if type(recipe['class_id']) is not int or not 1<=recipe['class_id']<=65535: raise ValueError('class_id must be an integer in 1–65535.')
    if tensor_quartz: quartz_optics.validate_settings(recipe)
    if recipe['phase']=='isotropic' and recipe['birefringence']!=0: raise ValueError('The ideal isotropic model requires zero birefringence.')
    if recipe['phase'] in ('twins','undulose') and recipe['spread_deg']<=0: raise ValueError('A spatially varying phase requires positive spread_deg.')
    if recipe['phase'] in ('albite','albite-reference'): twin_optics.validate_settings(recipe)
    if 'twin_stability_version' in recipe and (type(recipe['twin_stability_version']) is not int or recipe['twin_stability_version'] != 1):
        raise ValueError('Unsupported twin stability diagnostic version.')
    if not isinstance(recipe['mineral'],str) or not recipe['mineral'].strip(): raise ValueError('A mineral name is required.')


def prepare(profile,out,rock=None,size=1024,seed=0,recipe_path=None,phenomena=None,reference_ids=None,
            mode='physical',mineral=None,description=None):
    if mode not in phenomenon_policy.MODES: raise ValueError('Unknown generation mode: '+str(mode))
    recipe=json.loads(recipe_path.read_text(encoding='utf-8')) if recipe_path else dict(PROFILES[profile]) if profile else {}
    if mode=='appearance-only':
        if mineral is not None: recipe['mineral']=mineral
        if description is not None: recipe['morphology']=description
        return appearance_assets.prepare(recipe,out,rock,size,phenomena,reference_ids,source_profile=profile)
    if mineral is not None or description is not None:
        raise ValueError('--mineral/--description belong to --mode appearance-only; physical requests require a profile or numerical recipe.')
    if not recipe: raise ValueError('Physical mode requires --profile or --recipe.')
    recipe.setdefault('thickness_um',30.); recipe.setdefault('orientation_deg',0.); recipe.setdefault('spread_deg',0.)
    recipe['seed']=seed; validate_recipe(recipe)
    profile=profile or 'custom'
    feature_check=phenomenon_policy.check_request(profile,recipe,phenomena,reference_ids)
    if not feature_check['allowed']:
        raise ValueError('不生成：'+'；'.join(feature_check['errors']))
    card=lookup(recipe['mineral'],LIBRARY,max_chars=12000)
    rock_card=lookup(rock,LIBRARY,max_chars=6500) if rock else None
    request=dict(version=1,profile=profile,recipe=recipe,rock_context=rock,size=size,
                 created_at=now(),source_card=card,rock_card=rock_card,
                 context_review='unreviewed: assess conditional occurrence before generating; absence from a source is not exclusion',
                 parameter_status='synthetic settings constrained by source ranges; not measurements',
                 phenomena_check=feature_check,requested_phenomena=feature_check['phenomena'],
                 reference_ids=feature_check['reference_ids'],
                 reference_use_status='selected; must inspect and pass actual images to image_gen; not proof of tool use')
    prompt=("Use case: scientific-educational\nAsset type: grayscale transmission microtexture for a reusable mineral grain\n"
            "Primary request: "+recipe['morphology']+"\n"
            "Style/medium: realistic transmitted-light petrographic thin-section texture, flat 2D microscope view, not a 3D crystal photograph.\n"
            "Composition: exactly ONE isolated grain centered, complete outline, generous transparent margin, no text, labels, scale bar, border, shadow or background.\n"
            "Materials/textures: pale neutral grayscale interior with fine low-contrast natural microtexture. No baked-in extinction bands or interference colors. No shiny facets or surface lighting.\n"
            "Optical control: colors, twins and extinction are computed later; avoid painting those effects. The mineral identity and intended features are synthesis intent, not already verified labels.\n"
            +(f"Rock context: {rock}. Reflect this context only where supported by the accompanying source card; do not add other grains.\n" if rock else '')
            +"Output: true transparent background (alpha), one square image.\n")
    if recipe['phase']=='albite-reference':
        prompt=("Use case: scientific-educational\nPrimary request: "+recipe['morphology']+"\n"
                "Inspect real XPL thin-section reference images before generation. Exactly one complete flat low-relief grain, real transparent background. "
                "Fine unequal-width lamellae following the grain, not a barcode or repeated pointed wedges. "
                "Generate authentic first-order gray XPL twin contrast for one reference view, with quiet microtexture. "
                "Twin contacts are internal optical boundaries, not black cracks, grooves or drawn lines. "
                "No neighboring grains, 3D facets, shadows, text, scale bars or opaque inclusions. "
                "This synthetic appearance and its inferred domains remain pending; a single image is not measured optical evidence.\n")
    if feature_check['reference_images']:
        prompt += ('Required real reference files: '+json.dumps(feature_check['reference_images'],ensure_ascii=False)+'\n'
                   'Inspect these actual images and pass appropriate files as referenced_image_paths to image_gen. '
                   'Use only the target feature; exclude reference neighbors, inclusions, scale bars and annotations. '
                   'Reference selection is not proof of actual tool use or measured angular/compositional data.\n')
    if not feature_check['generation_eligible']:
        prompt = ('NUMERICAL DEMONSTRATION ONLY. Do not call image_gen for a realistic mineral asset.\n'
                  'Procedural lamella geometry tests the twin-law calculation; it is not a natural-appearance solution.\n')
    if recipe.get('orientation_model') == quartz_optics.MODEL:
        request['physics_constraints'] = dict(orientation_model=quartz_optics.MODEL,
            thickness_um=recipe['thickness_um'], birefringence='derived from the 3-D optic axis and material tensor',
            max_orientation_gradient_deg=recipe['max_orientation_gradient_deg'],
            status='synthetic constrained parameters, not measured strain')
        prompt += ('Physics v1: the renderer derives extinction and retardation from one smooth 3-D quartz optic-axis field. '
                   'Keep optical bands and colours out of the source microtexture.\n')
    target=output_dir(out); save_json(target/'request.json',request)
    (target/'imagegen-prompt.txt').write_text(prompt,encoding='utf-8')
    return {'request':str(target/'request.json'),'prompt':str(target/'imagegen-prompt.txt'),
            'phenomena_check':feature_check,
            'next':('Inspect source cards and actual reference images, pass suitable images to built-in image_gen with transparent_background=true, record actual reference use, inspect the result, then build.'
                    if feature_check['generation_eligible'] else 'Numerical demonstration only; no realistic image generation is authorized by this request.')}


def build(request_path,texture,out,*,_validated_replay=False):
    request=json.loads(request_path.read_text(encoding='utf-8'))
    if request.get('mode')=='appearance-only': return appearance_assets.build(request_path,texture,out)
    recipe=request['recipe']
    validate_recipe(recipe)
    feature_check=None
    policy_fields=('phenomena_check','requested_phenomena','reference_ids','reference_use_status')
    policy_request=any(key in request for key in policy_fields) or any(key in recipe for key in ('phenomena','requested_phenomena','reference_ids'))
    if policy_request and not _validated_replay:
        if not all(key in request for key in policy_fields) or not isinstance(request['phenomena_check'],dict):
            raise ValueError('New request policy fields are incomplete; prepare again instead of downgrading it to a historical request.')
        if request['phenomena_check'].get('allowed') is not True:
            raise ValueError('Request was not approved by preparation preflight; prepare a new request.')
        feature_check=phenomenon_policy.check_request(request['profile'],recipe,
                      request.get('requested_phenomena'),request.get('reference_ids'))
        if not feature_check['allowed']:
            raise ValueError('不构建：'+'；'.join(feature_check['errors']))
        if (feature_check['phenomena']!=request['phenomena_check'].get('phenomena') or
                feature_check['reference_ids']!=request['phenomena_check'].get('reference_ids')):
            raise ValueError('Request declarations differ from their preparation audit; prepare a new request instead of relabeling it.')
    optics = REFERENCE_DIR / 'optics'
    for filename in ('CIE_std_illum_D65.csv', 'CIE_xyz_1931_2deg.csv'):
        if not (optics / filename).is_file():
            raise FileNotFoundError(f"Physical rendering requires {filename} in {optics}. See README for CIE sources and XPL_REFERENCE_DIR.")
    with Image.open(texture) as original:
        if original.mode!='RGBA': raise ValueError('Texture must have a real alpha channel; do not infer a mask from black or white pixels.')
        source_size=original.size
        size=int(request['size']); image=original.resize((size,size),Image.Resampling.LANCZOS)
    rgba=np.asarray(image,dtype=np.float32)/255.; alpha=rgba[:,:,3]; mask=alpha>0
    if not np.any(mask) or np.all(mask): raise ValueError('Need one nonempty isolated grain on a transparent background.')
    # Some generated PNGs retain alpha=1/255 traces on the canvas edge. Preserve
    # that source coverage; completeness concerns the visible grain, not those traces.
    visible=alpha>.5
    if any(np.any(edge) for edge in (visible[0],visible[-1],visible[:,0],visible[:,-1])): raise ValueError('Visible grain touches the canvas boundary; generate a complete grain with margin.')
    # Grey texture changes achromatic transmission only. Model colors never become physical interference colors.
    linear=srgb_to_linear(rgba[:,:,:3]); grey=linear@np.array([.2126,.7152,.0722])
    trans=np.zeros_like(grey); trans[mask]=np.clip(.88*grey[mask]/max(np.mean(grey[mask]),1e-8),0.,1.)
    if float(np.mean(grey[mask]))<.03: raise ValueError('Texture is too dark for an achromatic transmission source.')
    optical_info=None; extras={}; guided_base=None
    if recipe['phase']=='albite':
        phi,gamma,domains,optical_info=twin_optics.fields(mask.shape,alpha,recipe,recipe['seed'])
    elif recipe['phase']=='albite-reference':
        phi,gamma,domains,trans,guided_base,extras,optical_info=image_domains.decompose(grey,alpha,recipe,recipe['seed'],physical_rgb)
    elif recipe.get('orientation_model') == quartz_optics.MODEL:
        phi,gamma,domains,extras,optical_info=quartz_optics.fields(mask.shape,alpha,recipe)
    else:
        phi,domains=phase_fields(mask.shape,mask,recipe,recipe['seed'])
        gamma=np.where(mask,float(recipe['thickness_um'])*1000*float(recipe['birefringence']),0.).astype(np.float32)
    if recipe['phase']=='isotropic': gamma[:]=0.
    if not np.isfinite(gamma).all() or gamma.min()<0 or gamma.max()>5000: raise ValueError('Unsupported retardation; use 0–5000 nm.')
    reference_record=request_path.parent/'reference-use.json'
    if reference_record.is_file():
        if not isinstance(json.loads(reference_record.read_text(encoding='utf-8')),dict):
            raise ValueError('reference-use.json must contain the actual reference-use record as an object.')
    target=output_dir(out)
    shutil.copyfile(request_path,target/'request.json')
    for prompt_name in ('generation-prompt.txt', 'imagegen-prompt.txt'):
        prompt_path=request_path.parent/prompt_name
        if prompt_path.is_file(): shutil.copyfile(prompt_path,target/prompt_path.name)
    if reference_record.is_file(): shutil.copyfile(reference_record,target/reference_record.name)
    source_name='source'+texture.suffix.lower(); shutil.copyfile(texture,target/source_name)
    np.savez_compressed(target/'fields.npz',alpha=alpha.astype(np.float32),phi_rad=phi,gamma_nm=gamma,
                        transmission=trans.astype(np.float32),domains=domains,**extras)
    Image.fromarray(np.uint8(np.round(alpha*255))).save(target/'alpha.png')
    Image.fromarray(np.where(mask,int(recipe['class_id']),0).astype(np.uint16)).save(target/'mineral-mask.png')
    Image.fromarray(domains).save(target/'optical-domains.png')
    # Store a compact linear-light base for identical Python and browser response.
    base=guided_base if guided_base is not None else (physical_rgb(gamma)*trans[:,:,None]).astype(np.float32)
    np.save(target/'base-linear.npy',base,allow_pickle=False)
    if optical_info: save_json(target/'optical-model.json',optical_info)
    new_physics = bool(recipe.get('orientation_model') or recipe.get('twin_stability_version'))
    metadata=dict(schema_version=2 if new_physics else 1,asset_id=target.name,profile=request['profile'],mineral=recipe['mineral'],
                  class_id=int(recipe['class_id']),rock_context=request.get('rock_context'),size=[size,size],created_at=now(),
                  review_status='pending',automated_checks='not_run',
                  optical_model='ideal nonabsorbing crossed polarizers, achromatic transmission, fixed geometry',
                  color_model='CIE D65 + 1931 2-degree observer; 380–780 nm at 1 nm, fixed white; linear light then sRGB',
                  angle_convention='degrees relative to fixed image axes; zero along x; intensity sin²[2(theta-phi)]',
                  alpha_semantics='coverage, independent of optical intensity; categorical support is alpha > 0',
                  source_edge_note='Generated trace alpha is preserved; completeness checks use alpha > 0.5 at canvas edges.',
                  parameter_status=request['parameter_status'],
                  limitations=['Twin orientations are a synthetic two-domain approximation, not a crystallographic twin-law solution.',
                               'A continuous quartz phase field approximates undulose extinction, not a strain measurement.',
                               'No inclusions, strong absorption, scattering, compensation plates or calibrated camera response.',
                               'Prompt features are intended attributes; human review is needed before using them as verified visual labels.'],
                  source_image=source_name,source_image_sha256=sha(texture),
                  resolution=dict(source_pixels=list(source_size),render_pixels=[size,size],
                                  texture_resampling='Lanczos; larger exports do not recover measured microdetail',
                                  optical_fields='recomputed from recipe at target grid'),
                  context_review=request.get('context_review'),files={})
    for key in ('phenomena_check','requested_phenomena','reference_ids','reference_use_status'):
        if key in request: metadata[key]=request[key]
    if feature_check is not None: metadata['phenomena_build_check']=feature_check
    if reference_record.is_file(): metadata['reference_use_record']=reference_record.name
    if optical_info:
        metadata.update(optical_model='Albite (010) twin tensor; central indicatrix section; ideal nonabsorbing crossed polarizers',
                        optical_model_file='optical-model.json',
                        limitations=optical_info['assumptions']+[
                            'Grayscale image-model microtexture is retained, not a measured PPL transmission map.',
                            'No pericline, Carlsbad, zoning, inclusions, absorption or calibrated microscope response.'])
        if recipe['phase']=='albite-reference':
            metadata.update(optical_model='Albite-law tensor pair with image-estimated twin area fractions; incoherent two-retarder mixing',
                            optical_domain_label_status='Estimated from a generated reference-angle image; not pixel-level ground truth.',
                            reference_angle_deg=optical_info['reference_decomposition']['reference_angle_deg'])
            if 'decomposition_stability' in optical_info:
                metadata['twin_decomposition_stability'] = optical_info['decomposition_stability']
        elif recipe.get('orientation_model') == quartz_optics.MODEL:
            metadata.update(optical_model='Quartz 3-D c-axis field and shared impermeability-tensor section solver',
                            orientation_model=quartz_optics.MODEL,
                            limitations=optical_info['assumptions']+[
                                'Grayscale image-model texture is not measured PPL transmission.',
                                'Numerical angular QA does not establish visual realism or agreement with a measured sequence.'])
    metadata['files']={p.name:sha(p) for p in target.iterdir() if p.is_file()}
    save_json(target/'asset.json',metadata)
    report=validate(target)
    metadata['automated_checks']='passed' if report['ok'] else 'failed'
    save_json(target/'asset.json',metadata,True); save_json(target/'validation.json',report)
    if not report['ok']:
        preview(target,target/'preview.html')
        raise ValueError('Asset validation failed: '+str(report['errors'])+'; diagnostics: '+str(target/'validation.json'))
    preview(target,target/'preview.html')
    render(target,[0,15,30,45,60,75],target/'inspection-frames')
    return {'asset':str(target),'review_status':'pending','validation':report,'preview':str(target/'preview.html')}


def load(asset):
    metadata=json.loads((asset/'asset.json').read_text(encoding='utf-8'))
    if metadata.get('mode')=='appearance-only':
        raise ValueError('外观资产不提供语义掩膜或光学参数场；不能使用物理render/rebuild推算角度或标签。新增视角须生成并另行标明为未校准外观。')
    with np.load(asset/'fields.npz',allow_pickle=False) as z: fields={k:z[k] for k in z.files}
    base=np.load(asset/'base-linear.npy',allow_pickle=False)
    return metadata,fields,base


def response(fields,base,angle):
    factor=np.sin(2*(np.deg2rad(np.asarray(angle,dtype=np.float64))-fields['phi_rad'].astype(np.float64)))**2
    result=base*factor[:,:,None]
    if 'base_twin' in fields:
        factor_twin=np.sin(2*(np.deg2rad(np.asarray(angle,dtype=np.float64))-fields['phi_twin_rad'].astype(np.float64)))**2
        result=result+fields['base_twin']*factor_twin[:,:,None]
    return result


def rgba_at(fields,base,angle):
    rgb=np.round(np.clip(linear_to_srgb(response(fields,base,angle)),0,1)*255).astype(np.uint8)
    rgb[fields['alpha']==0]=0
    return np.dstack([rgb,np.uint8(np.round(fields['alpha']*255))])


def render(asset,angles,out):
    if not angles or not all(np.isfinite(float(a)) for a in angles): raise ValueError('Angles must be finite.')
    metadata,fields,base=load(asset); target=output_dir(out)
    manifest=[]
    for i,angle in enumerate(angles):
        name=f'frame-{i:04d}.png'; Image.fromarray(rgba_at(fields,base,angle)).save(target/name)
        manifest.append({'file':name,'angle_deg':float(angle),'sha256':sha(target/name)})
    for name in ('mineral-mask.png','optical-domains.png','alpha.png'): shutil.copyfile(asset/name,target/name)
    save_json(target/'frames.json',{'asset_id':metadata['asset_id'],'geometry':'fixed','frames':manifest})
    return {'out':str(target),'frames':len(manifest)}


def rebuild(asset,size,out):
    """New grid from the unchanged source and exact stored recipe; never overwrite."""
    if not 32<=size<=4096: raise ValueError('Size must be 32–4096.')
    report=validate(asset)
    if not report['ok']: raise ValueError('Source asset failed validation.')
    metadata,_,_=load(asset)
    request=json.loads((asset/'request.json').read_text(encoding='utf-8'))
    request.update(size=size,created_at=now(),rebuild_of=dict(asset=str(asset.resolve()),
                   asset_metadata_sha256=sha(asset/'asset.json'),source_size=metadata['size'],
                   note='Same original texture and recipe; no recovered or newly generated texture detail.'))
    with tempfile.TemporaryDirectory(prefix='xpl-rebuild-') as temp:
        path=Path(temp)/'request.json'; save_json(path,request)
        for prompt_name in ('generation-prompt.txt', 'imagegen-prompt.txt'):
            if (asset/prompt_name).is_file(): shutil.copyfile(asset/prompt_name,Path(temp)/prompt_name)
        if (asset/'reference-use.json').is_file():
            shutil.copyfile(asset/'reference-use.json',Path(temp)/'reference-use.json')
        # Only this validated, source-preserving path grants replay. A public
        # request's rebuild_of JSON is lineage metadata, never a bypass token.
        return build(path,asset/metadata['source_image'],out,_validated_replay=True)


def preview(asset,out):
    if out.exists(): raise FileExistsError('Preview exists; choose a new filename.')
    if json.loads((asset/'asset.json').read_text(encoding='utf-8')).get('mode')=='appearance-only':
        return appearance_assets.preview(asset,out)
    metadata,fields,base=load(asset)
    stride=max(1,int(np.ceil(max(fields['alpha'].shape)/768)))
    base=base[::stride,::stride]
    fields={k:v[::stride,::stride] for k,v in fields.items()}
    h,w=fields['alpha'].shape
    # Raw float arrays avoid browser PNG color-management changing scientific values.
    payload={k:base64.b64encode(np.asarray(v,dtype='<f4').tobytes()).decode('ascii')
             for k,v in {'base':base,'phi':fields['phi_rad'],'alpha':fields['alpha'],'domains':fields['domains']}.items()}
    if 'base_twin' in fields:
        payload.update(baseTwin=base64.b64encode(np.asarray(fields['base_twin'],dtype='<f4').tobytes()).decode('ascii'),
                       phiTwin=base64.b64encode(np.asarray(fields['phi_twin_rad'],dtype='<f4').tobytes()).decode('ascii'))
    if 'twin_fraction_uncertainty' in fields:
        payload['twinUncertainty']=base64.b64encode(np.asarray(fields['twin_fraction_uncertainty'],dtype='<f4').tobytes()).decode('ascii')
    payload.update(w=w,h=h,title=metadata['asset_id'],metadata=metadata,preview_sample_stride=stride)
    template=(ROOT/'references'/'preview-template.html').read_text(encoding='utf-8')
    text=template.replace('__ASSET_DATA__',json.dumps(payload,ensure_ascii=True).replace('</','<\\/'))
    with out.open('x',encoding='utf-8') as f: f.write(text)
    return {'preview':str(out)}


def verify_angles(asset, out=None):
    metadata,fields,base=load(asset)
    recipe=json.loads((asset/'request.json').read_text(encoding='utf-8'))['recipe']
    info=json.loads((asset/metadata['optical_model_file']).read_text(encoding='utf-8')) if metadata.get('optical_model_file') else None
    report=angle_reference.verify(fields,base,recipe,info,REFERENCE_DIR/'optics',response)
    if out is not None: save_json(out,report)
    return report


def validate(asset):
    if json.loads((asset/'asset.json').read_text(encoding='utf-8')).get('mode')=='appearance-only':
        return appearance_assets.validate(asset)
    metadata,fields,base=load(asset); errors=[]; checks={}
    alpha=fields['alpha']; mask=alpha>0; shape=alpha.shape
    checks['nonempty_grain']=bool(mask.any() and (~mask).any())
    checks['finite_fields']=all(np.isfinite(v).all() for v in fields.values()) and bool(np.isfinite(base).all())
    checks['same_geometry']=all(v.shape==(shape+(3,) if k=='base_twin' else shape) for k,v in fields.items()) and base.shape==shape+(3,)
    checks['field_ranges']=bool(np.all((alpha>=0)&(alpha<=1)) and np.all(fields['gamma_nm']>=0)
                                and np.all((fields['transmission']>=0)&(fields['transmission']<=1)))
    checks['domain_support']=bool(np.array_equal(fields['domains']>0,mask))
    classes=np.asarray(Image.open(asset/'mineral-mask.png'))
    checks['categorical_labels']=bool(np.array_equal(classes>0,mask) and np.all(classes[mask]==metadata['class_id']))
    checks['alpha_saved_exactly']=bool(np.array_equal(np.asarray(Image.open(asset/'alpha.png')),np.uint8(np.round(alpha*255))))
    checks['domains_saved_exactly']=bool(np.array_equal(np.asarray(Image.open(asset/'optical-domains.png')),fields['domains']))
    checks['ninety_degree_period']=bool(np.allclose(response(fields,base,13.7),response(fields,base,103.7),atol=2e-7,rtol=1e-6))
    # Local phi can vary: test analytic extinction and max per pixel, without assuming one global grain extinction.
    angle=np.rad2deg(fields['phi_rad'].astype(np.float64))
    if 'base_twin' not in fields:
        checks['local_extinction']=bool(np.max(response(fields,base,angle))<1e-12)
        checks['local_45_degree_max']=bool(np.allclose(response(fields,base,angle+45),base,atol=1e-12))
    else:
        checks['mixture_is_bounded']=bool(all(np.all(response(fields,base,a)<=base+fields['base_twin']+1e-7) for a in (0,19,45,81)))
    checks['alpha_independent_of_angle']=bool(all(np.array_equal(rgba_at(fields,base,a)[:,:,3],np.uint8(np.round(alpha*255))) for a in (0,19,45,89)))
    expected=physical_rgb(fields['gamma_nm'])*fields['transmission'][:,:,None]
    if 'base_twin' in fields:
        info=json.loads((asset/metadata['optical_model_file']).read_text(encoding='utf-8'))
        exposure=info['reference_decomposition']['fixed_exposure']
        fraction=fields['twin_fraction'][:,:,None]
        expected=expected*(1-fraction)*exposure
        expected_twin=physical_rgb(fields['gamma_twin_nm'])*fields['transmission'][:,:,None]*fraction*exposure
        checks['twin_base_matches_optical_parameters']=bool(np.allclose(fields['base_twin'],expected_twin,atol=2e-7,rtol=1e-5))
        checks['twin_fractions_in_range']=bool(np.all((fraction>=0)&(fraction<=1)))
        checks['reference_anchor_preserved_in_luminance']=info['reference_decomposition']['linear_luminance_mae']<.003
    checks['base_matches_optical_parameters']=bool(np.allclose(base,expected,atol=1e-7,rtol=1e-5))
    if metadata['profile']=='almandine-isotropic': checks['isotropic_dark']=bool(np.max(base)==0 and np.max(fields['gamma_nm'])==0)
    if metadata['profile']=='quartz-undulose': checks['continuous_phase_variation']=bool(np.std(fields['phi_rad'][mask])>.01)
    if metadata['profile'] in ('plagioclase-twins','plagioclase-law-demo') and 'base_twin' not in fields:
        checks['multiple_optical_domains']=bool(len(np.unique(fields['domains'][mask]))>2 and len(np.unique(fields['phi_rad'][mask]))==2)
    if metadata.get('optical_model_file'):
        info=json.loads((asset/metadata['optical_model_file']).read_text(encoding='utf-8'))
        recipe=json.loads((asset/'request.json').read_text(encoding='utf-8'))['recipe']
        if recipe.get('orientation_model') == quartz_optics.MODEL:
            phi,gamma,domains,extra,recomputed=quartz_optics.fields(shape,alpha,recipe)
            checks['quartz_fields_reproduce_from_recipe']=bool(np.array_equal(phi,fields['phi_rad']) and
                np.array_equal(gamma,fields['gamma_nm']) and np.array_equal(domains,fields['domains']) and
                all(k in fields and np.array_equal(v,fields[k]) for k,v in extra.items()))
        elif recipe['phase']=='albite-reference':
            checks.update(twin_optics.check_model(info))
            with Image.open(asset/metadata['source_image']) as original:
                rgba=np.asarray(original.resize((shape[1],shape[0]),Image.Resampling.LANCZOS),dtype=np.float32)/255
            grey=srgb_to_linear(rgba[:,:,:3])@np.array([.2126,.7152,.0722])
            phi,gamma,domains,trans,b,extra,recomputed=image_domains.decompose(grey,alpha,recipe,recipe['seed'],physical_rgb)
            checks['image_estimated_fractions_reproduce']=bool(np.array_equal(extra['twin_fraction'],fields['twin_fraction']))
            if recipe.get('twin_stability_version') == 1:
                checks['twin_decomposition_stable']=bool(recomputed['decomposition_stability']['stable'])
                checks['twin_uncertainty_reproduces']=bool('twin_fraction_uncertainty' in fields and
                    np.array_equal(extra['twin_fraction_uncertainty'],fields['twin_fraction_uncertainty']))
        else:
            checks.update(twin_optics.check_model(info))
            phi,gamma,domains,recomputed=twin_optics.fields(shape,alpha,recipe,recipe['seed'])
        if recipe['phase'] in ('albite', 'albite-reference'):
            checks['albite_fields_reproduce_from_recipe']=bool(np.array_equal(phi,fields['phi_rad']) and
                   np.array_equal(gamma,fields['gamma_nm']) and np.array_equal(domains,fields['domains']))
        checks['optical_model_reproduces_from_recipe']=info==recomputed
    try:
        independent=verify_angles(asset)
        checks['independent_angle_reference']=independent['ok']
    except (ValueError,KeyError,TypeError) as exc:
        independent={'ok':False,'error':str(exc)}
        checks['independent_angle_reference']=False
    checks['source_preserved']=sha(asset/metadata['source_image'])==metadata['source_image_sha256']
    checks['file_hashes']=all((asset/name).is_file() and sha(asset/name)==expected for name,expected in metadata['files'].items())
    for name,value in checks.items():
        if not value: errors.append(name)
    return {'ok':not errors,'errors':errors,'checks':{k:bool(v) for k,v in checks.items()},
            'independent_angles':independent,'note':'Structural/physical QA does not certify visual realism or downstream model quality.'}


def review(asset,status,note):
    if not note.strip(): raise ValueError('A review note is required.')
    report=validate(asset)
    if status=='accepted' and not report['ok']: raise ValueError('Cannot accept an asset with failed automated checks.')
    metadata=json.loads((asset/'asset.json').read_text(encoding='utf-8'))
    metadata['review_status']=status; metadata.setdefault('review_history',[]).append({'at':now(),'status':status,'note':note})
    save_json(asset/'asset.json',metadata,True)
    temporary=asset/'preview-review-update.html'
    preview(asset,temporary); temporary.replace(asset/'preview.html')
    result={'review_status':status,'asset':str(asset)}
    if metadata.get('mode')=='appearance-only':
        result.update(mode='appearance-only',semantic_masks_provided=False,
                      semantic_mask_notice=phenomenon_policy.NO_MASK_NOTICE)
    return result


def main():
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare'); choice=q.add_mutually_exclusive_group()
    choice.add_argument('--profile',choices=PROFILES); choice.add_argument('--recipe',type=Path)
    q.add_argument('--mode',choices=phenomenon_policy.MODES,default='physical')
    q.add_argument('--mineral',help='Target mineral/assemblage for appearance-only synthesis; no class labels are assigned.')
    q.add_argument('--description',help='Complete requested XPL appearance, including all difficult features.')
    q.add_argument('--out',type=Path,required=True)
    q.add_argument('--rock'); q.add_argument('--size',type=int,default=1024); q.add_argument('--seed',type=int,default=0)
    q.add_argument('--phenomenon',action='append',default=[],help='Declare every requested special feature; repeat for combinations.')
    q.add_argument('--reference',action='append',default=[],help='Select a curated real-reference ID; selection does not claim actual image-tool use.')
    b=sub.add_parser('build'); b.add_argument('--request',type=Path,required=True); b.add_argument('--texture',type=Path,required=True); b.add_argument('--out',type=Path,required=True)
    r=sub.add_parser('render'); r.add_argument('--asset',type=Path,required=True); r.add_argument('--angles',type=float,nargs='+',required=True); r.add_argument('--out',type=Path,required=True)
    u=sub.add_parser('rebuild'); u.add_argument('--asset',type=Path,required=True); u.add_argument('--size',type=int,required=True); u.add_argument('--out',type=Path,required=True)
    v=sub.add_parser('validate'); v.add_argument('--asset',type=Path,required=True)
    a=sub.add_parser('verify-angles'); a.add_argument('--asset',type=Path,required=True); a.add_argument('--out',type=Path)
    h=sub.add_parser('preview'); h.add_argument('--asset',type=Path,required=True); h.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('review'); a.add_argument('--asset',type=Path,required=True); a.add_argument('--status',choices=['accepted','rejected','pending'],required=True); a.add_argument('--note',required=True)
    args=p.parse_args()
    if args.command=='prepare':
        if args.size<32 or args.size>4096: raise ValueError('Size must be 32–4096.')
        result=prepare(args.profile,args.out,args.rock,args.size,args.seed,args.recipe,args.phenomenon,args.reference,
                       mode=args.mode,mineral=args.mineral,description=args.description)
    elif args.command=='build': result=build(args.request,args.texture,args.out)
    elif args.command=='render': result=render(args.asset,args.angles,args.out)
    elif args.command=='rebuild': result=rebuild(args.asset,args.size,args.out)
    elif args.command=='verify-angles': result=verify_angles(args.asset,args.out)
    elif args.command=='preview': result=preview(args.asset,args.out)
    elif args.command=='review': result=review(args.asset,args.status,args.note)
    else:
        result=validate(args.asset)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if not result['ok']: raise SystemExit(1)
        return
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.command=='verify-angles' and not result['ok']: raise SystemExit(1)


if __name__=='__main__': main()
