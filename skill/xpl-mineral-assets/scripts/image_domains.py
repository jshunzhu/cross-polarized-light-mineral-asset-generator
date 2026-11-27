"""Estimate twin area fractions from a generated XPL appearance candidate.

This is an explicitly non-unique photometric decomposition, not a measurement
or a ground-truth segmentation. No procedural lamellae are drawn here.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

import twin_optics


def _bilinear(values, x, y):
    x = np.clip(x, 0, values.shape[1]-1)
    y = np.clip(y, 0, values.shape[0]-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, values.shape[1]-1), np.minimum(y0+1, values.shape[0]-1)
    wx, wy = x-x0, y-y0
    return ((1-wx)*(1-wy)*values[y0,x0] + wx*(1-wy)*values[y0,x1]
            + (1-wx)*wy*values[y1,x0] + wx*wy*values[y1,x1])


def fractions(grey, alpha, trace_angle_deg, *, filter_scale=1., population_quantiles=(.2,.8), subpixel=False):
    visible = alpha > .5
    # Filter across very fine image noise, then along the observed lamella trace.
    signal = np.asarray(Image.fromarray(np.uint8(np.clip(grey,0,1)*255)).filter(ImageFilter.MedianFilter(3)),dtype=np.float64)/255
    yy, xx = np.indices(grey.shape)
    angle = np.deg2rad(trace_angle_deg)
    total = np.zeros_like(grey,dtype=np.float64)
    weight = np.zeros_like(total)
    nominal_length = max(2, int(round(max(grey.shape)*.006)))
    length = max(1, int(round(nominal_length*filter_scale)))
    for distance in range(-length,length+1):
        kernel=np.exp(-.5*(distance/max(length*.65,1))**2)
        if subpixel:
            sx,sy=xx+distance*np.cos(angle),yy+distance*np.sin(angle)
            total += kernel*_bilinear(signal*alpha,sx,sy)
            weight += kernel*_bilinear(alpha,sx,sy)
        else:
            # Keep exact nearest-pixel historical sampling for unmarked recipes.
            sy=np.clip(yy+round(distance*np.sin(angle)),0,grey.shape[0]-1)
            sx=np.clip(xx+round(distance*np.cos(angle)),0,grey.shape[1]-1)
            local_weight=kernel*alpha[sy,sx]
            total += signal[sy,sx]*local_weight
            weight += local_weight
    signal=total/np.maximum(weight,1e-10)
    samples=signal[visible]
    low,high=np.quantile(samples,population_quantiles)
    for _ in range(30):
        split=(low+high)/2
        left=samples[samples<split];right=samples[samples>=split]
        if not len(left) or not len(right): raise ValueError('No two distinguishable twin intensity populations.')
        newlow,newhigh=float(left.mean()),float(right.mean())
        if abs(newlow-low)+abs(newhigh-high)<1e-7: break
        low,high=newlow,newhigh
    if high-low<.025: raise ValueError('Insufficient reference contrast for an estimated twin partition.')
    # Soft pixel fractions preserve natural transition texture and avoid adding
    # a new hard edge from a threshold mask. Categorical masks remain estimates.
    fraction=np.clip((signal-low)/(high-low),0,1)
    fraction[~(alpha>0)]=0
    return fraction.astype(np.float32),dict(dark_linear_mean=low,bright_linear_mean=high,
                                            along_trace_filter_pixels=length,
                                            ambiguous_fraction=float(np.mean((fraction[visible]>.15)&(fraction[visible]<.85))))


def stability(grey, alpha, trace_angle_deg, seed):
    """Deterministic sensitivity diagnostic, not a calibrated posterior.

    Limit the diagnostic grid to 256 pixels per side. Every perturbation uses
    that same grid. Continuous uncertainty is interpolated back for inspection;
    this does not create new domain labels or change the nominal decomposition.
    """
    shape = grey.shape
    scale = min(1., 256/max(shape))
    size = (max(1, round(shape[1]*scale)), max(1, round(shape[0]*scale)))
    def resize(values, target, method):
        return np.asarray(Image.fromarray(np.asarray(values, dtype=np.float32)).resize(target, method), dtype=np.float64)
    if scale < 1:
        g = resize(grey, size, Image.Resampling.BOX)
        a = resize(alpha, size, Image.Resampling.BOX)
    else:
        g, a = np.asarray(grey, dtype=np.float64), np.asarray(alpha, dtype=np.float64)
    visible = a > .5
    nominal, nominal_stats = fractions(g, a, trace_angle_deg, subpixel=True)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0., .003, g.shape)
    cases = [('noise-plus', g+noise, 0., 1., (.2,.8)),
             ('noise-minus', g-noise, 0., 1., (.2,.8)),
             ('trace-minus-2deg', g, -2., 1., (.2,.8)),
             ('trace-plus-2deg', g, 2., 1., (.2,.8)),
             ('filter-short', g, 0., .5, (.2,.8)),
             ('filter-long', g, 0., 1.5, (.2,.8)),
             ('population-wide', g, 0., 1., (.15,.85)),
             ('population-narrow', g, 0., 1., (.25,.75))]
    mean = np.zeros_like(g)
    mean_square = np.zeros_like(g)
    records, flips, changes = [], [], []
    for name, image, angle, length, quantiles in cases:
        try:
            fraction, stats = fractions(np.clip(image, 0, 1), a, trace_angle_deg+angle,
                                        filter_scale=length, population_quantiles=quantiles, subpixel=True)
            change = float(np.mean(np.abs(fraction[visible]-nominal[visible])))
            flip = float(np.mean((fraction[visible] >= .5) != (nominal[visible] >= .5)))
            mean += fraction
            mean_square += np.square(fraction.astype(np.float64))
            flips.append(flip); changes.append(change)
            records.append(dict(name=name, ok=True, mean_abs_fraction_change=change, dominant_domain_flip_fraction=flip,
                                trace_offset_deg=angle, filter_scale=length,
                                along_trace_filter_pixels=stats['along_trace_filter_pixels'], population_quantiles=list(quantiles)))
        except ValueError as exc:
            records.append(dict(name=name, ok=False, error=str(exc)))
    count = len(changes)
    uncertainty = np.sqrt(np.maximum(mean_square/max(count, 1)-(mean/max(count, 1))**2, 0.))
    p95 = float(np.quantile(uncertainty[visible], .95))
    thresholds = dict(max_fraction_change=.12, max_domain_flip_fraction=.10,
                      max_fraction_std_p95=.18, max_ambiguous_fraction=.60)
    checks = dict(all_perturbations_succeeded=count == len(cases),
                  fraction_change_bounded=bool(count and max(changes) <= thresholds['max_fraction_change']),
                  dominant_domains_stable=bool(count and max(flips) <= thresholds['max_domain_flip_fraction']),
                  fraction_uncertainty_bounded=p95 <= thresholds['max_fraction_std_p95'],
                  ambiguity_bounded=nominal_stats['ambiguous_fraction'] <= thresholds['max_ambiguous_fraction'])
    full_uncertainty = resize(uncertainty, (shape[1], shape[0]), Image.Resampling.BILINEAR) if scale < 1 else uncertainty
    full_uncertainty[alpha <= 0] = 0.
    report = dict(version=1, stable=all(checks.values()), seed=int(seed), checks=checks, thresholds=thresholds,
                  directional_sampling='alpha-weighted bilinear; angle perturbations are not rounded to integer offsets',
                  diagnostic_shape=list(g.shape), original_shape=list(shape), perturbations=records,
                  noise_sigma_linear=.003, fraction_std_p95=p95,
                  max_mean_abs_fraction_change=max(changes) if count else None,
                  max_dominant_domain_flip_fraction=max(flips) if count else None,
                  ambiguous_fraction=nominal_stats['ambiguous_fraction'],
                  interpretation='Engineering sensitivity thresholds, not calibrated probabilities. Stability does not establish true twin domains or a unique reference angle.')
    return full_uncertainty.astype(np.float32), report


def decompose(grey,alpha,recipe,seed,color_function):
    info=twin_optics.model(recipe)
    subpixel=recipe.get('twin_stability_version') == 1
    fraction,stats=fractions(grey,alpha,recipe['trace_angle_deg'],subpixel=subpixel)
    values=info['variants']
    colors=color_function(np.array([v['gamma_nm'] for v in values]))
    luminance=colors@np.array([.2126,.7152,.0722])
    ratio=stats['dark_linear_mean']/stats['bright_linear_mean']
    # Select a single global reference angle by contrast, keeping both optical
    # orientations constrained by the twin law. No per-domain angle fitting.
    candidates=np.linspace(0,90,9001)
    factors=np.sin(2*(np.deg2rad(candidates[:,None])-np.array([v['phi_rad'] for v in values])))**2
    response=factors*luminance
    scores=np.abs(np.log(np.maximum(response[:,0],1e-9)/np.maximum(response[:,1],1e-9))-np.log(ratio))
    scores[(response.min(axis=1)<.015) | (response[:,0]>=response[:,1])]=np.inf
    index=int(np.argmin(scores))
    if not np.isfinite(scores[index]): raise ValueError('Reference contrast cannot be represented by this twin pair.')
    anchor=float(candidates[index])
    predicted=(1-fraction)*response[index,0]+fraction*response[index,1]
    mask=alpha>0;visible=alpha>.5
    exposure=max(1.,float(np.quantile((grey/np.maximum(predicted,1e-9))[visible],.997)))
    trans=np.zeros_like(grey,dtype=np.float32)
    trans[mask]=np.clip(grey[mask]/np.maximum(exposure*predicted[mask],1e-9),0,1)
    phi=np.where(mask,values[0]['phi_rad'],0).astype(np.float32)
    gamma=np.where(mask,values[0]['gamma_nm'],0).astype(np.float32)
    phi_twin=np.where(mask,values[1]['phi_rad'],0).astype(np.float32)
    gamma_twin=np.where(mask,values[1]['gamma_nm'],0).astype(np.float32)
    domains=np.where(mask,np.where(fraction>=.5,2,1),0).astype(np.uint16)
    base=(colors[0]*trans[:,:,None]*(1-fraction[:,:,None])*exposure).astype(np.float32)
    base_twin=(colors[1]*trans[:,:,None]*fraction[:,:,None]*exposure).astype(np.float32)
    reconstructed=exposure*trans*predicted
    info.update(geometry=dict(method='Estimated from generated XPL appearance, no procedural stripe geometry',
                               trace_angle_deg=recipe['trace_angle_deg'],
                               labels='1=estimated host-dominant pixel, 2=estimated twin-dominant pixel',
                               label_status='inferred, not pixel-level ground truth; pending human inspection'),
                reference_decomposition=dict(stats,reference_angle_deg=anchor,fixed_exposure=exposure,
                     linear_luminance_mae=float(np.mean(np.abs(reconstructed[visible]-grey[visible]))),
                     clipped_fraction=float(np.mean(grey[visible]>exposure*predicted[visible])),
                     method='Directional denoising and two intensity populations; soft incoherent area mixing of the two twin-related retarders.',
                     nonuniqueness='One reference image cannot uniquely recover transmission, twin domains or optical parameters. These are synthesis estimates.'))
    info['assumptions'] += ['Image-derived twin fractions and neutral transmission are non-unique estimates from a synthetic reference view.',
                            'Boundary fractions use incoherent area mixing, not coherent multilayer ray tracing.',
                            'An image-guided boundary trace is an approximate morphology constraint, not a crystallographic reconstruction.']
    extras=dict(twin_fraction=fraction,phi_twin_rad=phi_twin,gamma_twin_nm=gamma_twin,
                base_twin=base_twin)
    if recipe.get('twin_stability_version') == 1:
        uncertainty, diagnostic = stability(grey, alpha, recipe['trace_angle_deg'], seed)
        info['decomposition_stability'] = diagnostic
        info['reference_decomposition']['directional_sampling'] = 'alpha-weighted bilinear along the twin trace'
        extras['twin_fraction_uncertainty'] = uncertainty
    return phi,gamma,domains,trans,base,extras,info
