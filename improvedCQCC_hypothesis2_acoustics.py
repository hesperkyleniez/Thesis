#!/usr/bin/env python3
"""Hypothesis 2: speaker-matched CQCC/F0 acoustic differences (development data ONLY).

Reads cached 40-D Akıncı–Özbay-inspired CQCC sidecars and 8-D F0 vectors.
Does NOT regenerate features, change checkpoints or access held-out test recordings.

Method:
- First average window statistics within each recording (each recording equal weight).
- Then average recordings for each speaker and class (each speaker equal weight).
- Form AI-minus-authentic paired speaker differences (32 expected development speakers).
- Shapiro-Wilk of paired differences; paired t-test if approximately normal,
  otherwise Wilcoxon signed-rank; two-sided; Bonferroni correction per family
  (20 CQCC means, 20 CQCC standard deviations, 2 F0 measures) and globally (42).
- Paired Cohen dz, a speaker-bootstrap 95% interval for mean paired difference,
  and complete traceable CSV, JSON and Markdown tables.

Integrated full Chapter 4 analysis (all conditions, ONE cache scan):
    python improvedCQCC_hypothesis2_acoustics.py --stage full
Creates a single README.md with clean, degraded pooled, each degradation,
and combined condition sections, with five PNG/SVG figures and statistical tables.

Optional legacy report stage (no cache scan):
    python improvedCQCC_hypothesis2_acoustics.py --stage report --condition clean
    python improvedCQCC_hypothesis2_acoustics.py --stage report --condition all
Creates separate legacy Chapter 4 reports and 7-8 figures for clean/all.
Requires matplotlib; no pandas needed. Existing CSV/JSON summaries are preserved.

Not a causal attribution test: generated audio may differ from original recordings
in sources, text, synthesis systems or recording chain beyond class identity.
Uses speaker matching, NOT assumed utterance-level matching.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

FAMILIES = [('CQCC_mean', 20), ('CQCC_std', 20), ('F0', 2)]
FEATURE_NAMES = ([f'CQCC_mean_C{i:02d}' for i in range(1, 21)] +
                 [f'CQCC_std_C{i:02d}' for i in range(1, 21)] +
                 ['F0_mean_logHz', 'F0_std_logHz'])
FEATURE_FAMILIES = ['CQCC_mean'] * 20 + ['CQCC_std'] * 20 + ['F0'] * 2
F0_VOICED_FRACTION_IDX = 6


def find_files(root: Path):
    train = root / 'features' / 'train'
    sidecar = root / 'features_experiments' / 'akinci_ozbay_cqcc' / 'train'
    if not train.is_dir() or not sidecar.is_dir():
        raise FileNotFoundError(f'Expected both cache directories:\n  {train}\n  {sidecar}')
    paths = sorted(train.rglob('*.npz'))
    if not paths:
        raise RuntimeError(f'No .npz caches in {train}')
    return train, sidecar, paths


def read_metadata(path, sidecar_path, *, check_sidecar=True):
    with np.load(path, allow_pickle=False) as d:
        for key in ('label','speaker','condition','f0'):
            if key not in d.files:
                raise ValueError(f'{path}: missing {key}')
        label = int(d['label'])
        speaker = str(d['speaker'])
        condition = str(d['condition'])
        f0_shape = d['f0'].shape
        if label not in (0,1):
            raise ValueError(f'{path}: invalid label {label}')
        if len(f0_shape)!=2 or f0_shape[1] != 8 or f0_shape[0]<1:
            raise ValueError(f'{path}: expected (windows, 8) F0, got {f0_shape}')
        if d['f0'].shape[0] != int(d['n_windows']):
            raise ValueError(f'{path}: F0/n_windows mismatch')
    if check_sidecar:
        if not sidecar_path.is_file():
            raise FileNotFoundError(f'Missing improved CQCC sidecar: {sidecar_path}')
        with np.load(sidecar_path, allow_pickle=False) as q:
            if 'cqcc' not in q.files or q['cqcc'].shape!=(f0_shape[0],40):
                raise ValueError(f'{sidecar_path}: expected {(f0_shape[0],40)} CQCC')
            if (int(q['label'])!=label or str(q['speaker'])!=speaker or
                str(q['condition'])!=condition):
                raise ValueError(f'{sidecar_path}: metadata disagrees with base cache')
    return label, speaker, condition


def feature_record(path, sidecar_path):
    # Load ONLY arrays needed for H2.  Neither file is modified.
    with np.load(path,allow_pickle=False) as d:
        label = int(d['label']); speaker=str(d['speaker']); condition=str(d['condition'])
        f0 = np.asarray(d['f0'],dtype=np.float64)
        if f0.ndim!=2 or f0.shape[1]!=8 or f0.shape[0]<1:
            raise ValueError(f'{path}: invalid F0 shape {f0.shape}')
    with np.load(sidecar_path,allow_pickle=False) as d:
        if 'cqcc' not in d.files:
            raise ValueError(f'{sidecar_path}: missing cqcc')
        cqcc = np.asarray(d['cqcc'],dtype=np.float64)
        if (cqcc.shape!=(len(f0),40) or int(d['label'])!=label or
            str(d['speaker'])!=speaker or str(d['condition'])!=condition):
            raise ValueError(f'{sidecar_path}: shape or metadata mismatch')
    if not np.isfinite(cqcc).all() or not np.isfinite(f0).all():
        raise ValueError(f'Non-finite features at {path}')

    # Each clip gets equal weight regardless of its 1 s / 0.5 s window count.
    cqcc_summary = cqcc.mean(axis=0)
    voiced = f0[:,F0_VOICED_FRACTION_IDX] > 0.0
    f0_summary = f0[voiced, :2].mean(axis=0) if np.any(voiced) else np.array([np.nan,np.nan])
    vec=np.concatenate([cqcc_summary,f0_summary])
    return label,speaker,condition,vec,int(len(f0)),int(np.sum(voiced))


def shapiro_and_test(difference: np.ndarray):
    n=len(difference)
    if n<3:
        return 'insufficient', math.nan, math.nan, math.nan
    if np.allclose(difference,0.0,rtol=0,atol=1e-12):
        return 'constant_zero', 1.0, 0.0, 1.0
    if float(np.std(difference,ddof=1)) < 1e-12:
        # All speaker deltas are (nearly) the same nonzero value; t-stat undefined.
        return 'constant_nonzero', math.nan, math.nan, math.nan
    try:
        p_normal=float(stats.shapiro(difference).pvalue)
    except Exception:
        p_normal=math.nan
    if math.isfinite(p_normal) and p_normal>=0.05:
        result=stats.ttest_1samp(difference,popmean=0,alternative='two-sided')
        return 'paired_t',p_normal,float(result.statistic),float(result.pvalue)
    try:
        result=stats.wilcoxon(difference,alternative='two-sided',zero_method='wilcox',
                              method='auto')
        return 'wilcoxon',p_normal,float(result.statistic),float(result.pvalue)
    except ValueError:
        return 'wilcoxon_undefined',p_normal,math.nan,math.nan


def analyze(groups, *, n_bootstrap=4000, seed=42, min_speakers=16):
    speaker_set=sorted({speaker for speaker,label in groups})
    complete_speakers=[s for s in speaker_set if (s,0) in groups and (s,1) in groups]
    if len(complete_speakers)<min_speakers:
        raise ValueError(f'Only {len(complete_speakers)} speakers have both classes; '
                         f'at least {min_speakers} required. Check the cache and condition filter.')
    if len(complete_speakers)!=len(speaker_set):
        print(f'WARNING: dropping {len(speaker_set)-len(complete_speakers)} speakers missing one class')
    # Each (speaker,label) group has 42 means. F0 means may be NaN for clips without pitch.
    class_means={}
    class_counts={}
    for key,values in groups.items():
        a=np.asarray(values,dtype=np.float64)
        with np.errstate(invalid='ignore'):
            class_means[key]=np.nanmean(a,axis=0)
        class_counts[key]=len(a)
    rows=[]
    rng=np.random.default_rng(seed)
    for col,(name,family) in enumerate(zip(FEATURE_NAMES,FEATURE_FAMILIES)):
        real=np.array([class_means[(s,0)][col] for s in complete_speakers])
        ai=np.array([class_means[(s,1)][col] for s in complete_speakers])
        valid=np.isfinite(real)&np.isfinite(ai)
        differences=ai[valid]-real[valid]
        n=int(np.sum(valid))
        if n < min_speakers:
            method='insufficient'; p_normal=stat=raw_p=math.nan
            dz=math.nan; lower=upper=math.nan
        else:
            method,p_normal,stat,raw_p=shapiro_and_test(differences)
            sd=float(np.std(differences,ddof=1))
            dz=float(np.mean(differences)/sd) if sd>1e-12 else math.nan
            idx=rng.integers(0,n,size=(n_bootstrap,n))
            bs=np.mean(differences[idx],axis=1)
            lower,upper=[float(x) for x in np.percentile(bs,[2.5,97.5])]
        rows.append({'feature':name, 'family':family,'n_speaker_pairs':n,
                     'real_mean':float(np.mean(real[valid])) if n else math.nan,
                     'ai_mean':float(np.mean(ai[valid])) if n else math.nan,
                     'delta_ai_minus_real':float(np.mean(differences)) if n else math.nan,
                     'bootstrap_ci95_low':lower,'bootstrap_ci95_high':upper,
                     'normality_shapiro_p':p_normal,'test':method,'test_statistic':stat,
                     'p_raw':raw_p,'effect_dz':dz})
    for row in rows:
        family_k=dict(FAMILIES)[row['family']]
        p=row['p_raw']
        row['p_bonferroni_family']=float(min(1,p*family_k)) if math.isfinite(p) else math.nan
        row['p_bonferroni_global42']=float(min(1,p*len(rows))) if math.isfinite(p) else math.nan
        row['significant_family']=bool(row['p_bonferroni_family']<0.05)
        row['significant_global42']=bool(row['p_bonferroni_global42']<0.05)
    return rows,complete_speakers,class_means,class_counts


def fmt(v, places=4):
    try:
        if v is None or not math.isfinite(float(v)): return 'NA'
        return f'{float(v):.{places}f}'
    except Exception:return str(v)


def write_csv(path, data, columns):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader(); w.writerows(data)


# ------------------ Chapter 4 visualization/reporting (read-only) ------------------

def _as_bool(value):
    return value is True or str(value).strip().lower() in ('true','1','yes')


def load_prior_analysis(output: Path):
    required = ['h2_feature_tests.csv', 'h2_speaker_aggregates.csv', 'h2_summary.json']
    missing = [str(output / name) for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing earlier H2 results: ' + ', '.join(missing) +
                                '. First run --stage run for this condition.')
    with (output/'h2_feature_tests.csv').open(encoding='utf-8-sig',newline='') as f:
        raw=list(csv.DictReader(f))
    with (output/'h2_speaker_aggregates.csv').open(encoding='utf-8-sig',newline='') as f:
        speaker_raw=list(csv.DictReader(f))
    with (output/'h2_summary.json').open(encoding='utf-8') as f:
        summary=json.load(f)
    if len(raw)!=42 or len({r['feature'] for r in raw})!=42:
        raise RuntimeError(f'Expected 42 distinct H2 features in {output}; found {len(raw)} rows')
    numeric_cols=('n_speaker_pairs','real_mean','ai_mean','delta_ai_minus_real',
                  'bootstrap_ci95_low','bootstrap_ci95_high','normality_shapiro_p',
                  'test_statistic','p_raw','effect_dz','p_bonferroni_family',
                  'p_bonferroni_global42')
    rows=[]
    for raw_row in raw:
        row=dict(raw_row)
        for name in numeric_cols:
            if name in row and row[name]!='':
                row[name]=float(row[name])
        for name in ('significant_family','significant_global42'):
            row[name]=_as_bool(row.get(name))
        rows.append(row)
    speakers=[]
    for raw_row in speaker_raw:
        row=dict(raw_row)
        row['label']=int(row['label'])
        row['n_clips']=int(row['n_clips'])
        for name in FEATURE_NAMES:
            row[name]=float(row[name]) if row.get(name) else float('nan')
        speakers.append(row)
    by_feature={r['feature']:r for r in rows}
    if not all(k in by_feature for k in ('F0_mean_logHz','F0_std_logHz')):
        raise RuntimeError('Could not locate both F0 feature rows')
    return rows,speakers,summary,by_feature


def _paired_speaker_points(speakers,feature):
    pairs=defaultdict(dict)
    for r in speakers:
        pairs[r['speaker']][r['label']]=float(r[feature])
    keys=sorted(s for s,d in pairs.items() if 0 in d and 1 in d and
                math.isfinite(d[0]) and math.isfinite(d[1]))
    return keys,np.array([pairs[s][0] for s in keys]),np.array([pairs[s][1] for s in keys])


def _pvalue(v):
    if v is None or not math.isfinite(float(v)): return 'NA'
    v=float(v)
    if v<0.0001:return f'{v:.2e}'
    return f'{v:.4f}'


def _significance_label(row):
    if row['significant_global42']:return 'Global (42)'
    if row['significant_family']:return 'Within family only'
    return 'Not significant'


def make_chapter4_figures(output, rows, speakers, summary, by_feature):
    """Create descriptive figures from saved speaker aggregates, no feature-file access."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError('Graphs need matplotlib. Install: python -m pip install matplotlib') from exc
    import matplotlib.ticker as ticker
    figs=output/'figures'
    figs.mkdir(parents=True,exist_ok=True)
    blue='#28658a'; orange='#c26b41'; gray='#838b95'; green='#16725b'
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white',
                         'figure.facecolor':'white'})
    manifest=[]

    def save(fig,stem,caption):
        fig.savefig(figs/(stem+'.png'),dpi=240,bbox_inches='tight')
        fig.savefig(figs/(stem+'.svg'),bbox_inches='tight')
        plt.close(fig)
        manifest.append({'filename':stem+'.png','vector_filename':stem+'.svg',
                         'caption':caption})

    # 1-2. Paired speaker-level Real-vs-AI scatter (F0 mean and variability).
    for index,(feature,kind) in enumerate([('F0_mean_logHz','mean'),
                                           ('F0_std_logHz','std')],start=1):
        _,real,ai=_paired_speaker_points(speakers,feature)
        if not len(real):raise RuntimeError(f'No valid paired speaker values for {feature}')
        row=by_feature[feature]
        lo=float(min(real.min(),ai.min())); hi=float(max(real.max(),ai.max()))
        span=max(hi-lo,1e-6); lo-=span*.08; hi+=span*.08
        fig,ax=plt.subplots(figsize=(6.8,6.1))
        ax.plot([lo,hi],[lo,hi],linestyle='--',color=gray,lw=1.5,
                label='Equal authentic and AI values')
        ax.scatter(real,ai,s=43,alpha=.82,color=blue,edgecolor='white',linewidth=.5)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Authentic: speaker-averaged log-F0 '+kind,
               ylabel='AI-generated: speaker-averaged log-F0 '+kind,
               title=('F0 overall pitch level' if kind=='mean' else 'F0 pitch variability')+' — paired speakers')
        ax.text(.02,.97,f"n={len(real)} speakers  |  AI − Real={row['delta_ai_minus_real']:+.4f}\n"
                f"Paired d$_z$={row['effect_dz']:+.3f}  |  F0-family p={_pvalue(row['p_bonferroni_family'])}\n"
                f"Global 42-test p={_pvalue(row['p_bonferroni_global42'])}",
                transform=ax.transAxes,va='top',fontsize=9,
                bbox={'facecolor':'white','alpha':.88,'edgecolor':'#e0e0e0','pad':7})
        ax.legend(loc='lower right',fontsize=8)
        save(fig,f'fig_{index:02d}_f0_{kind}_paired',
             ('Speaker-matched authentic-versus-AI mean log-F0; the dashed identity line '
              'indicates equal values.' if kind=='mean' else
              'Speaker-matched authentic-versus-AI within-window log-F0 standard deviation; '
              'points below the identity line indicate lower AI variability.'))

    # 3. Distributions of speaker-paired AI-minus-Real F0 differences, separate scales.
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    rg=np.random.default_rng(2026)
    for ax,feature,title_ in zip(axes,['F0_mean_logHz','F0_std_logHz'],
                                 ['Mean log-F0','Log-F0 variability']):
        _,real,ai=_paired_speaker_points(speakers,feature)
        delta=ai-real
        row=by_feature[feature]
        jitter=rg.uniform(-.1,.1,size=len(delta))
        ax.scatter(jitter,delta,color=orange if feature.endswith('std_logHz') else blue,
                   s=32,alpha=.72)
        ax.axhline(0,color=gray,linestyle='--',lw=1.4)
        ax.errorbar(0,float(np.mean(delta)),
                    yerr=[[max(0,float(np.mean(delta))-row['bootstrap_ci95_low'])],
                          [max(0,row['bootstrap_ci95_high']-float(np.mean(delta)))]],
                    fmt='D',capsize=5,color='black',markersize=8,
                    label='Mean and bootstrap 95% CI')
        ax.set(xlim=(-.37,.37),xticks=[],ylabel='AI − Authentic (log-F0 units)',title=title_)
        ax.text(.03,.97,f"Family p={_pvalue(row['p_bonferroni_family'])}\n"
                f"Global p={_pvalue(row['p_bonferroni_global42'])}",
                transform=ax.transAxes,va='top',fontsize=9)
    axes[1].legend(loc='lower left',fontsize=8)
    save(fig,'fig_03_f0_speaker_differences',
         'Per-speaker paired AI-minus-authentic F0 differences; diamonds mark mean '
         'paired difference with a non-multiplicity-adjusted bootstrap 95% confidence interval.')

    # 4. CQCC standardized paired effect sizes, mean vs within-window standard deviation.
    fig,axes=plt.subplots(2,1,figsize=(12,8.2),sharex=True)
    fig.subplots_adjust(bottom=.22,hspace=.32)
    for ax,family in zip(axes,['CQCC_mean','CQCC_std']):
        subset=sorted((r for r in rows if r['family']==family),
                      key=lambda r:int(r['feature'].split('C')[-1]))
        x=np.arange(1,len(subset)+1)
        colors=[green if r['significant_global42'] else
                orange if r['significant_family'] else gray for r in subset]
        ax.bar(x,[r['effect_dz'] for r in subset],color=colors,alpha=.85,width=.72)
        ax.axhline(0,color='#333333',lw=1)
        ax.set(ylabel='Paired Cohen $d_z$',title=('CQCC pooled means' if family=='CQCC_mean'
                                                 else 'CQCC pooled within-window SDs'))
        ax.grid(axis='y',alpha=.15)
    axes[-1].set(xticks=np.arange(1,21),xlabel='CQCC coefficient (C01–C20)')
    fig.legend(handles=[Line2D([0],[0],color=green,lw=8,label='Global Bonferroni p < 0.05'),
                        Line2D([0],[0],color=orange,lw=8,label='Family only p < 0.05'),
                        Line2D([0],[0],color=gray,lw=8,label='Not significant')],
               loc='lower center',ncol=3,bbox_to_anchor=(.5,.045),frameon=False)
    save(fig,'fig_04_cqcc_effect_sizes',
         'Paired Cohen dz for 20 CQCC mean and 20 CQCC standard-deviation features; '
         'bar color distinguishes global, family-only, and nonsignificant findings.')

    # 5. CQCC global-adjusted significance across the 20 coefficients.
    fig,ax=plt.subplots(figsize=(11,4.8))
    for family,label_,color in [('CQCC_mean','CQCC mean',blue),
                                ('CQCC_std','CQCC std',orange)]:
        subset=sorted((r for r in rows if r['family']==family),
                      key=lambda r:int(r['feature'].split('C')[-1]))
        x=np.arange(1,21)
        y=-np.log10(np.maximum([r['p_bonferroni_global42'] for r in subset],1e-300))
        ax.plot(x,y,'o-',lw=1.6,ms=5,color=color,label=label_)
    ax.axhline(-np.log10(.05),color=gray,ls='--',lw=1.4,label='Global adjusted p = 0.05')
    ax.set(xticks=np.arange(1,21),xlabel='CQCC coefficient number',
           ylabel=r'$-\log_{10}(p_{\rm Bonferroni,42})$',
           title='CQCC significance by coefficient — across-42 correction')
    ax.legend(fontsize=9);ax.grid(axis='y',alpha=.16)
    save(fig,'fig_05_cqcc_adjusted_p',
         'Across-42 Bonferroni-adjusted significance of CQCC mean and standard-deviation '
         'coefficients; points above the dashed line satisfy adjusted p < 0.05.')

    # 6. Counts: family-adjusted versus global-adjusted (not percentages of clips).
    counts=[]
    for name,label_ in [('CQCC_mean','CQCC mean'),('CQCC_std','CQCC std'),('F0','F0')]:
        subset=[r for r in rows if r['family']==name]
        counts.append((label_,len(subset),sum(r['significant_family'] for r in subset),
                       sum(r['significant_global42'] for r in subset)))
    fig,ax=plt.subplots(figsize=(8.5,4.7))
    x=np.arange(len(counts)); width=.35
    ax.bar(x-width/2,[r[2] for r in counts],width,color=blue,label='Within-family adjusted')
    ax.bar(x+width/2,[r[3] for r in counts],width,color=orange,label='Globally adjusted (42)')
    ax.set(xticks=x,xticklabels=[f'{r[0]}\n(n={r[1]} tests)' for r in counts],
           ylabel='Number of statistically significant features',
           title='Significant acoustic features by correction family')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend();ax.grid(axis='y',alpha=.16)
    for container in ax.containers:
        ax.bar_label(container,padding=3)
    save(fig,'fig_06_significant_feature_counts',
         'Count of features passing the within-family and across-42 Bonferroni '
         'corrections; counts are not proportions of independent audio clips.')

    # 7. Top-ranked standardised differences, including F0 std even if not globally significant.
    sorted_rows=sorted(rows,key=lambda r:(not r['significant_global42'],
                        r['p_bonferroni_global42']))
    wanted=sorted_rows[:11]
    f0std=by_feature['F0_std_logHz']
    if f0std['feature'] not in {r['feature'] for r in wanted}:
        wanted.append(f0std)
    wanted=sorted(wanted,key=lambda r:r['effect_dz'])
    fig,ax=plt.subplots(figsize=(10,5.6))
    y=np.arange(len(wanted))
    ax.barh(y,[r['effect_dz'] for r in wanted],
            color=[green if r['significant_global42'] else
                   orange if r['significant_family'] else gray for r in wanted],alpha=.9)
    ax.set(yticks=y,yticklabels=[r['feature'] for r in wanted],
           xlabel='Paired standardized difference ($d_z$): AI minus authentic',
           title='Largest ranked acoustic effects and F0 variability')
    ax.axvline(0,color='#333333',lw=1);ax.grid(axis='x',alpha=.15)
    save(fig,'fig_07_key_feature_effect_sizes',
         'Ranked paired Cohen dz for top globally significant features plus F0 standard '
         'deviation; comparison is descriptive and does not imply causal contribution to detection.')

    # 8 (when both analyses exist). Clean-versus-all sensitivity, same speakers,
    # partly overlapping observations; NOT an independent replication.
    other_name='all' if summary.get('primary_condition')=='clean' else 'clean'
    other_output=output.parent/other_name
    if other_output!=output and all((other_output/f).is_file() for f in
                                    ('h2_feature_tests.csv','h2_speaker_aggregates.csv','h2_summary.json')):
        _,_,other_summary,other_by_feature=load_prior_analysis(other_output)
        if {str(summary['primary_condition']),str(other_summary['primary_condition'])}=={'clean','all'}:
            values={str(summary['primary_condition']):by_feature,
                    str(other_summary['primary_condition']):other_by_feature}
            fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
            for ax,feature,title_ in zip(axes,['F0_mean_logHz','F0_std_logHz'],
                                         ['Mean log-F0','Log-F0 variability']):
                selected=[values[c][feature] for c in ('clean','all')]
                means=np.array([r['delta_ai_minus_real'] for r in selected])
                intervals=np.array([[(r['delta_ai_minus_real']-r['bootstrap_ci95_low']),
                                      (r['bootstrap_ci95_high']-r['delta_ai_minus_real'])]
                                      for r in selected]).T
                ax.errorbar([0,1],means,yerr=np.maximum(intervals,0),fmt='o',
                            color=blue if feature=='F0_mean_logHz' else orange,
                            markersize=8,capsize=7,lw=2)
                ax.axhline(0,color=gray,ls='--',lw=1.5)
                ax.set(xticks=[0,1],xticklabels=['Clean','All conditions'],
                       ylabel='Mean speaker-paired AI − Authentic',title=title_)
                ax.grid(axis='y',alpha=.17)
            save(fig,'fig_08_f0_clean_all_sensitivity',
                 'F0 mean and variability contrasts under clean versus all development '
                 'conditions; intervals are per-feature bootstrap 95% CIs, not globally '
                 'multiplicity-adjusted. The analyses reuse speakers and are not independent.')

    captions=['# Hypothesis 2 — Figure captions','']
    for i,item in enumerate(manifest,1):
        captions.extend([f"**Figure H2.{i}. {item['caption']}**",
                         f"File: `figures/{item['filename']}`",''])
    (output/'h2_figure_captions.md').write_text('\n'.join(captions),encoding='utf-8')
    with (output/'h2_figure_manifest.json').open('w',encoding='utf-8') as f:
        json.dump(manifest,f,indent=2)
    return manifest


def write_chapter4_package(output,rows,speakers,summary,by_feature,manifest):
    condition=str(summary['primary_condition'])
    n_clips=int(summary['n_selected_clips']); n_speakers=int(summary['n_speaker_pairs'])
    f0mean=by_feature['F0_mean_logHz']; f0std=by_feature['F0_std_logHz']
    n_global=sum(r['significant_global42'] for r in rows)
    n_family=sum(r['significant_family'] for r in rows)
    def feature_line(r):
        return ('| '+ ' | '.join([r['feature'],fmt(r['real_mean']),fmt(r['ai_mean']),
                  f"{r['delta_ai_minus_real']:+.4f}",r['test'],
                  _pvalue(r['p_raw']),_pvalue(r['p_bonferroni_family']),
                  _pvalue(r['p_bonferroni_global42']),f"{r['effect_dz']:+.3f}",
                  _significance_label(r)])+' |')
    table_head=['| Feature | Authentic | AI | AI−Real | Paired test | Raw p | Family-adjusted p | Global-adjusted p | $d_z$ | Finding |',
                '|---|---:|---:|---:|---|---:|---:|---:|---:|---|']
    f0table=table_head+[feature_line(f0mean),feature_line(f0std)]
    f0_pct=100*(f0std['ai_mean']/f0std['real_mean']-1) if f0std['real_mean'] else float('nan')
    familycounts={fam:(sum(r['significant_family'] for r in rows if r['family']==fam),
                       sum(r['significant_global42'] for r in rows if r['family']==fam))
                  for fam,_ in FAMILIES}
    cqcc_top=sorted((r for r in rows if r['family']!='F0'),
                    key=lambda r:(not r['significant_global42'],r['p_bonferroni_global42']))
    cqcc_tab=table_head+[feature_line(r) for r in cqcc_top[:12]]
    all_sig=[r for r in rows if r['significant_global42']]
    write_csv(output/'h2_f0_detailed.csv',[f0mean,f0std],list(rows[0].keys()))
    write_csv(output/'h2_cqcc_global_significant.csv',all_sig,list(rows[0].keys()))
    (output/'h2_chapter4_tables.md').write_text('\n'.join([
        '# Chapter 4 — Hypothesis 2 tables','',
        f'Table H2.1. Fundamental-frequency statistics ({condition}; n={n_speakers} paired speakers).','',
        *f0table,'','Note: log-F0 features use the mean of voiced-window measurements; '
        'they are not mean raw pitch in Hertz. The 95% bootstrap confidence intervals '
        'are available in `h2_feature_tests.csv`.','',
        'Table H2.2. Twelve strongest-ranked CQCC measurements.','',
        *cqcc_tab,'','Table H2.3. Significant-feature counts by acoustic family.','',
        '| Feature family | Features tested | Family-adjusted significant | Globally adjusted significant |',
        '|---|---:|---:|---:|---:|',
        *[f'| {fam} | {k} | {familycounts[fam][0]} | {familycounts[fam][1]} |' for fam,k in FAMILIES],
        '','All figures: see `h2_figure_captions.md` and `figures/`.','']),encoding='utf-8')
    clean_str=('the clean condition' if condition=='clean' else 'all development acoustic conditions')
    support_direction=(f0std['delta_ai_minus_real']<0)
    support_family=f0std['significant_family']
    mean_nonsig=not f0mean['significant_family']
    report=['# Chapter 4 — Hypothesis 2: CQCC and fundamental frequency','',
      '## Statistical analysis of acoustic differences','',
      f'The analysis examined **{n_clips:,} development recordings** under {clean_str}, '
      f'aggregated into **{n_speakers} speaker-matched authentic/AI observations**. '
      'The speaker was the paired statistical unit; recordings and overlapping '
      '1-second windows were not treated as independent replicates. Each recording '
      'contributed equally within speaker/class before the paired comparisons. '
      'The selected features comprised 20 CQCC pooled means, 20 CQCC pooled '
      'standard deviations, voiced-window mean log-F0, and voiced-window log-F0 '
      'standard deviation. Normality was assessed on paired differences using '
      'Shapiro–Wilk; a paired t-test or Wilcoxon signed-rank test was used as '
      'appropriate. Reported p-values are two-sided, with Bonferroni adjustments '
      'both within each feature family and across all 42 comparisons.','',
      '## F0 mean versus pitch variability','',*f0table,'',
      f'Mean log-F0 was {fmt(f0mean["real_mean"])} for authentic speech and '
      f'{fmt(f0mean["ai_mean"])} for AI-generated speech, with a paired difference '
      f'(AI minus authentic) of {f0mean["delta_ai_minus_real"]:+.4f}, '
      f'paired $d_z={f0mean["effect_dz"]:+.3f}$, and F0-family-adjusted '
      f'$p={_pvalue(f0mean["p_bonferroni_family"])}$. '
      + ('No statistically significant difference in mean log-F0 was detected '
         'at the family-adjusted 0.05 level.' if mean_nonsig else
         'A difference in mean log-F0 was detected at the family-adjusted 0.05 level.'),
      f'The mean within-window **log-F0 standard deviation** was '
      f'{fmt(f0std["real_mean"])} for authentic and {fmt(f0std["ai_mean"])} '
      f'for AI-generated recordings. The AI-minus-authentic difference was '
      f'{f0std["delta_ai_minus_real"]:+.4f} log-scale units '
      f'({f0_pct:+.1f}% relative to the mean authentic variability statistic), '
      f'with paired $d_z={f0std["effect_dz"]:+.3f}$. '
      f'The F0-family-adjusted p-value was '
      f'{_pvalue(f0std["p_bonferroni_family"])} and the across-42-adjusted p-value '
      f'was {_pvalue(f0std["p_bonferroni_global42"])}. '
      + ('AI recordings displayed lower measured pitch variability than authentic '
         'recordings. ' if support_direction else
         'AI recordings did not display lower measured pitch variability in this contrast. ')
      + ('The contrast was statistically significant within the two-feature F0 family, '
         'but ' if support_family and not f0std['significant_global42'] else
         'Within-family significance and ') +
      ('did not survive the stricter correction across all 42 features.'
       if not f0std['significant_global42'] else
       'the difference also survived the correction across all 42 features.'),
      '', '## CQCC coefficient mean and variability findings','',
      f'After correcting for all 42 comparisons, **{familycounts["CQCC_mean"][1]}/20** '
      f'CQCC mean features and **{familycounts["CQCC_std"][1]}/20** CQCC '
      'within-window standard-deviation features were significant. '
      'These results indicate differences in specific cepstral characteristics; '
      'they do not show that every CQCC coefficient is discriminative or prove '
      'that any one feature causes the classifier to perform better.','',
      *cqcc_tab,'',
      '## Comparison with the reviewed literature','',
      '**Fundamental frequency (F0).** The supplied review describes two separate '
      'expectations: authentic and synthetic voices can have comparable typical '
      'pitch levels, while synthetic speech may exhibit more concentrated or stable '
      'pitch contours (discussion attributed in the review to Xue et al., 2022). '
      + ('The absence of a significant mean log-F0 difference is consistent with '
         'the first expectation, but a nonsignificant test is not an equivalence test. '
         if mean_nonsig else
         'The observed difference in mean log-F0 does not follow the expectation of '
         'similar overall pitch levels. ') +
      ('The negative log-F0 standard-deviation difference is directionally consistent '
       'with the second expectation. ' if support_direction else
       'The log-F0 standard-deviation contrast is not directionally consistent '
       'with the second expectation. ') +
      'The cited study used richer F0-based spectral-subband representations, whereas '
      'our experiment analyzes compact pooled YIN-derived log-F0 descriptors; '
      'therefore this is **conceptual consistency, not a direct replication**.','',
      '**CQCC.** The proposed CQCC branch is inspired by the Akıncı–Özbay (2026) '
      'cepstral-feature study described in the thesis. Our observed coefficient-level '
      'differences provide dataset-specific acoustic evidence; they do not independently '
      'verify the cited study’s performance findings or establish a causal role for CQCC '
      'in the final late-fusion model.','',
      '## Hypothesis 2 decision and limitations','',
      f'**{n_global}/42** features survived global Bonferroni correction '
      f'({n_family}/42 survived corrections within their respective families). '
      '**The originally stated majority-of-features decision rule is not met.** '
      'Accordingly, these findings should be described as *feature-specific evidence '
      'of acoustic differences*, not rejection of the overall null under that majority '
      'rule. The direction and size of F0 variability remain of descriptive interest '
      'even where global multiplicity correction is not met.','',
      'These analyses are exploratory and observational. The F0 expert itself has '
      'eight input features, but this RRL-directed statistical test examines only '
      'the two voiced pitch-summary measures (mean and standard deviation). '
      'Speaker-matching controls '
      'speaker identity at the summary level but does not guarantee exact '
      'utterance/transcript matching; source recordings, synthetic voice generation, '
      'and condition processing may differ beyond real/AI status. Analyses are '
      'based on speaker means, not the entire raw pitch-contour distribution. '
      'Bootstrap confidence intervals in the CSV are per-feature and not '
      'simultaneously adjusted for 42 comparisons. The all-conditions analysis '
      'is a sensitivity check on partly augmented versions of the development data, '
      'not an independent replication.','',
      '## Recommended Chapter 4 figures','']
    other_name='all' if condition=='clean' else 'clean'
    other_output=output.parent/other_name
    if other_output!=output and (other_output/'h2_feature_tests.csv').is_file() and (other_output/'h2_summary.json').is_file():
        _,_,other_summary,other_by_feature=load_prior_analysis(other_output)
        if {condition,str(other_summary['primary_condition'])}=={'clean','all'}:
            paired_conditions={condition:by_feature,other_name:other_by_feature}
            c=paired_conditions['clean']['F0_std_logHz']
            a=paired_conditions['all']['F0_std_logHz']
            report.extend(['## Clean-versus-all-condition sensitivity','',
                f'The F0 variability difference (AI minus authentic) was '
                f'{c["delta_ai_minus_real"]:+.4f} for clean development recordings '
                f'and {a["delta_ai_minus_real"]:+.4f} across all conditions. '
                f'The corresponding within-F0-family-adjusted p-values were '
                f'{_pvalue(c["p_bonferroni_family"])} and '
                f'{_pvalue(a["p_bonferroni_family"])}; global 42-test adjusted p-values '
                f'were {_pvalue(c["p_bonferroni_global42"])} and '
                f'{_pvalue(a["p_bonferroni_global42"])}. '
                'These analyses share speakers and partially overlapping recordings, '
                'so they are not independent replications.',''])
    for i,item in enumerate(manifest,1):
        report.extend([f"**Figure H2.{i}.** {item['caption']}",
                       f"`figures/{item['filename']}`",''])
    report.extend(['## Source-to-claim caution','',
        'The F0 expectation above is taken from the supplied thesis RRL, not '
        'independently verified against Xue et al. (2022). Cite the underlying '
        'papers as appropriate after checking their exact claims. Do not assert '
        'that F0 is globally significant if only the F0-family correction passes.',''])
    (output/'h2_chapter4_discussion.md').write_text('\n'.join(report),encoding='utf-8')
    print(f'  Chapter 4: {output/"h2_chapter4_discussion.md"}')
    print(f'  Tables: {output/"h2_chapter4_tables.md"}')
    print(f'  Figures: {output/"figures"} ({len(manifest)} PNG + SVG pairs)')


def write_chapter4_from_saved(output):
    rows,speakers,summary,by_feature=load_prior_analysis(output)
    manifest=make_chapter4_figures(output,rows,speakers,summary,by_feature)
    write_chapter4_package(output,rows,speakers,summary,by_feature,manifest)
    print('\nH2 graphical/Chapter 4 report completed without rereading audio features.')


# ----------------- Integrated Chapter 4: all development conditions -----------------

H2_DEGRADATIONS = ('mobile_gaussian', 'mobile_urban', 'room_gaussian',
                   'room_urban', 'telephone')
H2_FULL_ORDER = ('clean', 'degraded', *H2_DEGRADATIONS, 'combined')
H2_DISPLAY = {
    'clean': 'Clean speech', 'degraded': 'Degraded speech (pooled)',
    'mobile_gaussian': 'Mobile + Gaussian noise',
    'mobile_urban': 'Mobile + urban noise',
    'room_gaussian': 'Room + Gaussian noise',
    'room_urban': 'Room + urban noise', 'telephone': 'Telephone',
    'combined': 'All conditions combined',
}


def run_full_chapter4(a):
    """Read all development caches ONCE; analyze eight overlapping strata.

    No held-out test recordings, neural checkpoints, or cached features are modified.
    Clean/degraded/combined analyses are *overlapping* summaries, not replicates.
    """
    base, side, paths = find_files(a.root)
    sidecar_count = sum(1 for _ in side.rglob('*.npz'))
    if a.expected_clips and (len(paths) != a.expected_clips or sidecar_count != a.expected_clips):
        raise RuntimeError(f'Expected {a.expected_clips} base and paper sidecars; '
                           f'found {len(paths)} and {sidecar_count}.')
    output = a.output or a.root / 'results' / 'hypothesis2_cqcc_f0' / 'full_chapter4'
    output.mkdir(parents=True, exist_ok=True)
    groups = {name: defaultdict(list) for name in H2_FULL_ORDER}
    counts = Counter()
    unvoiced = Counter()
    print('H2 INTEGRATED ANALYSIS | DEVELOPMENT DATA ONLY')
    print(f'  Reading {len(paths):,} prepared recordings once; no model retraining.')
    for i, path in enumerate(paths, 1):
        paired_sidecar = side / path.relative_to(base)
        if not paired_sidecar.is_file():
            raise FileNotFoundError(f'Missing improved CQCC sidecar: {paired_sidecar}')
        label, speaker, condition, vec, n_windows, n_voiced = feature_record(path, paired_sidecar)
        if condition not in ('clean', *H2_DEGRADATIONS):
            raise RuntimeError(f'Unexpected condition {condition!r} at {path}')
        strata = ('combined', 'clean') if condition == 'clean' else ('combined', 'degraded', condition)
        for stratum in strata:
            groups[stratum][(speaker, label)].append(vec)
        counts[condition] += 1
        if n_voiced == 0:
            unvoiced[condition] += 1
        if i % 3000 == 0:
            print(f'  processed {i:,}/{len(paths):,}')
    if sum(counts.values()) != len(paths):
        raise AssertionError('Condition counts do not add to the number of recordings')
    if not counts['clean'] or any(counts[c] == 0 for c in H2_DEGRADATIONS):
        raise RuntimeError('Missing at least one of the six required acoustic conditions')

    results = {}
    for stratum in H2_FULL_ORDER:
        # Narrow strata can have fewer speaker pairs. Always disclose observed n.
        rows, speakers, avgs, class_counts = analyze(
            groups[stratum], n_bootstrap=a.bootstrap,
            min_speakers=min(a.min_speakers, 8) if stratum in H2_DEGRADATIONS else a.min_speakers,
            seed=42)
        if stratum in ('clean', 'degraded', 'combined') and len(speakers) != a.min_speakers:
            print(f'WARNING: {stratum} used {len(speakers)} paired speakers, '
                  f'expected at least {a.min_speakers}.')
        n_clips = sum(len(v) for v in groups[stratum].values())
        report_dir = output / 'by_condition' / stratum
        report_dir.mkdir(parents=True, exist_ok=True)
        write_csv(report_dir / 'h2_feature_tests.csv', rows, list(rows[0]))
        aggregates = []
        for speaker in speakers:
            for label in (0, 1):
                rec = {'speaker': speaker, 'label': label,
                       'class': 'AI' if label else 'Real',
                       'n_clips': class_counts[(speaker, label)]}
                rec.update(dict(zip(FEATURE_NAMES, [float(x) for x in avgs[(speaker,label)]])))
                aggregates.append(rec)
        write_csv(report_dir / 'h2_speaker_aggregates.csv', aggregates,
                  ['speaker','label','class','n_clips'] + FEATURE_NAMES)
        results[stratum] = {
            'rows': rows, 'by_feature': {r['feature']: r for r in rows},
            'speaker_rows': aggregates, 'n_clips': n_clips,
            'n_speakers': len(speakers),
            'family_counts': {
                family: {'within_family': sum(r['significant_family'] for r in rows if r['family'] == family),
                         'across42': sum(r['significant_global42'] for r in rows if r['family'] == family)}
                for family, _ in FAMILIES},
        }
        print(f"  {stratum:16s} {n_clips:6,d} clips | {len(speakers):2d} paired speakers | "
              f"global features={sum(r['significant_global42'] for r in rows)}/42")
    summary = {
        'design': 'speaker-matched, development-only, exploratory observational analysis',
        'source': 'one scan of original base and improved Akıncı–Özbay CQCC caches',
        'all_development_recordings': len(paths),
        'condition_counts': dict(sorted(counts.items())),
        'recordings_without_voiced_pitch_windows': dict(sorted(unvoiced.items())),
        'feature_families': {'CQCC_mean':20,'CQCC_std':20,'F0':2},
        'strata': {name: {'recordings':res['n_clips'], 'paired_speakers':res['n_speakers'],
                          'global_significant':sum(r['significant_global42'] for r in res['rows']),
                          'by_family':res['family_counts']} for name,res in results.items()},
        'interpretation_notes': [
            'Clean, degraded, per-augmentation and combined strata reuse development recordings and speakers.',
            'Within each stratum the family and 42-feature corrections are performed separately; no correction across eight strata.',
            'F0 standard deviation is mean within-window log-F0 SD, averaged at recording then speaker level.',
            'Non-significance is not proof of acoustic equivalence.',
            'Unadjusted 95% bootstrap intervals do not account for multiplicity.',
        ],
    }
    (output / 'h2_full_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    generate_full_chapter4_report(output, results, summary)
    print(f'\nDONE: {output / "README.md"}')
    print('  Five figures (PNG/SVG), comprehensive interpretation, and per-stratum raw CSV files.')
    print('  Old clean/ and all/ outputs were NOT modified.')


def _h2_count_global(result, family=None):
    return sum(bool(r['significant_global42']) for r in result['rows']
               if family is None or r['family'] == family)


def _h2_f0_comment(row, what):
    p_f = row['p_bonferroni_family']
    p_g = row['p_bonferroni_global42']
    diff = row['delta_ai_minus_real']
    if what == 'std':
        direction = ('lower' if diff < 0 else 'higher' if diff > 0 else 'unchanged')
        expected = ('consistent with the RRL expectation of reduced synthetic pitch variability'
                    if diff < 0 else 'not consistent with the expected reduction in synthetic pitch variability')
        subject = f'AI-generated speech had {direction} measured within-window log-F0 variability; this direction is {expected}.'
    else:
        subject = ('AI-generated speech had a slightly higher overall mean log-F0.' if diff > 0
                   else 'AI-generated speech had a slightly lower overall mean log-F0.' if diff < 0
                   else 'The estimated mean log-F0 was unchanged.')
    if p_g < .05:
        significance = 'The contrast remains significant after the global 42-feature Bonferroni correction.'
    elif p_f < .05:
        significance = ('The contrast passes correction within the two-measure F0 family, '
                        'but **not** the stricter 42-feature correction.')
    else:
        significance = 'No statistically significant contrast was detected after F0-family correction.'
    return f'{subject} {significance} (family p={_pvalue(p_f)}; global p={_pvalue(p_g)}; paired d_z={fmt(row["effect_dz"],3)}).'


def _h2_compact_f0_table(r):
    hdr = ['| F0 measure | Authentic mean | AI mean | AI − Authentic | Family p | Global p (42) | Paired d_z |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for f in ('F0_mean_logHz', 'F0_std_logHz'):
        v = r['by_feature'][f]
        hdr.append('| ' + ' | '.join([
            'Mean log-F0' if f.startswith('F0_mean') else 'Within-window log-F0 SD',
            fmt(v['real_mean']), fmt(v['ai_mean']), fmt(v['delta_ai_minus_real']),
            _pvalue(v['p_bonferroni_family']), _pvalue(v['p_bonferroni_global42']),
            fmt(v['effect_dz'],3)])+' |')
    return hdr


def _h2_cqcc_table(res, limit=6):
    sig=sorted([r for r in res['rows'] if r['family'].startswith('CQCC') and
                r['significant_global42']], key=lambda r:r['p_bonferroni_global42'])
    selected = sig[:limit]
    if len(selected) < min(3,limit):
        extras=sorted([r for r in res['rows'] if r['family'].startswith('CQCC') and
                        r['feature'] not in {t['feature'] for t in selected}],
                       key=lambda r:r['p_bonferroni_global42'])
        selected += extras[:max(0,min(3,limit)-len(selected))]
    lines=['| CQCC measurement | AI − Authentic | Global-adjusted p | Paired d_z | Significant globally? |',
           '|---|---:|---:|---:|---|']
    for row in selected:
        lines.append(f"| {row['feature']} | {fmt(row['delta_ai_minus_real'])} | "
                     f"{_pvalue(row['p_bonferroni_global42'])} | "
                     f"{fmt(row['effect_dz'],3)} | {'Yes' if row['significant_global42'] else 'No'} |")
    return lines


def _h2_cqcc_explanation(res):
    nmean=_h2_count_global(res,'CQCC_mean'); nstd=_h2_count_global(res,'CQCC_std')
    return (f'**Interpretation:** {nmean}/20 CQCC pooled-mean features and {nstd}/20 '
            'pooled within-window-variability features pass the across-42 correction '
            'in this acoustic condition. The table shows selected measurements, not all '
            '42 tests; see the corresponding CSV for the complete coefficient results. '
            'A detected acoustic difference does not by itself prove a gain from the fusion classifier.')


def generate_full_chapter4_figures(output, res):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Figures require matplotlib: python -m pip install matplotlib') from exc
    figs=output/'figures'; figs.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.size':10,'figure.facecolor':'white',
                         'axes.spines.top':False,'axes.spines.right':False})
    palette={'Real':'#346986','AI':'#c06b46'}
    manifest={}
    def save(fig,key):
        fig.savefig(figs/(key+'.png'),dpi=230,bbox_inches='tight')
        fig.savefig(figs/(key+'.svg'),bbox_inches='tight')
        plt.close(fig)
        manifest[key]=f'figures/{key}.png'

    # Figure 1. F0 mean-vs-variability paired speaker contrasts on clean speech.
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.25),layout='constrained')
    for ax,f,title in zip(axes,['F0_mean_logHz','F0_std_logHz'],
                          ['Mean voiced log-F0','Within-window log-F0 variability']):
        _,real,ai=_paired_speaker_points(res['clean']['speaker_rows'],f)
        lo=min(real.min(),ai.min()); hi=max(real.max(),ai.max())
        pad=max((hi-lo)*.07,.004); low=lo-pad;high=hi+pad
        ax.plot([low,high],[low,high],'--',color='#888',lw=1.5)
        ax.scatter(real,ai,s=32,color='#346986',alpha=.8,edgecolor='white',linewidth=.4)
        ax.set(xlim=(low,high),ylim=(low,high),xlabel='Authentic speaker mean',
               ylabel='AI-generated speaker mean',title=title)
        ax.grid(alpha=.14)
    fig.suptitle('Clean speech: 32 speaker-matched F0 comparisons',weight='bold')
    save(fig,'figure_01_clean_f0_means_and_variability')

    # Figure 2. Pooled degraded voiced log-F0 variability by speaker.
    fig,ax=plt.subplots(figsize=(6.5,5.5),layout='constrained')
    _,real,ai=_paired_speaker_points(res['degraded']['speaker_rows'],'F0_std_logHz')
    lo=min(real.min(),ai.min());hi=max(real.max(),ai.max());pad=max(.004,(hi-lo)*.07)
    ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'--',color='#888',label='Equal variability')
    ax.scatter(real,ai,s=42,color='#c06b46',alpha=.85,edgecolor='white',linewidth=.45)
    ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),
           xlabel='Authentic speaker mean (log-F0 SD)',
           ylabel='AI-generated speaker mean (log-F0 SD)',
           title='Degraded speech: paired pitch-variability measurements')
    ax.legend();ax.grid(alpha=.13)
    save(fig,'figure_02_degraded_f0_variability')

    # Figure 3. Speaker bootstrap CIs across the five degradation types.
    fig,ax=plt.subplots(figsize=(10.8,5),layout='constrained')
    names=list(H2_DEGRADATIONS)
    for k,name in enumerate(names):
        v=res[name]['by_feature']['F0_std_logHz']
        delta=v['delta_ai_minus_real'];lo=v['bootstrap_ci95_low'];hi=v['bootstrap_ci95_high']
        color='#296e65' if v['significant_global42'] else '#c48242' if v['significant_family'] else '#78818a'
        if all(math.isfinite(t) for t in (delta,lo,hi)):
            ax.errorbar(delta,k,xerr=[[max(0,delta-lo)],[max(0,hi-delta)]],
                        fmt='o',color=color,capsize=4,ms=7,lw=1.7)
    ax.axvline(0,ls='--',color='#555')
    ax.set(yticks=range(len(names)),yticklabels=[H2_DISPLAY[n] for n in names],
           ylim=(len(names)-.5,-.5),xlabel='AI − Authentic: mean within-window log-F0 SD',
           title='Pitch-variability differences within each augmentation type')
    ax.grid(axis='x',alpha=.2)
    save(fig,'figure_03_augmentation_f0_variability_ci')

    # Figure 4. CQCC paired effect sizes, strongest combined-condition features.
    combined=res['combined']
    select=sorted((r for r in combined['rows'] if r['family'].startswith('CQCC')),
                  key=lambda r:(not r['significant_global42'],r['p_bonferroni_global42']))[:10]
    select=sorted(select,key=lambda r:r['effect_dz'])
    fig,ax=plt.subplots(figsize=(10.5,5.7),layout='constrained')
    colors=['#296e65' if r['significant_global42'] else '#9ba3ab' for r in select]
    ax.barh(range(len(select)),[r['effect_dz'] for r in select],color=colors)
    ax.set(yticks=range(len(select)),yticklabels=[r['feature'] for r in select],
           xlabel='Paired Cohen d_z (AI minus authentic)',
           title='Ten leading CQCC contrasts — all development conditions')
    ax.axvline(0,color='#555',lw=1);ax.grid(axis='x',alpha=.16)
    save(fig,'figure_04_combined_top_cqcc_effects')

    # Figure 5. Number of globally significant features, by family and condition.
    keys=['clean','degraded',*H2_DEGRADATIONS,'combined']
    fig,ax=plt.subplots(figsize=(12,5.6),layout='constrained')
    x=np.arange(len(keys));width=.24
    for i,(family,color,label) in enumerate([('CQCC_mean','#346986','CQCC mean'),
                                            ('CQCC_std','#296e65','CQCC variability'),
                                            ('F0','#c06b46','F0')]):
        vals=[_h2_count_global(res[key],family) for key in keys]
        ax.bar(x+(i-1)*width,vals,width,color=color,label=label)
    ax.set(xticks=x,xticklabels=[H2_DISPLAY[k].replace(' + ',' +\n') for k in keys],
           ylabel='Features with global Bonferroni p < 0.05',
           title='Acoustic differences observed under each analysis condition')
    ax.tick_params(axis='x',labelsize=8,rotation=18)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(ncol=3);ax.grid(axis='y',alpha=.17)
    save(fig,'figure_05_significant_features_by_condition')
    return manifest


def generate_full_chapter4_report(output, res, summary):
    figs=generate_full_chapter4_figures(output,res)
    total=summary['all_development_recordings']
    lines=[
        '# Chapter 4 — Hypothesis 2: acoustic differences in Filipino speech',
        '',
        '**Exploratory speaker-matched analysis of improved CQCC and F0 features — development dataset only.**',
        '',
        '## A. Hypothesis and statistical treatment',
        '',
        '**H₀:** Authentic and AI-generated Filipino speech do not differ significantly '
        'in the tested Akıncı–Özbay-inspired CQCC and F0 summary measurements.',
        '',
        '**H₁:** There are statistically significant differences in the tested '
        'CQCC and F0 feature distributions between authentic and AI-generated '
        'Filipino speech. This retains the original hypothesis wording, with '
        'MFCC updated to CQCC.',
        '',
        '**Design.** The analysis uses only the development feature caches and '
        'does not access held-out test recordings. Each 1-second window is first '
        'summarized within its recording; each recording contributes equally within '
        'speaker and class; the speaker-class means are then paired across authentic '
        'and AI-generated speech. **A speaker pair, not a window or clip, is the '
        'statistical observation.** Speaker matching does not guarantee the same '
        'utterance/transcript or exclude different synthesizers and recording chains.',
        '',
        'Forty CQCC descriptors are tested (20 means and 20 within-window standard '
        'deviations), alongside two F0 measurements (mean voiced log-F0 and mean '
        'within-window voiced log-F0 SD). The F0 expert uses an 8-D input but only '
        'these two interpretable RRL-directed measurements are tested here. '
        'Paired-difference normality is assessed using Shapiro–Wilk; a two-sided '
        'paired t-test or Wilcoxon signed-rank test follows. Bonferroni adjustments '
        'are applied **within each feature family (20/20/2)** and independently '
        '**across 42 features per condition**. Paired Cohen d_z is the effect size; '
        'bootstrap confidence intervals are speaker-based and not adjusted for '
        'multiple comparisons.',
        '',
        '**Analysis strata:** clean; all five degraded conditions pooled; each '
        'individual degradation type; and all development conditions combined. '
        '**These strata overlap, so results are not independent replications.** '
        'The family/global Bonferroni corrections are within each stratum; '
        'this exploratory series is not additionally corrected across the eight '
        'condition analyses. The original thesis majority-of-features rule is '
        'reported rather than replaced after examining results.',
        '',
        '### Table H2.1. Development dataset composition',
        '',
        '| Acoustic condition | Recordings | Share of development data |',
        '|---|---:|---:|',
    ]
    for key in ('clean',*H2_DEGRADATIONS):
        count=summary['condition_counts'][key]
        lines.append(f'| {H2_DISPLAY[key]} | {count:,} | {100*count/total:.1f}% |')
    lines.extend([f'| **Total (combined)** | **{total:,}** | **100%** |','',
                  '**Explanation (Table H2.1):** Each original development recording has '
                  'one assigned acoustic condition. The pooled degraded group contains '
                  'the five non-clean rows; the combined group contains all six rows. '
                  'Neither group adds new independent recordings.',''])

    def add_figure(key,caption,explanation):
        lines.extend([f'**{caption}**','',f'![{caption}]({figs[key]})','',
                      f'**Explanation ({caption.split(". ", 1)[0]}):** {explanation}',''])

    def add_f0(name, tablabel):
        data=res[name];mean=data['by_feature']['F0_mean_logHz'];std=data['by_feature']['F0_std_logHz']
        lines.extend([f'### Table {tablabel}. F0 mean and variability ({H2_DISPLAY[name]})','',
                      *_h2_compact_f0_table(data),'',
                      '**Explanation:** '+_h2_f0_comment(mean,'mean')+' '+_h2_f0_comment(std,'std'),''])
    def add_cqcc(name,tablabel):
        data=res[name]
        lines.extend([f'### Table {tablabel}. Selected CQCC measurements ({H2_DISPLAY[name]})',
                      '',*_h2_cqcc_table(data),'',_h2_cqcc_explanation(data),''])

    clean=res['clean']
    lines.extend(['## B. Clean-speech acoustic differences','',
                  f"The primary clean-speech analysis used **{clean['n_clips']:,} recordings** "
                  f"summarized into **{clean['n_speakers']} speaker-matched observations**. "
                  'No training or inference results are used in this acoustic-feature test.',''])
    add_f0('clean','H2.2')
    add_cqcc('clean','H2.3')
    add_figure('figure_01_clean_f0_means_and_variability','Figure H2.1. Clean F0 speaker-matched means and variability',
               'Each point represents one speaker, with authentic features on the '
               'horizontal axis and AI-generated features on the vertical axis. '
               'The dashed identity line marks equality. Points below this line '
               'in the variability panel indicate lower AI pitch variability. '
               'The figure does not test equivalence of mean pitch.')

    degraded=res['degraded']
    lines.extend(['## C. Degraded-speech acoustic differences (all degradations pooled)','',
                  f"The pooled degraded analysis used **{degraded['n_clips']:,} recordings** "
                  f"from five assigned degradations and **{degraded['n_speakers']} "
                  'speaker-matched observations**. These are development augmentation '
                  'conditions, **not the separate held-out degraded test set**.',''])
    add_f0('degraded','H2.4')
    add_cqcc('degraded','H2.5')
    add_figure('figure_02_degraded_f0_variability','Figure H2.2. Pooled degraded F0 variability by speaker',
               'The scatterplot shows the mean within-window log-F0 SD per speaker '
               'for authentic and AI speech. Its paired pattern is descriptive; '
               'family- and global-adjusted tests in Table H2.4 determine '
               'whether this contrast meets the predefined thresholds.')

    lines.extend(['## D. Acoustic differences within each individual degradation','',
                  '### Table H2.6. Augmentation-specific F0 variability and CQCC significance',
                  '',
                  '| Augmentation | Recordings | Speaker pairs | F0 SD AI−Real | F0 family p | F0 global p | Global CQCC hits |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for name in H2_DEGRADATIONS:
        r=res[name];f0=r['by_feature']['F0_std_logHz']
        lines.append(f'| {H2_DISPLAY[name]} | {r["n_clips"]:,} | {r["n_speakers"]} | '
                     f'{fmt(f0["delta_ai_minus_real"])} | {_pvalue(f0["p_bonferroni_family"])} | '
                     f'{_pvalue(f0["p_bonferroni_global42"])} | '
                     f'{_h2_count_global(r,"CQCC_mean")+_h2_count_global(r,"CQCC_std")}/40 |')
    lines.extend(['','**Explanation (Table H2.6):** Every row is a separate '
                  'speaker-matched comparison within one augmentation. Different '
                  'recording and speaker counts affect precision. Negative F0 SD '
                  'differences indicate less measured AI variability. Significance '
                  'is corrected within each condition’s 42 features, **not across '
                  'all five augmentation analyses**.',''])
    for idx,name in enumerate(H2_DEGRADATIONS,1):
        r=res[name]; f0=r['by_feature']['F0_std_logHz']
        mean=r['by_feature']['F0_mean_logHz']
        lines.extend([f'### D.{idx}. {H2_DISPLAY[name]}','',
                      f"The comparison contains **{r['n_clips']:,} recordings** "
                      f"from **{r['n_speakers']} paired speakers**.",'',
                      f'**Table H2.6.{idx}. F0 contrasts within {H2_DISPLAY[name].lower()}**','',
                      *_h2_compact_f0_table(r),'',
                      '**Explanation:** '+_h2_f0_comment(mean,'mean')+' '+_h2_f0_comment(f0,'std')+' '+
                      f"{_h2_count_global(r,'CQCC_mean')+_h2_count_global(r,'CQCC_std')}/40 "
                      'CQCC features passed the across-42 correction in this condition. '
                      'The smaller stratum is exploratory and must not be treated '
                      'as an independently replicated experiment.',''])
    add_figure('figure_03_augmentation_f0_variability_ci',
               'Figure H2.3. Augmentation-specific F0 variability differences',
               'Dots show mean speaker-paired AI-minus-authentic variability; horizontal '
               'error bars are unadjusted speaker-bootstrap 95% confidence intervals. '
               'Left of zero means lower AI variability. These intervals are not '
               'simultaneous 95% confidence bands and should not replace '
               'the adjusted significance tests in the tables.')

    both=res['combined']
    lines.extend(['## E. Combined clean and degraded development conditions','',
                  f"The combined analysis includes **{both['n_clips']:,} unique development "
                  f"recordings** and **{both['n_speakers']} paired speakers**. "
                  'It summarizes the complete assigned development distribution '
                  'and overlaps completely with the clean/degraded components.',''])
    add_f0('combined','H2.7')
    add_cqcc('combined','H2.8')
    add_figure('figure_04_combined_top_cqcc_effects',
               'Figure H2.4. Leading combined-condition CQCC effect sizes',
               'Bars show the paired Cohen d_z effect size for the ten leading '
               'combined-condition CQCC measurements ranked by global-adjusted '
               'p-value; green indicates a globally significant coefficient. '
               'Standardized effect sizes indicate direction/magnitude, not '
               'independent predictive gain or causal importance.')

    lines.extend(['## F. Cross-condition summary and comparison','',
                  '### Table H2.9. Significant acoustic features by analysis condition','',
                  '| Analysis | Pairs | CQCC mean (of 20) | CQCC SD (of 20) | F0 (of 2) | All (of 42) |',
                  '|---|---:|---:|---:|---:|---:|'])
    for name in H2_FULL_ORDER:
        r=res[name]
        lines.append(f'| {H2_DISPLAY[name]} | {r["n_speakers"]} | '
                     f'{_h2_count_global(r,"CQCC_mean")} | '
                     f'{_h2_count_global(r,"CQCC_std")} | '
                     f'{_h2_count_global(r,"F0")} | '
                     f'{_h2_count_global(r)} |')
    lines.extend(['', '**Explanation (Table H2.9):** The counts reflect features with '
                  'Bonferroni-adjusted p < 0.05 within each analysis of 42 measurements. '
                  'Conditions reuse speakers and recordings and were not jointly '
                  'corrected for eight separate investigations; the results show '
                  'patterns, not independent confirmations.',''])
    add_figure('figure_05_significant_features_by_condition',
               'Figure H2.5. Counts of significant features across acoustic conditions',
               'The bars summarize the condition-specific globally corrected feature '
               'counts from Table H2.9. They should not be interpreted as a model '
               'accuracy score or compared as independent sample replications.')

    # RRL interpretation anchored to the supplied thesis excerpt, not invented citations.
    c_mean=clean['by_feature']['F0_mean_logHz'];c_std=clean['by_feature']['F0_std_logHz']
    d_mean=degraded['by_feature']['F0_mean_logHz'];d_std=degraded['by_feature']['F0_std_logHz']
    lines.extend([
        '## G. Comparison with the Review of Related Literature',
        '',
        '**F0 mean (overall pitch level).** The thesis RRL explains that genuine '
        'and generated recordings may occupy similar pitch ranges. In the present '
        f'clean contrast, the F0-family-adjusted mean-pitch p-value is {_pvalue(c_mean["p_bonferroni_family"])}; '
        f'for pooled degradation it is {_pvalue(d_mean["p_bonferroni_family"])}. '
        'Where those differences are nonsignificant, the finding is compatible with '
        'the RRL expectation but **does not prove equivalence** of mean pitch.',
        '',
        '**F0 standard deviation (pitch variability).** The supplied RRL attributes '
        'to Xue et al. (2022) the broader expectation that generated speech may exhibit '
        'more concentrated or stable pitch structure. The observed F0-variability '
        f'differences are {fmt(c_std["delta_ai_minus_real"])} on clean recordings and '
        f'{fmt(d_std["delta_ai_minus_real"])} under pooled degradation (AI minus authentic). '
        'Their direction should be assessed jointly with both within-family and '
        'across-42-adjusted p-values in Tables H2.2 and H2.4, rather than assumed '
        'significant based only on a matching sign. The cited work examined richer '
        'F0-based subbands, whereas this thesis uses two compact pooled YIN/log-F0 '
        'descriptors; this is *conceptual comparison*, not direct replication.',
        '',
        '**CQCC means and variability.** The study’s proposed CQCC features are '
        'inspired by the Akıncı–Özbay (2026) cepstral-feature investigation described '
        'in the thesis. Differences in individual means and within-window standard '
        'deviations may support the motivation to include CQCC, but do **not** '
        'demonstrate that CQCC causes a specific improvement in classifier F1 or EER. '
        'Such a claim requires a controlled model-component ablation.',
        '',
        '## H. Hypothesis 2 decision and limitations',
        '',
        f"For the primary clean condition, **{_h2_count_global(clean)}/42** measurements "
        f"were significant across the 42-feature correction; for the pooled degraded "
        f"condition **{_h2_count_global(degraded)}/42**; and for the complete combined "
        f"development set **{_h2_count_global(both)}/42**. "
        'The original Chapter 3 majority-of-tested-features decision rule should '
        'remain explicit. If fewer than 22 of 42 features are significant, that '
        'majority threshold is not satisfied, even when several individual '
        'measurements are significantly different. '
        + (f'**Primary clean-condition decision:** '
           f'{"The majority criterion was met" if _h2_count_global(clean)>=22 else "The majority criterion was not met"} '
           f'({_h2_count_global(clean)}/42 globally significant features). ') +
        'Report **feature-specific CQCC '
        'and F0 evidence** rather than claiming blanket support for H₁ or retroactively '
        'changing the rejection criterion. If any stratum does satisfy that rule, '
        'interpret it with caution because of multiple exploratory stratum analyses.',
        '',
        '**Limitations.** The test unit is the speaker, not individual overlapping '
        'windows. Speaker matching does not necessarily match exact utterance or '
        'transcript. Augmentation effects and synthesis/recording-chain differences '
        'can affect the observed features. F0 SD describes *within-window log-F0* '
        'variability averaged over voiced windows, not the raw-F0 distribution across '
        'an entire clip. A nonsignificant difference cannot establish equivalence. '
        'Each analysis controls 42 simultaneous feature comparisons, but not '
        'the additional eight analyses across acoustic conditions. Bootstrap CIs '
        'are unadjusted. These results must not be described as analysis of a '
        'new independent degraded test set.',
        '',
        '## I. Reproducibility and accompanying data',
        '',
        '- `h2_full_summary.json`: condition counts, denominators, and significant-feature summaries.',
        '- `by_condition/<condition>/h2_feature_tests.csv`: all 42 raw and adjusted statistical tests for each condition.',
        '- `by_condition/<condition>/h2_speaker_aggregates.csv`: exact speaker/class summaries used in the tests.',
        '- `figures/`: five corresponding PNG and editable SVG figure pairs.',
        '',
        '**Recommended thesis usage:** paste the relevant sections into Chapter 4, '
        'renumber tables and figures to match your manuscript, and verify the '
        'underlying cited studies before final submission.',
        '',
    ])
    # Keep all interpretations adjacent to their tables or charts in the one primary README.
    path=output/'README.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    (output/'h2_figure_manifest.json').write_text(
        json.dumps(figs,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(os.environ.get('THESIS_BASE_DIR',r'E:\Thesis')),
                   help='Thesis root containing features/ and features_experiments/')
    p.add_argument('--stage',choices=('check','run','report','full'),default='check',
                   help='report generates Chapter 4 tables/figures using existing H2 CSV+JSON, without rescanning features')
    p.add_argument('--condition',choices=('clean','all'),default='clean',
                   help='Clean-only primary analysis; all is an augmentation sensitivity analysis')
    p.add_argument('--expected-clips',type=int,default=26209)
    p.add_argument('--min-speakers',type=int,default=32)
    p.add_argument('--bootstrap',type=int,default=4000)
    p.add_argument('--output',type=Path,default=None)
    a=p.parse_args()
    if a.bootstrap<100 or a.min_speakers<3 or a.expected_clips<0:
        p.error('Require bootstrap >=100, min-speakers >=3, expected-clips >=0')
    if a.stage=='full':
        run_full_chapter4(a)
        return
    if a.stage=='report':
        output=a.output or (a.root/'results'/'hypothesis2_cqcc_f0'/a.condition)
        write_chapter4_from_saved(output)
        return
    base,side,paths=find_files(a.root)
    sidecar_count=sum(1 for _ in side.rglob('*.npz'))
    print('H2 CQCC–F0 analysis | development dataset ONLY')
    print(f'  Base clips: {len(paths):,}, improved sidecars: {sidecar_count:,}')
    print(f'  Input condition: {a.condition}; one observation per speaker/class')
    if a.expected_clips and (len(paths)!=a.expected_clips or sidecar_count!=a.expected_clips):
        raise RuntimeError(f'Expected {a.expected_clips:,} complete files in both caches, '
                           f'found base={len(paths):,}, improved={sidecar_count:,}. '
                           'Run preparation verification before statistical testing.')
    if a.stage=='check':
        sample=paths[:2]+paths[-2:] if len(paths)>3 else paths
        for x in sample:
            y=side/x.relative_to(base)
            print('  checking',x.relative_to(base))
            print('    label,speaker,condition:',read_metadata(x,y))
        print('PASS: directory counts and sampled schema/metadata verified (not full cache).')
        print('NEXT: python improvedCQCC_hypothesis2_acoustics.py --stage run')
        return
    groups=defaultdict(list)
    total_counts=Counter(); used_counts=Counter()
    skipped_unvoiced=0
    for i,x in enumerate(paths,1):
        # Read base metadata first to avoid opening paper sidecars of excluded conditions.
        with np.load(x,allow_pickle=False) as meta:
            condition=str(meta['condition'])
        total_counts[condition]+=1
        if a.condition!='all' and condition!=a.condition:
            continue
        y=side/x.relative_to(base)
        if not y.is_file():
            raise FileNotFoundError(f'Improved CQCC sidecar missing: {y}')
        label,speaker,condition,values,n_win,n_voiced=feature_record(x,y)
        if not np.isfinite(values[40:]).all(): skipped_unvoiced+=1
        groups[(speaker,label)].append(values)
        used_counts[condition]+=1
        if i%3000==0:print(f'  examined {i:,}/{len(paths):,} recordings')
    rows,speakers,averages,counts=analyze(groups,n_bootstrap=a.bootstrap,
                                          min_speakers=a.min_speakers)
    output=a.output or (a.root/'results'/'hypothesis2_cqcc_f0'/a.condition)
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'h2_feature_tests.csv',rows,list(rows[0].keys()))
    by_speaker=[]
    for speaker in speakers:
        for label in (0,1):
            entry={'speaker':speaker,'label':label,'class':'AI' if label else 'Real',
                   'n_clips':counts[(speaker,label)]}
            entry.update({name:float(v) for name,v in zip(FEATURE_NAMES,averages[(speaker,label)])})
            by_speaker.append(entry)
    write_csv(output/'h2_speaker_aggregates.csv',by_speaker,
              ['speaker','label','class','n_clips']+FEATURE_NAMES)
    family_summary=[]
    for family,k in FAMILIES:
        chunk=[r for r in rows if r['family']==family]
        family_summary.append({'family':family,'features':k,
                               'family_significant':sum(r['significant_family'] for r in chunk),
                               'global_significant':sum(r['significant_global42'] for r in chunk),
                               'alpha_family':.05/k})
    global_hits=sum(r['significant_global42'] for r in rows)
    summary={
        'analysis':'H2 speaker-matched acoustic differences; exploratory, not causal',
        'primary_condition':a.condition, 'split':'development ONLY',
        'n_development_clips':len(paths),'n_selected_clips':sum(used_counts.values()),
        'n_speaker_pairs':len(speakers),'selected_conditions':dict(sorted(used_counts.items())),
        'all_conditions':dict(sorted(total_counts.items())),
        'n_clips_without_voiced_F0_windows':skipped_unvoiced,
        'features_tested':len(rows),'global_bonferroni_alpha_per_raw_p':.05/len(rows),
        'global_significant_features':global_hits, 'families':family_summary,
        'cqcc_global_significant':any(r['significant_global42'] for r in rows if r['family'].startswith('CQCC')),
        'f0_global_significant':any(r['significant_global42'] for r in rows if r['family']=='F0'),
        'methodology':{
            'unit':'speaker matched class mean; equal weight per recording, then equal weight per speaker',
            'contrast':'AI minus authentic, per speaker',
            'cqcc':'20 averaged per-window coefficient means plus 20 averaged per-window coefficient standard deviations',
            'f0':'mean of per-window log-F0 mean and log-F0 std, only windows with >0 voiced fraction',
            'normality':'Shapiro–Wilk on speaker-level paired differences for each feature',
            'selection':'paired two-sided t-test when Shapiro p >= 0.05; otherwise two-sided Wilcoxon signed-rank',
            'multiplicity':'Bonferroni in CQCC mean (20), CQCC std (20), F0 (2), also across all 42',
            'effect_size':'paired Cohen dz (mean paired difference / SD of paired differences)',
            'confidence_interval':'speaker bootstrap percentile 95% CI of mean paired difference (not multiplicity-adjusted)',
            'limitation':'speaker-matched, NOT verified utterance-paired; no independence claimed for clips/windows'},
    }
    with (output/'h2_summary.json').open('w',encoding='utf-8') as f:
        json.dump(summary,f,indent=2,ensure_ascii=False,allow_nan=False)
    ordered=sorted(rows,key=lambda x:(not x['significant_global42'],
                         x['p_bonferroni_global42'] if math.isfinite(x['p_bonferroni_global42']) else math.inf))
    lines=['# Hypothesis 2 — Acoustic differences: CQCC and F0','',
           'Development-only, speaker-matched analysis. Exploratory interpretation.','',
           f'- Conditions: **{a.condition}**; clips analyzed: **{sum(used_counts.values()):,}**',
           f'- Complete speaker pairs: **{len(speakers)}** (real and AI), one speaker pair = one statistical observation',
           '- CQCC: 20 window-pooled coefficient means + 20 window-pooled coefficient standard deviations',
           '- F0: log-F0 mean and standard deviation, voiced windows only',
           '- Shapiro–Wilk on *paired speaker differences*, then paired t-test or Wilcoxon; two-sided',
           '- Bonferroni: per family and across all 42 primary comparisons',
           '- Effect size: paired Cohen dz; positive means AI > Real',
           '', '## Significance by feature family','',
           '| Family | Tests | Family-corrected significant | Globally corrected significant |',
           '|---|---:|---:|---:|']
    for f in family_summary:
        lines.append(f"| {f['family']} | {f['features']} | {f['family_significant']} | {f['global_significant']} |")
    lines += ['',f'At least one of 42 globally Bonferroni-significant features: **{"yes" if global_hits else "no"}**.',
              'This supports evidence for at least one acoustic distribution shift, **not** a claim that',
              'both CQCC and F0 independently differ or that a majority of coefficients differ.',
              '', '## Top features by globally adjusted p-value','',
              '| Feature | Real mean | AI mean | AI − Real | Test | Adjusted p (42) | Paired dz |',
              '|---|---:|---:|---:|---|---:|---:|']
    for r in ordered[:15]:
        lines.append('| '+ ' | '.join([
            r['feature'],fmt(r['real_mean']),fmt(r['ai_mean']),fmt(r['delta_ai_minus_real']),
            r['test'],fmt(r['p_bonferroni_global42'],6),fmt(r['effect_dz'])])+' |')
    lines += ['', '## Interpretation safeguards','',
              '1. Each clip contributes one summary; overlapping windows were **not** treated as independent observations.',
              '2. Class comparisons are matched **by speaker**, not by exact transcript or audio pair.',
              '3. CQCC standard deviations summarize variation **within** analysis windows, averaged for each clip.',
              '4. F0 standard deviation is the mean of voiced-window log-F0 standard deviations; fully unvoiced windows are excluded.',
              '5. Raw source/generator and audio-processing differences may contribute to observed feature shifts.',
              '6. No held-out test recordings or labels are used; results do not establish classification performance.',
              '7. Bootstrapped 95% intervals are not simultaneously multiple-comparison-corrected.',
              '', 'Full results: `h2_feature_tests.csv`; speaker summary: `h2_speaker_aggregates.csv`.', '']
    (output/'h2_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\nH2 RESULTS (AI minus Real; corrected for multiple testing)')
    print(f'  Selected clips: {sum(used_counts.values()):,}; paired development speakers: {len(speakers)}')
    for f in family_summary:
        print(f"  {f['family']:<12}: {f['family_significant']}/{f['features']} family adjusted; "
              f"{f['global_significant']}/{f['features']} globally adjusted")
    print(f'  Significant after across-42 correction: {global_hits}/{len(rows)}')
    for r in ordered[:5]:
        print(f"  {r['feature']:<19}: delta={fmt(r['delta_ai_minus_real'])} "
              f"p_global={fmt(r['p_bonferroni_global42'],6)} dz={fmt(r['effect_dz'])}")
    print(f'\nReports saved ONLY to: {output}')
    print('  h2_feature_tests.csv, h2_speaker_aggregates.csv, h2_summary.json, h2_report.md')
    print('Interpretation: speaker-matched exploratory test; no inference of causal class effect.')
    write_chapter4_from_saved(output)


if __name__=='__main__':
    main()
