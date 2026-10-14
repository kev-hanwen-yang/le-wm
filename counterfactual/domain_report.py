"""Diagnostic tables/figures from saved numbers only, never model calls."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def write_domain_report(results):
    results=Path(results)
    manifest=json.loads((results/'domain_manifest.json').read_text())
    records=json.loads((results/'family_metrics.json').read_text())
    targets=('agent_location','block_location','block_angle')
    rows=[]
    for r in records:
        for d in r['doses']:
            rows.append(dict(family_id=r['family_id'],partition=r['partition'],stratum=r['stratum'],dose=d['dose'],
                **{f'{kind}_{t}':d[kind][t] for kind in ('absolute','differential') for t in targets},
                **{k:d[k] for k in ('latent_l2','latent_norm','nearest_source_reference','direction_cosine','magnitude_error_world_px')}))
    with (results/'readout_measurements.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    forecasts=[]
    for r in records:
        for key,value in r['trajectory_metrics'].items():
            for scene,curve in enumerate(value):
                for k,v in enumerate(curve): forecasts.append(dict(family_id=r['family_id'],stratum=r['stratum'],metric=key,dose=(0,10,20,30)[scene] if 'effect' not in key else (10,20,30)[scene],h=15+5*k,value=v))
    with (results/'natural_predictor_measurements.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(forecasts[0]));writer.writeheader();writer.writerows(forecasts)
    nuisance=[dict(family_id=r['family_id'],condition=k,**{name:value for name,value in v.items() if name!='latent_to_dose_ratio'}) for r in records for k,v in r['nuisance'].items()]
    with (results/'nuisance_measurements.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(nuisance[0]));writer.writeheader();writer.writerows(nuisance)
    for kind in ('absolute','differential'):
        fig,axes=plt.subplots(1,3,figsize=(11,3.2))
        for ax,t in zip(axes,targets):
            for stratum,marker in (('free_space','o'),('near_contact','x')):
                x=[d['dose'] for r in records if r['stratum']==stratum for d in r['doses']]
                y=[d[kind][t] for r in records if r['stratum']==stratum for d in r['doses']]
                ax.scatter(x,y,s=15,alpha=.6,marker=marker,label=stratum)
            ax.axhline(.1 if t=='block_angle' else 5.,color='red',ls='--',label='physical cap')
            ax.set(xlabel='Agent dose (world px)',ylabel='Error (rad)' if t=='block_angle' else 'Error (world px)',title=t.replace('_',' '))
        axes[0].legend(fontsize=7);fig.suptitle(f"{manifest['partition']} — natural readout {kind} error (not H2)")
        fig.tight_layout();fig.savefig(results/f'readout_{kind}.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.2))
    for ax,t in zip(axes,targets):
        for mode in ('encoded','teacher_forced','open_loop'):
            keys=[f'{mode}_absolute_{t}_h{h}' for h in (15,20,25,30,35)]
            stats=[manifest['summary'][k] for k in keys]
            ax.plot([15,20,25,30,35],[v['rms'] for v in stats],label=mode)
            ax.fill_between([15,20,25,30,35],[v['ci95'][0] for v in stats],[v['ci95'][1] for v in stats],alpha=.12)
        ax.set(xlabel='Raw horizon h',ylabel='RMS rad' if t=='block_angle' else 'RMS world px',title=t.replace('_',' '))
    axes[0].legend(fontsize=7);fig.suptitle('Natural prediction baseline; family intervals, no latent intervention')
    fig.tight_layout();fig.savefig(results/'natural_prediction.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.5))
    labels=['dose10','dose20','dose30','zero_repeat','ffv1','x264_420_crf23','x264_444_crf23']
    data=[[r['doses'][j]['latent_l2'] for r in records] for j in (1,2,3)]
    data += [[r['nuisance'][k]['latent_l2'] for r in records] for k in labels[3:]]
    ax.boxplot(data,tick_labels=labels);ax.set(ylabel='Latent Euclidean L2',title='Physical signals versus same-origin nuisance')
    ax.tick_params(axis='x',labelrotation=20);fig.tight_layout();fig.savefig(results/'signal_and_nuisance.png',dpi=160);plt.close(fig)
