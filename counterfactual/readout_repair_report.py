"""R0/R1 reporting from saved numerical evidence, never new fitting."""
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_r0_report(results):
    results=Path(results);m=json.loads((results/'repair_manifest.json').read_text())
    rows=[]
    for domain in m['measurements']['absolute_domains']:
        for target,v in domain['absolute'].items():
            rows.append(dict(domain=domain['domain'],target=target,n_families=v['n_families'],euclidean_or_wrapped_rmse=v['rms'],
                ci_low=v['ci95'][0],ci_high=v['ci95'][1],coordinate_pooled_rmse=v['coordinate_pooled_rmse'],
                x_or_angle_rmse=v['coordinate_rmse'][0],y_rmse=v['coordinate_rmse'][1] if len(v['coordinate_rmse'])>1 else None))
    write_csv(results/'r0_domain_comparison.csv',rows)
    path=m['artifacts']['audit_arrays.h5']['path']
    with h5py.File(path) as f:
        ids=json.loads(f.attrs['identities']);source=next(i for i,row in enumerate(ids) if row['domain']=='source')
        near=next(i for i,row in enumerate(ids) if row['domain']=='prospective' and row['stratum']=='near_contact')
        fig,axes=plt.subplots(1,4,figsize=(12,3.2))
        for ax,index,title in zip(axes,(0,near,source,source+1),('Prospective free','Prospective near','Source (historical)','Same stored pose, prospective')):
            image=f['images'][index];label=f['labels'][index]
            ax.imshow(image)
            scale=lambda p:(np.asarray(p)+.5)*224/512-.5
            a=scale(label[:2]);b=scale(label[2:4]);c=scale(label[2:4]+np.array([-np.sin(label[4]),np.cos(label[4])])*45)
            ax.scatter(*a,s=35,c='red',marker='+',label='agent target')
            ax.scatter(*b,s=35,c='yellow',marker='+',label='block origin target')
            ax.scatter(*c,s=25,c='magenta',marker='x',label='block mass center (not target)')
            ax.set(title=title);ax.axis('off')
        axes[0].legend(fontsize=6,loc='lower left');fig.suptitle('Independent label geometry: 512 world → 224 RGB grid')
        fig.tight_layout();fig.savefig(results/'r0_geometry.png',dpi=160);plt.close(fig)


def write_r1_report(results):
    results=Path(results);rows=json.loads((results/'readout_comparison.json').read_text())
    curves=json.loads((results/'learning_curves.json').read_text())
    absolute=[];doses=[]
    for row in rows:
        a=row['metrics']['absolute']
        absolute.append(dict(candidate_id=row['candidate_id'],method=row['method'],head=row['head'],training_families=row['training_families'],selected=row['selected'],
            rmse=a.get('rms'),ci_low=a.get('ci95',[None,None])[0],ci_high=a.get('ci95',[None,None])[1],upper95=a.get('upper95'),
            coordinate_rmse_x_or_angle=a.get('coordinate_rmse',[None])[0],coordinate_rmse_y=a.get('coordinate_rmse',[None,None])[1] if len(a.get('coordinate_rmse',[]))>1 else None,
            coordinate_pooled_rmse=a.get('coordinate_pooled_rmse'),cap_status=a.get('cap_status'),r2=a.get('mean_coordinate_r2')))
        for dose,v in row['metrics']['pairs'].items():
            doses.append(dict(candidate_id=row['candidate_id'],head=row['head'],training_families=row['training_families'],dose_condition=dose,
                rmse=v.get('rms'),ci_low=v.get('ci95',[None,None])[0],ci_high=v.get('ci95',[None,None])[1],upper95=v.get('upper95'),
                direction_mean=v.get('direction',{}).get('mean'),direction_lower95=v.get('direction',{}).get('lower95'),
                magnitude_rmse=v.get('magnitude_error',{}).get('rms'),resolution_status=v.get('resolution_status'),
                non_target_status=v.get('non_target_status')))
    write_csv(results/'readout_comparison.csv',absolute);write_csv(results/'dose_error_comparison.csv',doses)
    epoch_rows=[dict(candidate_id=c['candidate_id'],head=c['head'],training_families=c['count'],seed=c['seed'],**e) for c in curves for e in c['epochs']]
    write_csv(results/'epoch_learning_curves.csv',epoch_rows)
    heads=('agent_location','block_location','angle_scalar','angle_sincos')
    fig,axes=plt.subplots(1,4,figsize=(14,3.2))
    for ax,head in zip(axes,heads):
        for method,label in (('ridge','Ridge'),('mlp256','MLP')):
            for count in (16,32,48):
                selected=[r for r in rows if r['head']==head and r['method']==method and r['training_families']==count and r['selected']]
                for r in selected:
                    a=r['metrics']['absolute'];ax.scatter(count,a['rms'],marker='o' if method=='ridge' else 'x',color='C0' if method=='ridge' else 'C1',label=label if count==16 else None)
                    ax.plot([count,count],a['ci95'],color='C0' if method=='ridge' else 'C1',alpha=.35)
                    ax.scatter(count,r['train_absolute']['rms'],marker='.',color='C0' if method=='ridge' else 'C1',alpha=.35)
        ax.axhline(.1 if head.startswith('angle') else 5.,ls='--',color='red',lw=1)
        ax.set(title=head,xlabel='Independent training families',ylabel='Wrapped rad' if head.startswith('angle') else 'Euclidean world px')
    axes[0].legend(fontsize=7);fig.suptitle('R1 learning with families: validation marks/95% intervals; faint dots training')
    fig.tight_layout();fig.savefig(results/'family_learning_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.4))
    for ax,head,kind,xvalues in zip(axes,('agent_location','block_location','angle_sincos'),('agent','block','angle'),((10,20,30),(1,2,5,10),(.01,.025,.05,.1))):
        for method in ('ridge','mlp256'):
            for r in [r for r in rows if r['head']==head and r['method']==method and r['training_families']==48 and r['selected']]:
                values=[r['metrics']['pairs'][f'{kind}_{x:g}'] for x in xvalues]
                ax.plot(xvalues,[v['rms'] for v in values],marker='o',label=r['candidate_id'].replace('n48_','').replace('_'+head,''))
                ax.fill_between(xvalues,[v['ci95'][0] for v in values],[v['ci95'][1] for v in values],alpha=.12)
        ax.plot(xvalues,[min(.1 if kind=='angle' else 5.,x/2) for x in xvalues],'k--',label='proposed half-effect/cap')
        ax.set(title=head,xlabel='True physical dose (rad)' if kind=='angle' else 'True physical dose (world px)',ylabel='Differential error (rad)' if kind=='angle' else 'Differential error (world px)')
    axes[0].legend(fontsize=7);fig.suptitle('R1 natural differential resolution; pilot-selected, not confirmatory')
    fig.tight_layout();fig.savefig(results/'dose_resolution.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(14,3.2))
    for ax,head in zip(axes,heads):
        for c in [c for c in curves if c['head']==head and c['count']==48]:
            ax.plot([e['epoch'] for e in c['epochs']],[e['train_physical_rmse'] for e in c['epochs']],ls='--',label=f"train seed{c['seed']}")
            ax.plot([e['epoch'] for e in c['epochs']],[e['validation_physical_rmse'] for e in c['epochs']],label=f"pilot seed{c['seed']}")
        ax.set(title=head,xlabel='Epoch',ylabel='Physical RMSE')
    axes[0].legend(fontsize=6);fig.suptitle('R1 training versus pilot validation; one fixed MLP schedule')
    fig.tight_layout();fig.savefig(results/'epoch_learning_curves.png',dpi=160);plt.close(fig)
