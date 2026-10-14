"""Post-fit descriptive strata/bias/reproducibility; no fitting/criteria changes."""
import json
from pathlib import Path
import sys
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from counterfactual.baseline import file_metadata,write_json
from counterfactual.readout_feasibility import absolute_metrics,paired_rms_gain
from counterfactual.readout_audit import absolute_summary

HERE=Path(__file__).resolve().parent;REPLAY=HERE.parent/'readout-r1-feasibility-replay-v1'
R0=HERE.parent/'readout-r0-sanity-verified-v1'
m=json.loads((HERE/'repair_manifest.json').read_text());rows=json.loads((HERE/'readout_comparison.json').read_text())
lookup={r['candidate_id']:r for r in rows}
selected=[r for r in rows if r['training_families']==48 and r['selected']]
extra=dict(scope='Post-assessment descriptive summaries only; no selected model, margin or readiness changed.',strata={},latest_learning_gain={},render_signal_controls={})
for head in ('agent_location','block_location','angle_scalar','angle_sincos'):
    for method in ('ridge','mlp_seed_mean'):
        small=[lookup[m['selected_ridge_candidates'][f'n32_ridge_{head}']]] if method=='ridge' else [lookup[f'n32_mlp_{head}_seed{s}'] for s in (42,43)]
        large=[lookup[m['selected_ridge_candidates'][f'n48_ridge_{head}']]] if method=='ridge' else [lookup[f'n48_mlp_{head}_seed{s}'] for s in (42,43)]
        a=np.mean([np.sum(r['metrics']['absolute']['family_mse'],axis=1) for r in small],axis=0)
        b=np.mean([np.sum(r['metrics']['absolute']['family_mse'],axis=1) for r in large],axis=0)
        extra['latest_learning_gain'][f'{head}_{method}_32_to_48']=paired_rms_gain(a,b)
array=Path(m['dataset']['natural_array']['path']);pred_path=Path(m['artifacts']['candidate_predictions.h5']['path'])
with h5py.File(array) as f,h5py.File(pred_path) as p:
    ids=json.loads(f.attrs['identities']);owners=np.array([i['family_id'] for i in ids]);labels=f['physical_labels'][:]
    z=f['latents'][:];images=f['images']
    for stratum in ('free_space','near_contact'):
        mask=np.array([i['role']=='pilot_validation' and i['stratum']==stratum and i['kind']!='zero' for i in ids])
        extra['strata'][stratum]={r['candidate_id']:absolute_metrics(p[r['candidate_id']][:][mask],labels[mask],owners[mask],r['head']) for r in selected}
    for kind in ('agent','block','angle'):
        for dose in sorted(set(i['dose'] for i in ids if i['kind']==kind)):
            selected_indices=[n for n,i in enumerate(ids) if i['role']=='pilot_validation' and i['kind']==kind and i['dose']==dose]
            base={owner:next(n for n,i in enumerate(ids) if i['family_id']==owner and i['kind']=='base') for owner in set(owners)}
            identical=[np.array_equal(images[n],images[base[owners[n]]]) for n in selected_indices]
            distances=[float(np.linalg.norm(z[n].astype(float)-z[base[owners[n]]])) for n in selected_indices]
            extra['render_signal_controls'][f'{kind}_{dose:g}']=dict(n_repeated_measurements=len(selected_indices),n_independent_families=16,
                exactly_identical_image_count=int(sum(identical)),zero_latent_signal_count=int(sum(d==0 for d in distances)),median_latent_l2=float(np.median(distances)))
    # Keep arithmetic signed wrapped-error means distinct from circular means.
    extra['circular_angle_bias']={}
    for r in selected:
        if not r['head'].startswith('angle'):continue
        mask=np.array([i['role']=='pilot_validation' and i['kind']!='zero' for i in ids])
        difference=p[r['candidate_id']][:][mask,0]-labels[mask,4]
        extra['circular_angle_bias'][r['candidate_id']]=float(np.arctan2(np.sin(difference).mean(),np.cos(difference).mean()))

fig,axes=plt.subplots(1,2,figsize=(10,3.4))
for ax,target in zip(axes,('agent_location','block_location')):
    for r in [r for r in selected if r['head']==target]:
        keys=['agent_10','agent_20','agent_30'] if target=='block_location' else ['block_1','block_2','block_5','block_10']
        values=[r['metrics']['pairs'][k] for k in keys]
        ax.plot(range(len(keys)),[v['rms'] for v in values],marker='o',label=r['candidate_id'].replace('n48_',''))
        ax.fill_between(range(len(keys)),[v['ci95'][0] for v in values],[v['ci95'][1] for v in values],alpha=.1)
    ax.set_xticks(range(len(keys)),keys);ax.axhline(5.,ls='--',color='red');ax.set(ylabel='Spurious readout change (world px)',title=f'Non-target {target}')
axes[0].legend(fontsize=6);fig.suptitle('Natural non-target preservation, not latent patch selectivity')
fig.tight_layout();fig.savefig(HERE/'non_target_preservation.png',dpi=160);plt.close(fig)

audit=json.loads((R0/'audit_measurements.json').read_text())
source_rows=[r for r in audit if r['domain']=='prospective']
owners=np.array([r['family_id'] for r in source_rows]);labels=np.array([r['truth'] for r in source_rows])
pred={target:np.array([r['predictions'][target] for r in source_rows]) for target in ('agent_location','block_location','block_angle')}
positions={}
for axis in (0,1):
    for high in (False,True):
        mask=(labels[:,axis]>=256)==high
        if mask.any():positions[f'agent_{"x" if axis==0 else "y"}_{"high" if high else "low"}']=absolute_summary({k:v[mask] for k,v in pred.items()},labels[mask],owners[mask])
unique=sorted(set(owners));family_truth=np.array([labels[owners==o].mean(axis=0) for o in unique])
error=np.array([np.mean(pred['agent_location'][owners==o]-labels[owners==o,:2],axis=0) for o in unique])
positions['signed_agent_error_vs_agent_xy_correlation']=[[float(np.corrcoef(error[:,k],family_truth[:,j])[0,1]) for j in (0,1)] for k in (0,1)]
write_json(R0/'position_concentration_supplement.json',positions)

other=json.loads((REPLAY/'repair_manifest.json').read_text());datasets=[];attrs=[]
for name,path,right_path in [('inputs',array,Path(other['dataset']['natural_array']['path'])),('predictions',pred_path,Path(other['artifacts']['candidate_predictions.h5']['path']))]:
    with h5py.File(path) as a,h5py.File(right_path) as b:
        an=[];bn=[];a.visit(an.append);b.visit(bn.append);assert an==bn
        def compare(key,item):
            assert set(item.attrs)==set(b[key].attrs)
            if isinstance(item,h5py.Dataset):
                assert item.dtype==b[key].dtype and item.shape==b[key].shape and np.array_equal(item[()],b[key][()],equal_nan=True),(name,key)
                datasets.append([name,key])
            for k in item.attrs:assert np.array_equal(item.attrs[k],b[key].attrs[k]);attrs.append([name,key,k])
        assert set(a.attrs)==set(b.attrs)
        for k in a.attrs:assert np.array_equal(a.attrs[k],b.attrs[k]);attrs.append([name,'/',k])
        a.visititems(compare)
for filename in ('family_assignments.json','attempts.json','learning_curves.json'):
    assert (HERE/filename).read_bytes()==(REPLAY/filename).read_bytes(),filename
model_count=0
for row in rows:
    if 'artifact' not in row:continue
    name=Path(row['artifact']['path']).name
    left=Path(row['artifact']['path']);right=Path(other['artifacts'][f'readouts/{name}']['path'])
    assert file_metadata(left,include_hash=True)['sha256']==file_metadata(right,include_hash=True)['sha256'],name
    model_count+=1
assert m['feasibility_assessment']==other['feasibility_assessment']
verification=dict(status='EXACT',dataset_count=len(datasets),attribute_count=len(attrs),model_artifacts_byte_identical=model_count,
    assignments_attempts_learning_curves_identical=True,feasibility_assessment_identical=True,
    repeated_pilot_validation_is_NOT_additional_independent_evidence=True,command=[sys.executable,*sys.argv],
    script=file_metadata(Path(__file__),include_hash=True),inputs={str(p):file_metadata(p,include_hash=True) for p in (array,pred_path)})
write_json(HERE/'reproducibility_checks.json',verification)
extra['reproducibility']=verification;write_json(HERE/'supplemental_summary.json',extra)
print(json.dumps(verification,indent=2))
