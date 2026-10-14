"""Post-assessment descriptive summaries; does not change any gate or margin.

Run from repo root with .venv/bin/python and this path. Uses saved numerical
outputs only; no simulator/model/readout invocation or selection is performed.
"""
import json
from pathlib import Path
import sys
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from counterfactual.baseline import file_metadata,write_json
from counterfactual.domain_diagnostics import TARGETS,effect_vectors,rms_bounds

HERE=Path(__file__).resolve().parent
DEV=HERE.parent/'step2-domain-calibration-dev-v2'
REPLAY=HERE.parent/'step2-domain-validation-replay-v2'
records=json.loads((HERE/'family_metrics.json').read_text())
manifest=json.loads((HERE/'domain_manifest.json').read_text())
development=json.loads((DEV/'domain_manifest.json').read_text())
old=json.loads((DEV/'signed_source_diagnosis.json').read_text())

def bounds(values): return rms_bounds(np.square(values))

summary=dict(scope='Post-assessment descriptive summaries only; no new acceptance rule.',
    inputs={str(p):file_metadata(p,include_hash=True) for p in [HERE/'family_metrics.json',HERE/'domain_manifest.json',DEV/'domain_manifest.json',DEV/'signed_source_diagnosis.json']},
    source_centroid_world_shift=bounds([r['agent_centroid_difference_world_px'] for r in old['cases']]),
    source_centroid_world_shift_max=max(r['agent_centroid_difference_world_px'] for r in old['cases']),
    doses={},nuisance={},strata={},natural_effects={},predictor_vs_encoded={},latent_forecast={})
for dose in (10,20,30):
    rows=[next(d for d in r['doses'] if d['dose']==dose) for r in records]
    summary['doses'][dose]=dict(latent_signal=bounds([d['latent_l2'] for d in rows]),
        latent_signal_median=float(np.median([d['latent_l2'] for d in rows])),
        positive_direction_count=sum(d['direction_cosine']>0 for d in rows),
        direction_cosine_median=float(np.median([d['direction_cosine'] for d in rows])),
        decoded_magnitude_median=float(np.median([np.linalg.norm(d['decoded_agent_change']) for d in rows])),
        magnitude_error=bounds([d['magnitude_error_world_px'] for d in rows]))
for condition in records[0]['nuisance']:
    summary['nuisance'][condition]={key:bounds([r['nuisance'][condition][key] for r in records])
        for key in ('latent_l2',*TARGETS)}
for stratum in ('free_space','near_contact'):
    subset=[r for r in records if r['stratum']==stratum]
    summary['strata'][stratum]={f'{kind}_{target}_dose{dose}':bounds([next(d[kind][target] for d in r['doses'] if d['dose']==dose) for r in subset])
        for kind in ('absolute','differential') for target in TARGETS for dose in (10,20,30)}
for mode in ('teacher_forced','open_loop'):
    for target in TARGETS:
        key=f'{mode}_vs_encoded_{target}'
        summary['predictor_vs_encoded'][key]=[rms_bounds([np.square(np.array(r['trajectory_metrics'][key])[:,k]).mean() for r in records]) for k in range(5)]
    key=f'{mode}_latent_error'
    summary['latent_forecast'][mode]=[rms_bounds([np.square(np.array(r['trajectory_metrics'][key])[:,k]).mean() for r in records]) for k in range(5)]
ratios=[value for r in records for curve in r['trajectory_metrics']['teacher_transition_ratio'] for value in curve[:3] if value is not None]
summary['teacher_transition_active_h15_25']=dict(n_values=len(ratios),median=float(np.median(ratios)),
    scope='Repeated family/dose/horizon values, descriptive median only; not independent N.')
summary['source_predictor_reference']=development['source_predictor_reference']
array_path=Path(manifest['artifacts']['natural_diagnostics.h5']['path'])
with h5py.File(array_path) as f:
    source=f['reference_latents'][:]
    summary['norm_reference']=dict(source_bank_median=float(np.median(np.linalg.norm(source.astype(float),axis=1))),
        prospective_initial_median=float(np.median([d['latent_norm'] for r in records for d in r['doses']])),
        source_bank_norm_range=[float(x) for x in np.quantile(np.linalg.norm(source.astype(float),axis=1),[0,1])],
        prospective_initial_range=[float(x) for x in np.quantile([d['latent_norm'] for r in records for d in r['doses']],[0,1])],
        nearest_bank_distance_median=float(np.median([d['nearest_source_reference'] for r in records for d in r['doses']])))
    owners=json.loads(f.attrs['family_ids'])
    assert len(f['encoded_latents'])==36*len(owners)
    per_target={t:[] for t in TARGETS}; per_encoded={t:[] for t in TARGETS}
    real_arrays={t:[] for t in TARGETS}
    for i,owner in enumerate(owners):
        group=f[f'{owner}_model']
        truth=group['physical_truth'][:]
        assert np.array_equal(f['encoded_latents'][i*36:i*36+32].reshape(4,8,192),group['natural_latents'][:])
        for target,selection in TARGETS.items():
            decoded=f[f'readouts/{target}'][i*36:i*36+32].reshape(4,8,-1)
            _,real,error=effect_vectors(decoded[:,3:],truth[:,3:,selection],target)
            magnitude=np.abs(real[...,0]) if target=='block_angle' else np.linalg.norm(real,axis=-1)
            per_target[target].append(np.mean(magnitude**2,axis=0))
            per_encoded[target].append(np.mean(error**2,axis=0))
            real_arrays[target].append(magnitude.tolist())
    summary['natural_effects']={t:dict(real_rms=[rms_bounds(np.array(per_target[t])[:,k]) for k in range(5)],
        encoded_readout_effect_error=[rms_bounds(np.array(per_encoded[t])[:,k]) for k in range(5)],
        per_family_dose_horizon_real_magnitudes=real_arrays[t]) for t in TARGETS}

# Independent-process reproducibility of canonical arrays/attrs, not timestamps.
replay=json.loads((REPLAY/'domain_manifest.json').read_text())
other=Path(replay['artifacts']['natural_diagnostics.h5']['path'])
datasets=[];attributes=[]
with h5py.File(array_path) as left,h5py.File(other) as right:
    def compare(name,item):
        assert name in right
        assert set(item.attrs)==set(right[name].attrs)
        if isinstance(item,h5py.Dataset):
            a,b=item[()],right[name][()]
            assert item.dtype==right[name].dtype and item.shape==right[name].shape and np.array_equal(a,b),name
            datasets.append(name)
        for key in item.attrs:
            assert key in right[name].attrs and np.array_equal(item.attrs[key],right[name].attrs[key]),(name,key)
            attributes.append([name,key])
    for key in left.attrs:
        assert np.array_equal(left.attrs[key],right.attrs[key]);attributes.append(['/',key])
    left.visititems(compare)
    assert set(left.attrs)==set(right.attrs)
    lnames=[];rnames=[];left.visit(lnames.append);right.visit(rnames.append)
    assert lnames==rnames
comparison=dict(status='EXACT',datasets=len(datasets),attributes=len(attributes),
    inputs={str(p):file_metadata(p,include_hash=True) for p in (array_path,other)},
    family_metrics_byte_equal=(HERE/'family_metrics.json').read_bytes()==(REPLAY/'family_metrics.json').read_bytes(),
    freeze_byte_equal=(HERE/'prevalidation_freeze.json').read_bytes()==(REPLAY/'prevalidation_freeze.json').read_bytes(),
    independent_validation_repeat_is_NOT_additional_statistical_evidence=True)
assert comparison['family_metrics_byte_equal'] and comparison['freeze_byte_equal']
write_json(HERE/'reproducibility_checks.json',comparison)
summary['reproducibility']=comparison
write_json(HERE/'supplemental_summary.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ('source_predictor_reference','strata','natural_effects','inputs')},indent=2))
