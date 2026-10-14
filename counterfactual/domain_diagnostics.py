"""Bounded natural-scene measurement diagnostics; never a Step 3/H2 unlock.

Reuse frozen contracts and audited fresh replay. No parameter fitting, latent
patching, goal changes, or replacement renderer is implemented here.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import h5py
import hdf5plugin  # noqa: F401
import imageio_ffmpeg
import numpy as np
from omegaconf import OmegaConf
import torch

from counterfactual.baseline import (configure_determinism, file_metadata,
    git_metadata, package_versions, sha256_file, source_metadata, write_json,
    write_source_snapshot)
from counterfactual.contracts import TimeActionContract
from counterfactual.model import checkpoint_path, load_frozen_lewm, validate_model_contract
from counterfactual.observation_compatibility import (codec_roundtrip, encode_images,
    readout_latents, spatial_regions, spatial_residual)
from counterfactual.render_provenance import color_masks, run_regressions, validate_simulator_parent
from counterfactual.simulator import (SceneSpec, SimulatorSettings, SimulatorValidationError,
    canonical_json, compare_replays, jsonable, replay_fixed_actions, wrap_angle)
from counterfactual.simulator_validation import validate_baseline_parent, validation_cases, write_replay
from probe.evaluate import load_probe_checkpoint, probe_checkpoint_path
from probe.rollout_ablation import open_loop_rollout_from_embeddings

DOSES = (0., 10., 20., 30.)
H = np.arange(0, 36, 5)
TARGETS = {'agent_location': slice(0, 2), 'block_location': slice(2, 4), 'block_angle': slice(4, 5)}


def reserved_families(section):
    """Outcome-independent poses; all related doses/trajectories share identity."""
    result = []
    for partition, seed in (("development", section.development_seed), ("validation", section.validation_seed)):
        rng = np.random.default_rng(int(seed))
        for index in range(int(section.families_per_partition)):
            block = (float(rng.uniform(220, 290)), float(rng.uniform(205, 290)))
            angle = float(rng.uniform(0, 2 * np.pi))
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            stratum = "free_space" if index % 2 == 0 else "near_contact"
            local = np.array([-150., -60.] if stratum == "free_space" else [-107., 15.])
            agent = np.array(block) + rotation @ local
            result.append(dict(family_id=f"domain_v2_{partition}_{index:02d}", partition=partition,
                stratum=stratum, seed=int(seed) + index, block_position=list(block), block_angle=angle,
                agent_position=agent.tolist(), direction=rotation[:, 0].tolist()))
    return result


def family_scenes(family):
    return [SceneSpec(f"{family['family_id']}_dose{int(dose)}", int(family['seed']),
        tuple(np.array(family['agent_position']) + dose * np.array(family['direction'])),
        tuple(family['block_position']), family['block_angle']) for dose in DOSES]


def near_duplicate(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.linalg.norm(a[:2]-b[:2]) <= 1. and np.linalg.norm(a[2:4]-b[2:4]) <= 1.
                and abs(wrap_angle(a[4]-b[4])) <= .01)


def validate_reservation(families, historical):
    poses = [(f['family_id'], s.public_state()) for f in families for s in family_scenes(f)]
    for index, (owner, pose) in enumerate(poses):
        if any(near_duplicate(pose, old) for old in historical):
            raise SimulatorValidationError(f"Diagnostic pose overlaps exposed/source/fixture family: {owner}")
        for other, previous in poses[:index]:
            if other != owner and near_duplicate(pose, previous):
                raise SimulatorValidationError(f"Diagnostic families overlap: {owner}, {other}")


def fixed_actions(family):
    actions = np.zeros((35, 2), np.float32)
    actions[10:25] = .10 * np.array(family['direction'], np.float32)
    return actions


def physical_error(pred, truth, target):
    difference = np.asarray(pred, np.float64) - np.asarray(truth, np.float64)
    return np.abs(wrap_angle(difference[..., 0])) if target == 'block_angle' else np.linalg.norm(difference, axis=-1)


def effect_vectors(readouts, truth, target):
    """Natural dose-minus-zero contrasts; no intervention or donor latent."""
    predicted = readouts[1:] - readouts[:1]
    real = truth[1:] - truth[:1]
    if target == 'block_angle':
        predicted, real = wrap_angle(predicted), wrap_angle(real)
    return predicted, real, physical_error(predicted, real, target)


def rms_bounds(squared, seed=2026101403, draws=10000):
    values = np.asarray(squared, np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Family squared values must be finite nonnegative and nonempty")
    sample = np.random.default_rng(seed).integers(0, len(values), (draws, len(values)))
    distribution = np.sqrt(values[sample].mean(axis=1))
    return dict(n_families=len(values), rms=float(np.sqrt(values.mean())),
        ci95=np.quantile(distribution, [.025, .975]).tolist(),
        lower95=float(np.quantile(distribution, .05)), upper95=float(np.quantile(distribution, .95)))


def bound_decision(bounds, margin):
    if bounds['upper95'] <= margin:
        return 'PASS'
    return 'FAIL' if bounds['lower95'] > margin else 'INCONCLUSIVE'


def predict_natural(model, natural_z, raw_actions, normalizer):
    """Correct shared seven-token path, with independent natural TF contexts."""
    z = torch.as_tensor(natural_z).float()
    if z.ndim != 3 or z.shape[1:] != (8, 192):
        raise ValueError('Natural trajectory must contain eight retained 192-D latents')
    raw = torch.as_tensor(raw_actions)
    if raw.shape != (len(z), 35, 2) or not torch.isfinite(raw).all():
        raise ValueError('Expected aligned (B,35,2) raw actions')
    tokens = torch.stack([normalizer.tokenize(item, 5) for item in raw])
    with torch.inference_mode():
        actions = model.action_encoder(tokens)
        encoded = dict(context_emb=z[:, :3], action_emb=actions, action_tokens=tokens,
            target_states=torch.zeros((len(z), 5, 7)), action_ends=torch.tensor([15,20,25,30,35]))
        open_loop = open_loop_rollout_from_embeddings(encoded, model, 'cpu', batch_size=16, history_size=3)
        teacher = torch.cat([model.predict(z[:, k:k+3], actions[:, k:k+3])[:, -1:] for k in range(5)], dim=1)
    if not torch.equal(open_loop[:, 0], teacher[:, 0]):
        raise SimulatorValidationError('Teacher/open-loop first target mismatch')
    if not all(torch.isfinite(item).all() for item in (teacher, open_loop, actions, tokens)):
        raise SimulatorValidationError('Nonfinite natural prediction')
    return dict(open_loop=open_loop.numpy(), teacher_forced=teacher.numpy(), standardized_tokens=tokens.numpy())


def trajectory_metrics(z, truth, actions, model, loaded, normalizer):
    prediction = predict_natural(model, z, actions, normalizer)
    reads = {key: value.reshape(len(z), 8, -1) for key, value in readout_latents(z.reshape(-1,192), loaded).items()}
    metrics = {}
    for target, selection in TARGETS.items():
        metrics[f'encoded_absolute_{target}'] = physical_error(reads[target][:,3:], truth[:,3:,selection], target).tolist()
    for mode in ('open_loop', 'teacher_forced'):
        forecast = prediction[mode]
        decoded = {key: value.reshape(len(z),5,-1) for key,value in readout_latents(forecast.reshape(-1,192), loaded).items()}
        prediction[f'{mode}_readouts'] = decoded
        metrics[f'{mode}_latent_error'] = np.linalg.norm(forecast.astype(float)-z[:,3:],axis=-1).tolist()
        for target, selection in TARGETS.items():
            metrics[f'{mode}_absolute_{target}'] = physical_error(decoded[target],truth[:,3:,selection],target).tolist()
            metrics[f'{mode}_vs_encoded_{target}'] = physical_error(decoded[target],reads[target][:,3:],target).tolist()
            predicted_effect, real_effect, error = effect_vectors(decoded[target],truth[:,3:,selection],target)
            metrics[f'{mode}_effect_{target}'] = error.tolist()
            prediction[f'{mode}_effect_vectors_{target}'] = dict(predicted=predicted_effect, real=real_effect)
    true_transition = np.linalg.norm(z[:,3:].astype(float)-z[:,2:7],axis=-1)
    tf_transition = np.linalg.norm(prediction['teacher_forced'].astype(float)-z[:,2:7],axis=-1)
    metrics['teacher_transition_l2'] = tf_transition.tolist()
    metrics['true_transition_l2'] = true_transition.tolist()
    metrics['teacher_transition_ratio'] = [[float(a/b) if b > 0 else None for a,b in zip(p,t)] for p,t in zip(tf_transition,true_transition)]
    return prediction, metrics, reads


def signed_source_diagnosis(parent, output):
    """Exposed cases only: diagnose constant bias without modifying any readout."""
    records = json.loads(Path(parent['provenance_parent']['path']).with_name('sample_records.json').read_text())
    truth_by_row = {int(r['raw_row']): r['source_state'] for r in records}
    path = Path(parent['artifacts']['compatibility_arrays.h5']['path'])
    if sha256_file(path) != parent['artifacts']['compatibility_arrays.h5']['sha256']:
        raise SimulatorValidationError('Prior compatibility arrays changed')
    cases = json.loads(Path(parent['provenance_parent']['path']).parents[1].joinpath(parent['run_id'],'case_metrics.json').read_text())
    result = dict(scope='All prior 16+5 families are now exposed development; render-relevant matching only.', cases=[], targets={})
    with h5py.File(path,'r') as f:
        ids = json.loads(f.attrs['image_identities'])
        mapping = {(item['family_id'], item['condition']): i for i,item in enumerate(ids) if item['condition'] in ('hdf5_source','raw_reconstruction')}
        values = {target:f[f'readouts/{target}'][:] for target in TARGETS}
        for case in cases:
            left, right = (mapping[(case['family_id'], c)] for c in ('hdf5_source','raw_reconstruction'))
            record = dict(family_id=case['family_id'], former_partition=case['split'], raw_row=case['raw_row'], truth=truth_by_row[case['raw_row']],
                latent_l2=case['raw_reconstruction']['latent_l2'], spatial=spatial_residual(f['images'][left],f['images'][right],spatial_regions(f['images'][right])))
            record['readouts'] = {}
            for target, selection in TARGETS.items():
                true = np.array(record['truth'])[selection]
                s, r = values[target][left].astype(float), values[target][right].astype(float)
                delta = r-s
                if target == 'block_angle': delta=wrap_angle(delta)
                record['readouts'][target] = dict(source=s.tolist(),render=r.tolist(),signed_drift=delta.tolist(),
                    source_error=float(physical_error(s,true,target)),render_error=float(physical_error(r,true,target)))
            # Segmented centroid shifts check obvious agent-image displacement, not exact subpixel state recovery.
            centers = []
            for idx in (left,right):
                mask = color_masks(f['images'][idx])['agent']; y,x=np.nonzero(mask)
                centers.append([float(x.mean()),float(y.mean())] if len(x) else None)
            record['agent_color_centroids_image_px'] = centers
            record['agent_centroid_difference_world_px'] = float(np.linalg.norm(np.subtract(*centers))*512/224) if all(x is not None for x in centers) else None
            result['cases'].append(record)
    for target in TARGETS:
        drift = np.array([r['readouts'][target]['signed_drift'] for r in result['cases']])
        dev = np.array([r['former_partition']=='development' for r in result['cases']])
        bias = drift[dev].mean(axis=0)
        if target == 'block_angle': bias=np.arctan2(np.sin(drift[dev]).mean(axis=0),np.cos(drift[dev]).mean(axis=0))
        corrected = drift-bias
        if target == 'block_angle': corrected=wrap_angle(corrected)
        point = dict(development_bias_vector=bias.tolist(), all_exposed_rms=float(np.sqrt(np.mean(np.sum(drift**2,axis=1)))),
            residual_rms_after_DEVELOPMENT_constant_bias=float(np.sqrt(np.mean(np.sum(corrected**2,axis=1)))),
            bias_correction_applied_to_readout=False)
        pose=np.array([r['truth'][:5] for r in result['cases']])
        pose=np.c_[pose[:,:4],np.sin(pose[:,4]),np.cos(pose[:,4])]
        # Descriptive correlation only; no pose-dependent model selection/fitting.
        point['signed_drift_pose_correlations']=[[float(np.corrcoef(pose[:,j],drift[:,k])[0,1]) if np.std(pose[:,j])>0 and np.std(drift[:,k])>0 else None for j in range(6)] for k in range(drift.shape[1])]
        result['targets'][target]=point
    write_json(output,jsonable(result))
    return result


def collect(families, artifacts, results, model, loaded, normalizer, section, reference):
    settings=SimulatorSettings()
    pending=[]; images=[]; commands=[]; attempts=[]
    deadline=time.monotonic()+float(section.timeout_seconds)
    with h5py.File(artifacts/'natural_diagnostics.h5','x') as f:
        f.attrs['scope']='readout-calibration diagnostic only; NOT formal Step 3 or H2'
        f.create_dataset('reference_latents',data=reference)
        for family in families:
            print(f"Fresh diagnostic: {family['family_id']}",flush=True)
            traces=[]; indices=[]
            try:
                for scene in family_scenes(family):
                    trace=replay_fixed_actions(scene,fixed_actions(family),settings,deadline=deadline)
                    if not np.all(np.abs(trace['body'][:11]-trace['body'][0])<=settings.atol):
                        raise SimulatorValidationError('Zero-action history changed physical state')
                    if np.max(np.abs(trace['goal_state']-scene.goal_state))>0:
                        raise SimulatorValidationError('Goal changed during natural replay')
                    write_replay(f.create_group(scene.scene_id),trace,normalizer)
                    indices.append(list(range(len(images),len(images)+8)))
                    images.extend(trace['images'][H]); traces.append(trace)
                zero_repeat=replay_fixed_actions(family_scenes(family)[0],fixed_actions(family),settings,deadline=deadline)
                repeat=compare_replays(traces[0],zero_repeat,settings.atol)
                if not repeat['passed']: raise SimulatorValidationError('New family full replay mismatch')
                write_replay(f.create_group(f"{family['family_id']}_zero_repeat"),zero_repeat,normalizer)
                repeat_index=len(images); images.append(zero_repeat['images'][0])
                codecs={}
                for name,codec,pixfmt,crf in [('ffv1','ffv1','bgr0',None),('x264_420_crf23','libx264','yuv420p',23),('x264_444_crf23','libx264','yuv444p',23)]:
                    codecs[name]=len(images)
                    images.append(codec_roundtrip(traces[0]['images'][0],artifacts/f"{family['family_id']}_{name}.mkv",codec,pixfmt,crf,commands))
                pending.append(dict(family=family,indices=indices,repeat_index=repeat_index,codecs=codecs,repeat=repeat,
                    truth=np.stack([t['body'][H][:,[0,1,9,10,11]] for t in traces]),
                    actions=np.stack([t['actions'] for t in traces]),metadata=traces[0]['metadata'],
                    clearances=[t['initial_validation']['clearances'] for t in traces],
                    contacts=[t['typed_contact_points'].sum(axis=0).tolist() for t in traces]))
                attempts.append(dict(family_id=family['family_id'],status='valid'))
            except SimulatorValidationError as error:
                attempts.append(dict(family_id=family['family_id'],status='invalid',reason=str(error)))
            write_json(results/'attempts.json',attempts)
        if not pending: raise SimulatorValidationError('No valid diagnostic families')
        print(f"Frozen encoding: {len(images)} diagnostic images",flush=True)
        z=encode_images(model,images)
        repeated=encode_images(model,images)
        if not np.array_equal(z,repeated): raise SimulatorValidationError('Repeated encoding mismatch')
        reads=readout_latents(z,loaded)
        f.create_dataset('encoded_images',data=np.stack(images),compression='gzip',compression_opts=1)
        f.create_dataset('encoded_latents',data=z)
        for target,value in reads.items(): f.create_dataset(f'readouts/{target}',data=value)
        records=[]
        for item in pending:
            family=item['family']; idx=np.array(item['indices']); natural=z[idx]
            prediction,metrics,natural_reads=trajectory_metrics(natural,item['truth'],item['actions'],model,loaded,normalizer)
            again=predict_natural(model,natural,item['actions'],normalizer)
            if any(not np.array_equal(prediction[k],again[k]) for k in again): raise SimulatorValidationError('Repeated prediction mismatch')
            group=f.create_group(f"{family['family_id']}_model")
            group.create_dataset('natural_latents',data=natural);group.create_dataset('physical_truth',data=item['truth'])
            for key,value in prediction.items():
                if isinstance(value,dict):
                    for sub,v in value.items(): group.create_dataset(f'{key}/{sub}',data=v)
                else: group.create_dataset(key,data=value)
            record=dict(**family,repeat=item['repeat'],clearances=item['clearances'],contacts=item['contacts'],
                simulator_contract=item['metadata'],trajectory_metrics=metrics,doses=[],nuisance={})
            for d,dose in enumerate(DOSES):
                base,other=idx[0,0],idx[d,0]
                decoded_change=reads['agent_location'][other].astype(float)-reads['agent_location'][base].astype(float)
                true_change=item['truth'][d,0,:2]-item['truth'][0,0,:2]
                magnitude=np.linalg.norm(decoded_change)
                row=dict(dose=dose,truth=item['truth'][d,0].tolist(), readouts={}, absolute={},differential={},
                    decoded_agent_change=decoded_change.tolist(),true_agent_change=true_change.tolist(),
                    direction_cosine=float(np.dot(decoded_change,true_change)/(magnitude*dose)) if magnitude>0 and dose>0 else None,
                    magnitude_error_world_px=float(abs(magnitude-dose)),latent_l2=float(np.linalg.norm(z[other].astype(float)-z[base])),
                    latent_norm=float(np.linalg.norm(z[other].astype(float))),nearest_source_reference=float(np.linalg.norm(reference.astype(float)-z[other],axis=1).min()))
                for target,selection in TARGETS.items():
                    value=reads[target][other].astype(float); basevalue=reads[target][base].astype(float)
                    row['readouts'][target]=value.tolist()
                    row['absolute'][target]=float(physical_error(value,item['truth'][d,0,selection],target))
                    pd,gt=value-basevalue,item['truth'][d,0,selection]-item['truth'][0,0,selection]
                    if target=='block_angle': pd,gt=wrap_angle(pd),wrap_angle(gt)
                    row['differential'][target]=float(physical_error(pd,gt,target))
                record['doses'].append(row)
            for label,index in dict(zero_repeat=item['repeat_index'],**item['codecs']).items():
                base=idx[0,0]
                row=dict(latent_l2=float(np.linalg.norm(z[index].astype(float)-z[base])))
                row.update({target:float(physical_error(reads[target][index],reads[target][base],target)) for target in TARGETS})
                row['latent_to_dose_ratio']={str(int(d)):row['latent_l2']/record['doses'][j]['latent_l2'] if record['doses'][j]['latent_l2']>0 else None for j,d in enumerate(DOSES) if d>0}
                record['nuisance'][label]=row
            records.append(record)
        f.attrs['family_ids']=canonical_json([r['family_id'] for r in records])
        f.attrs['retained_h']=H
    write_json(artifacts/'codec_commands.json',commands)
    write_json(results/'family_metrics.json',jsonable(records))
    return records,attempts


def aggregate(records,section):
    summary={}
    for target in TARGETS:
        for kind in ('absolute','differential'):
            for dose in (DOSES if kind == 'absolute' else DOSES[1:]):
                values=[next(d[kind][target]**2 for d in r['doses'] if d['dose']==dose) for r in records]
                summary[f'{kind}_{target}_dose{int(dose)}']=rms_bounds(values,int(section.bootstrap_seed),int(section.bootstrap_draws))
    for mode in ('encoded','open_loop','teacher_forced'):
        for target in TARGETS:
            key=f'{mode}_absolute_{target}'
            for k,h in enumerate(H[3:]):
                summary[f'{key}_h{h}']=rms_bounds([np.mean(np.square(np.array(r['trajectory_metrics'][key])[:,k])) for r in records])
    for mode in ('open_loop','teacher_forced'):
        for target in TARGETS:
            key=f'{mode}_effect_{target}'
            for k,h in enumerate(H[3:]):
                summary[f'{key}_h{h}']=rms_bounds([np.mean(np.square(np.array(r['trajectory_metrics'][key])[:,k])) for r in records])
    return summary


def calibration_from(summary):
    return {k:dict(margin=v['upper95'],cap=.10 if 'block_angle' in k else 5.,development=v)
            for k,v in summary.items() if k.startswith(('absolute_','differential_'))}


def gate_measurement(summary,calibration):
    checks={}
    for key,cal in calibration.items():
        bounds=summary[key]
        checks[key]=dict(cap=cal['cap'],development_margin=cal['margin'],bounds=bounds,
            cap_status=bound_decision(bounds,cal['cap']),calibrated_status=bound_decision(bounds,cal['margin']),
            development_within_cap=cal['development']['upper95']<=cal['cap'])
    statuses=[v[x] for v in checks.values() for x in ('cap_status','calibrated_status')]
    status='FAIL' if 'FAIL' in statuses or any(v['development']['lower95']>v['cap'] for v in calibration.values()) else 'PASS' if all(x=='PASS' for x in statuses) and all(v['development_within_cap'] for v in checks.values()) else 'INCONCLUSIVE'
    return dict(status=status,checks=checks)


def source_predictor_reference(parent,cache,artifacts,results,model,loaded,normalizer):
    cases=json.loads(Path(parent['provenance_parent']['path']).parents[1].joinpath(parent['run_id'],'case_metrics.json').read_text())
    images=[];states=[];actions=[];owners=[];missing=[]
    with h5py.File(cache/'pusht_expert_train.h5','r') as source:
        offsets=source['ep_offset'][:]
        for case in cases:
            row=case['raw_row'];episode=int(np.searchsorted(offsets,row,side='right')-1)
            if int(source['ep_len'][episode])<36:
                missing.append(dict(family_id=case['family_id'],reason='First source episode shorter than 36 frames'));continue
            images.extend(source['pixels'][row:row+36][H])
            states.append(source['state'][row:row+36][H,:5]);actions.append(source['action'][row:row+35]);owners.append(case['family_id'])
    if not owners: return dict(status='missing',exclusions=missing)
    z=encode_images(model,images).reshape(len(owners),8,192)
    # Do not form effects between unrelated source families.
    result=[]
    with h5py.File(artifacts/'source_predictor_reference.h5','x') as f:
        for i,owner in enumerate(owners):
            predicted,metrics,_=trajectory_metrics(z[i:i+1],np.array(states[i:i+1]),np.array(actions[i:i+1]),model,loaded,normalizer)
            result.append(dict(family_id=owner,trajectory_metrics=metrics))
            group=f.create_group(owner);group.create_dataset('natural_latents',data=z[i]);group.create_dataset('truth',data=states[i]);group.create_dataset('raw_actions',data=actions[i])
            for key in ('open_loop','teacher_forced','standardized_tokens'): group.create_dataset(key,data=predicted[key])
    write_json(results/'source_predictor_reference.json',result)
    summary={}
    for mode in ('encoded','open_loop','teacher_forced'):
        for target in TARGETS:
            key=f'{mode}_absolute_{target}'
            summary[key]=[rms_bounds([np.square(r['trajectory_metrics'][key][0][k]) for r in result]) for k in range(5)]
    return dict(n_families=len(owners),exclusions=missing,summary=summary,
        attribution='INCONCLUSIVE: source and fresh domains differ in physical states, goals and actions; no matched domain-only trajectory experiment')


def run_domain_stage(args,repo_root):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}',args.run_id): raise ValueError('Invalid run identifier')
    if args.device not in (None,'cpu') or args.skip_cem or args.rebuild_mismatched_caches or args.source_evidence:
        raise ValueError('Diagnostic requires frozen CPU inference; no baseline/cache changes')
    cfg=OmegaConf.load(repo_root/args.config);section=cfg.simulator.domain_diagnostics
    if args.partition not in ('development','validation'): raise ValueError('Explicit diagnostic partition required')
    results=repo_root/cfg.experiment.results_dir/'phase1'/args.run_id
    artifacts=repo_root/cfg.experiment.cache_dir/'phase1'/args.run_id
    for p in (results,artifacts):
        if p.exists(): raise FileExistsError(f'Refusing to overwrite {p}')
    results.mkdir(parents=True);artifacts.mkdir(parents=True)
    manifest_path=results/'domain_manifest.json'
    manifest=dict(stage='phase1_step2_natural_domain_diagnostic',schema_version=1,run_id=args.run_id,partition=args.partition,
        run_status='running',created_utc=datetime.now(timezone.utc).isoformat(),git=git_metadata(repo_root),command=[sys.executable,*sys.argv],
        config=OmegaConf.to_container(cfg,resolve=True),exact_historical_provenance='UNRESOLVED',encoder_dataset_ready=False,
        confirmatory_contract_frozen=False,formal_step3_collected=False,readouts_refitted=False,latent_patching_performed=False)
    write_json(manifest_path,manifest)
    try:
        manifest['baseline_parent'],normalizer=validate_baseline_parent(repo_root/cfg.simulator.baseline_manifest,repo_root,cfg)
        manifest['action_normalizer']=normalizer.as_dict()
        manifest['contract']=TimeActionContract().as_dict()
        manifest['device']='cpu';manifest['precision']='float32 model inference; float64 physical/statistical comparisons'
        old_path=repo_root/section.compatibility_manifest;old=json.loads(old_path.read_text())
        if old['run_status']!='completed': raise SimulatorValidationError('Completed compatibility parent required')
        manifest['compatibility_parent']=file_metadata(old_path,include_hash=True)
        manifest['ffmpeg']=file_metadata(imageio_ffmpeg.get_ffmpeg_exe(),include_hash=True)
        manifest['simulator_parent']=validate_simulator_parent(Path(old['simulator_parent']['path']),repo_root)
        hashes=source_metadata(repo_root)
        for path in (repo_root/args.config,repo_root/'observation_domain_protocol_v2.md',repo_root/'test_counterfactual_domain_diagnostics.py',repo_root/'probe/evaluate.py',repo_root/'probe/models.py'):
            hashes[str(path.resolve())]=sha256_file(path)
        manifest['source_sha256']=hashes
        manifest['versions']=package_versions();manifest['deterministic_settings']=configure_determinism(int(cfg.experiment.seed))
        families=reserved_families(section)
        historical=json.loads(Path(old['provenance_parent']['path']).with_name('initial_families.json').read_text())
        validate_reservation(families,[f['initial_state'] for f in historical]+[s.public_state() for s,_,_ in validation_cases(20261014)])
        write_json(results/'reserved_families.json',families)
        write_source_snapshot(artifacts/'source_snapshot.h5',hashes)
        development=None
        if args.partition=='validation':
            if not args.calibration_manifest: raise ValueError('Validation requires completed development manifest')
            devpath=Path(args.calibration_manifest).resolve();development=json.loads(devpath.read_text())
            if development.get('run_status')!='completed' or development.get('partition')!='development' or development.get('stage')!=manifest['stage']:
                raise SimulatorValidationError('Invalid development calibration parent')
            if development['source_sha256']!=hashes or development['config']!=manifest['config'] or development['versions']!=manifest['versions']:
                raise SimulatorValidationError('Code/config/protocol/runtime changed after development freeze')
            if json.loads(devpath.with_name('reserved_families.json').read_text())!=families:
                raise SimulatorValidationError('Independent family reservation changed')
            manifest['calibration_parent']=file_metadata(devpath,include_hash=True)
            manifest['prevalidation_freeze']=dict(calibration=development['calibration'],calibration_parent=manifest['calibration_parent'],sources=hashes)
            write_json(results/'prevalidation_freeze.json',manifest['prevalidation_freeze'])
        cache=repo_root/cfg.experiment.cache_dir
        model=load_frozen_lewm(cfg.experiment.policy_name,cache,'cpu');manifest['model_contract']=validate_model_contract(model,TimeActionContract())
        loaded={};manifest['readouts']={}
        for target in TARGETS:
            path=probe_checkpoint_path(cache,target,'mlp');loaded[target]=load_probe_checkpoint(path,'cpu');loaded[target]['probe'].requires_grad_(False)
            manifest['readouts'][target]=file_metadata(path,include_hash=True)
            if development and manifest['readouts'][target]!=development['readouts'][target]: raise SimulatorValidationError('Readout identity changed after development')
        reference_path=Path(old['artifacts']['compatibility_arrays.h5']['path'])
        if sha256_file(reference_path)!=old['artifacts']['compatibility_arrays.h5']['sha256']: raise SimulatorValidationError('Reference artifact changed')
        with h5py.File(reference_path) as f:
            reference=f['natural_reference_latents'][:]
            manifest['reference']=dict(parent=file_metadata(reference_path,include_hash=True),n_rows=len(reference),owners=json.loads(f.attrs['natural_reference_families']))
        if args.partition=='development':
            manifest['signed_source_diagnosis']=signed_source_diagnosis(old,results/'signed_source_diagnosis.json')['targets']
            manifest['source_predictor_reference']=source_predictor_reference(old,cache,artifacts,results,model,loaded,normalizer)
        selected=[f for f in families if f['partition']==args.partition]
        records,attempts=collect(selected,artifacts,results,model,loaded,normalizer,section,reference)
        manifest['summary']=aggregate(records,section)
        manifest['coverage']=dict(requested=len(selected),valid=len(records),attempts=attempts)
        manifest['repeat_encoding_prediction_max_abs']=0.
        manifest['production_contract']=records[0]['simulator_contract']
        if args.partition=='development':
            manifest['calibration']=calibration_from(manifest['summary'])
            write_json(results/'development_calibration.json',manifest['calibration'])
        else:
            manifest['calibration']=development['calibration']
            measurement=gate_measurement(manifest['summary'],manifest['calibration'])
            complete=len(records)==len(selected)
            manifest['gates']=dict(A=dict(status='PASS',basis='unchanged validated mechanics and all new zero full replays exact'),
                B=dict(status='INCONCLUSIVE',explicit_generation='PASS',reproducible_generation='PASS',
                    basis='Operational generation is explicit and controlled, but scientific observation compatibility cannot be frozen without C/D evidence; historical provenance separate'),
                C=measurement,D=dict(status='INCONCLUSIVE',basis='natural predictor ceiling measured; legacy readout OOD limits and domain-only attribution unresolved'),
                E=dict(status='FAIL' if measurement['status']=='FAIL' else 'INCONCLUSIVE',independent_families=complete,
                    basis='predeclared independent validation performed, but relevant compatibility checks must also pass'))
            if not complete:
                manifest['gates']['C']['status']='INCONCLUSIVE';manifest['gates']['E']['status']='INCONCLUSIVE'
            manifest['readiness']=dict(step3_development_only=False,step3_confirmatory_observation_contract=False,
                permissible_scope='Only explicitly incompatible measurement/geometry diagnostics; no compatibility-qualified Step 3 or physical causal claims. Human approval required for refit.')
        manifest['regressions']=run_regressions(repo_root,artifacts)
        for name in ('test_counterfactual_observation_compatibility.py','test_counterfactual_domain_diagnostics.py'):
            command=[sys.executable,str(repo_root/name)];p=subprocess.run(command,cwd=repo_root,capture_output=True,timeout=300)
            for label,data in (('stdout',p.stdout),('stderr',p.stderr)): (artifacts/'regression_logs'/f'{name}.{label}.txt').write_bytes(data)
            manifest['regressions'].append(dict(command=command,return_code=p.returncode))
            if p.returncode: raise SimulatorValidationError(f'Regression failed: {name}')
        for path,digest in hashes.items():
            if sha256_file(path)!=digest: raise SimulatorValidationError(f'Source changed during run: {path}')
        for record in manifest['readouts'].values():
            if sha256_file(record['path'])!=record['sha256']: raise SimulatorValidationError('Readout changed')
        if sha256_file(manifest['ffmpeg']['path'])!=manifest['ffmpeg']['sha256']: raise SimulatorValidationError('Codec binary changed')
        if sha256_file(checkpoint_path(cache,cfg.experiment.policy_name))!=cfg.baseline.checkpoint_sha256: raise SimulatorValidationError('Model changed')
        if file_metadata(cache/'pusht_expert_train.h5')!=manifest['baseline_parent']['dataset_metadata']: raise SimulatorValidationError('Dataset changed')
        manifest['artifacts']={str(p.relative_to(artifacts)):file_metadata(p,include_hash=True) for p in artifacts.rglob('*') if p.is_file()}
        manifest['run_status']='completed'
        write_json(manifest_path,jsonable(manifest))
        from counterfactual.domain_report import write_domain_report
        write_domain_report(results)
    except Exception as error:
        manifest['run_status']='failed';manifest['failure']=dict(type=type(error).__name__,message=str(error))
        (results/'failure_traceback.txt').write_text(traceback.format_exc());print(f'Diagnostic failed: {error}',file=sys.stderr)
    finally:
        manifest['finished_utc']=datetime.now(timezone.utc).isoformat();write_json(manifest_path,jsonable(manifest))
    print(f"Domain diagnostic {manifest['run_status']}; automatic readiness stays false; {manifest_path}",flush=True)
    return 0 if manifest['run_status']=='completed' else 1
