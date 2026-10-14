"""R0 input/label audit, preserving all frozen experimental contracts."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
import csv
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback

import cv2
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pygame
import torch
from omegaconf import OmegaConf

from counterfactual.baseline import (configure_determinism, file_metadata,
    git_metadata, package_versions, sha256_file, source_metadata, write_json, write_source_snapshot)
from counterfactual.contracts import TimeActionContract
from counterfactual.domain_diagnostics import family_scenes, rms_bounds, TARGETS
from counterfactual.model import checkpoint_path, load_frozen_lewm, validate_model_contract
from counterfactual.observation_compatibility import encode_images, readout_latents
from counterfactual.render_provenance import (color_masks, preprocessing_parity,
    run_regressions, validate_simulator_parent)
from counterfactual.simulator import (SceneSpec, SimulatorSettings, SimulatorValidationError,
    body_vector, create_fresh_scene, jsonable, physical_readback, wrap_angle)
from counterfactual.simulator_validation import validate_baseline_parent
from probe.evaluate import load_probe_checkpoint, probe_checkpoint_path
from probe.targets import TARGET_BUILDERS


def repair_context(args, repo_root, stage):
    """Shared provenance setup; not an experimental-policy abstraction."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}',args.run_id):
        raise ValueError('Invalid immutable run identifier')
    if args.device not in (None,'cpu') or args.skip_cem or args.rebuild_mismatched_caches or args.source_evidence or args.partition:
        raise ValueError('Readout repair uses CPU and cannot mutate baseline/cache/diagnostic partitions')
    cfg=OmegaConf.load(repo_root/args.config); section=cfg.simulator.readout_repair
    results=repo_root/cfg.experiment.results_dir/'phase1'/args.run_id
    artifacts=repo_root/cfg.experiment.cache_dir/'phase1'/args.run_id
    for directory in (results,artifacts):
        if directory.exists(): raise FileExistsError(f'Refusing to overwrite {directory}')
    # Verify immutable prerequisites before starting a recorded run.
    baseline,normalizer=validate_baseline_parent(repo_root/cfg.simulator.baseline_manifest,repo_root,cfg)
    parents={name:json.loads((repo_root/section[f'{name}_manifest']).read_text()) for name in ('development','validation')}
    for name,parent in parents.items():
        if parent.get('stage')!='phase1_step2_natural_domain_diagnostic' or parent.get('run_status')!='completed':
            raise SimulatorValidationError('Completed natural-domain diagnostics required')
        for key in ('natural_diagnostics.h5',):
            record=parent['artifacts'][key]
            if sha256_file(record['path'])!=record['sha256']: raise SimulatorValidationError('Parent natural arrays changed')
    simulator=validate_simulator_parent(Path(parents['development']['simulator_parent']['path']),repo_root)
    results.mkdir(parents=True);artifacts.mkdir(parents=True)
    hashes=source_metadata(repo_root)
    from stable_worldmodel.envs.utils import DrawOptions
    draw_source=Path(inspect.getfile(DrawOptions)).resolve()
    hashes[str(draw_source)]=sha256_file(draw_source)
    for relative in (args.config,'readout_repair_protocol_r0_r1.md','probe/evaluate.py','probe/models.py','probe/train.py','probe/targets.py','probe/metrics.py'):
        path=(repo_root/relative).resolve();hashes[str(path)]=sha256_file(path)
    for path in repo_root.glob('test_counterfactual_readout_*.py'): hashes[str(path.resolve())]=sha256_file(path)
    manifest=dict(stage=stage,schema_version=1,run_id=args.run_id,run_status='running',
        created_utc=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv],git=git_metadata(repo_root),
        config=OmegaConf.to_container(cfg,resolve=True),source_sha256=hashes,versions=package_versions(),
        deterministic_settings=configure_determinism(42),baseline_parent=baseline,simulator_parent=simulator,
        device='cpu',precision='float32 encoder/MLP; float64 geometry/Ridge/statistics',
        action_normalizer=normalizer.as_dict(),contract=TimeActionContract().as_dict(),
        diagnostic_parents={name:file_metadata(repo_root/section[f'{name}_manifest'],include_hash=True) for name in parents},
        exact_historical_provenance='UNRESOLVED',r2_authorized=False,formal_step3_collected=False,
        official_instrument_promoted=False,h1_projector_replaced=False,latent_interventions=False)
    write_json(results/'repair_manifest.json',manifest)
    write_source_snapshot(artifacts/'source_snapshot.h5',hashes)
    cache=repo_root/cfg.experiment.cache_dir
    model=load_frozen_lewm(cfg.experiment.policy_name,cache,'cpu')
    manifest['model_contract']=validate_model_contract(model,TimeActionContract())
    loaded={};manifest['original_readouts']={}
    for target in TARGETS:
        path=probe_checkpoint_path(cache,target,'mlp')
        loaded[target]=load_probe_checkpoint(path,'cpu');loaded[target]['probe'].requires_grad_(False)
        manifest['original_readouts'][target]=file_metadata(path,include_hash=True)
    return cfg,section,results,artifacts,manifest,parents,model,loaded


def finish_repair(manifest,results,artifacts):
    manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
    if manifest['run_status']=='completed':
        for path,digest in manifest['source_sha256'].items():
            if sha256_file(path)!=digest: raise SimulatorValidationError(f'Source changed during run: {path}')
        for record in manifest['original_readouts'].values():
            if sha256_file(record['path'])!=record['sha256']: raise SimulatorValidationError('Original readout changed')
        cfg=manifest['config'];cache=Path(manifest['baseline_parent']['dataset_metadata']['path']).parent
        if sha256_file(checkpoint_path(cache,cfg['experiment']['policy_name']))!=cfg['baseline']['checkpoint_sha256']:
            raise SimulatorValidationError('Frozen LeWM checkpoint changed')
        if file_metadata(cache/'pusht_expert_train.h5')!=manifest['baseline_parent']['dataset_metadata']:
            raise SimulatorValidationError('Original source dataset changed')
        manifest['artifacts']={str(p.relative_to(artifacts)):file_metadata(p,include_hash=True) for p in artifacts.rglob('*') if p.is_file()}
    write_json(results/'repair_manifest.json',jsonable(manifest))


def repair_regressions(repo_root,artifacts,extra):
    records=run_regressions(repo_root,artifacts)
    for name in ('test_counterfactual_observation_compatibility.py','test_counterfactual_domain_diagnostics.py',*extra):
        command=[sys.executable,str(repo_root/name)]
        completed=subprocess.run(command,cwd=repo_root,capture_output=True,timeout=300)
        for label,data in (('stdout',completed.stdout),('stderr',completed.stderr)):
            (artifacts/'regression_logs'/f'{name}.{label}.txt').write_bytes(data)
        records.append(dict(command=command,return_code=completed.returncode))
        if completed.returncode: raise SimulatorValidationError(f'Regression failed: {name}')
    return records


def polygon_area_centroid(vertices):
    vertices=np.asarray(vertices,np.float64);following=np.roll(vertices,-1,axis=0)
    cross=vertices[:,0]*following[:,1]-following[:,0]*vertices[:,1]
    twice=float(cross.sum())
    if twice==0: raise ValueError('Degenerate polygon')
    return abs(twice)/2,((vertices+following)*cross[:,None]).sum(axis=0)/(3*twice)


def geometry_audit(env,image):
    """Independent rigid transform and analytic silhouette, not a learned probe."""
    raw=np.transpose(pygame.surfarray.array3d(env.screen),(1,0,2))
    if raw.shape!=(512,512,3) or not np.array_equal(cv2.resize(raw,(224,224)),image):
        raise SimulatorValidationError('Canvas RGB/transpose/resize correspondence failed')
    angle=float(env.block.angle)
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    position=np.asarray(env.block.position)
    expected=np.zeros((512,512),np.uint8);matrix_deltas=[];areas=[];centroids=[];world_vertices=[]
    for shape in env.block.shapes:
        local=np.array(shape.get_vertices())
        independent=local@rotation.T+position
        engine=np.array([env.block.local_to_world(tuple(v)) for v in local])
        matrix_deltas.append(float(np.abs(engine-independent).max()))
        world_vertices.append(independent.tolist())
        area,center=polygon_area_centroid(independent);areas.append(area);centroids.append(center)
        cv2.fillConvexPoly(expected,np.rint(independent).astype(np.int32),1)
    origin=np.asarray(env.block.local_to_world((0.,0.)))
    if max(matrix_deltas)>1e-10 or not np.allclose(origin,position,atol=1e-10,rtol=0):
        raise SimulatorValidationError('Body origin / local-to-world target definition mismatch')
    masks=color_masks(raw)
    expected_agent=np.zeros((512,512),np.uint8)
    cv2.circle(expected_agent,tuple(np.rint(env.agent.position).astype(int)),15,1,-1)
    raster={}
    for name,mask,band in (('agent',expected_agent,2),('block',expected,4)):
        kernel=np.ones((2*band+1,2*band+1),np.uint8)
        interior=cv2.erode(mask,kernel).astype(bool)
        exterior=~cv2.dilate(mask,kernel).astype(bool)
        missing=int((interior&~masks[name]).sum());outside=int((exterior&masks[name]).sum())
        y,x=np.nonzero(masks[name]);centroid=np.array([x.mean(),y.mean()])
        raster[name]=dict(missing_deep_interior=missing,outside_boundary_band=outside,
            boundary_band_world_px=band,mask_pixels=len(x),pixel_centroid_world=centroid.tolist())
        if missing or outside: raise SimulatorValidationError(f'Geometry outside raster-derived band: {name}, {missing}, {outside}')
    return dict(canvas_resize_exact=True,y_axis_down=True,rgb_order=True,world_size=512,render_size=224,
        analytic_transform_max_abs=max(matrix_deltas),block_body_origin=position.tolist(),
        block_world_mass_center=list(env.block.local_to_world(env.block.center_of_gravity)),
        block_world_area_center=np.average(centroids,axis=0,weights=areas).tolist(),
        block_local_mass_center=list(env.block.center_of_gravity),world_vertices=world_vertices,raster=raster)


def verify_original_readouts(z,loaded):
    results={};x=torch.from_numpy(z).float()
    for target,record in loaded.items():
        payload=record['payload'];probe=record['probe']
        dimension=1 if target=='block_angle' else 2
        if record['target_name']!=target or record['probe_type']!='mlp': raise SimulatorValidationError('Wrong original target/head')
        if record['target_mean'].shape!=(1,dimension) or record['target_std'].shape!=(1,dimension) or not torch.all(record['target_std']>0):
            raise SimulatorValidationError('Wrong target normalization shape/scale')
        if any(module.training for module in probe.modules()): raise SimulatorValidationError('Original readout dropout not disabled')
        layers=[layer for layer in probe.net if isinstance(layer,torch.nn.Linear)]
        if [(layer.in_features,layer.out_features) for layer in layers]!=[(192,256),(256,dimension)]:
            raise SimulatorValidationError('Original MLP differs from its actual 256-hidden-unit contract')
        for key,value in probe.state_dict().items():
            if not torch.equal(value,payload['probe_state_dict'][key]) or not torch.isfinite(value).all():
                raise SimulatorValidationError('Original checkpoint weights incorrectly loaded')
        with torch.inference_mode():
            manual=torch.nn.functional.linear(x,layers[0].weight,layers[0].bias).relu()
            manual=torch.nn.functional.linear(manual,layers[1].weight,layers[1].bias)
            actual=probe(x)
            if not torch.equal(actual,manual): raise SimulatorValidationError('Independent original MLP forward mismatch')
            raw=actual*record['target_std']+record['target_mean']
            restored=(raw-record['target_mean'])/record['target_std']
        results[target]=dict(architecture=str(probe),input_dim=192,output_dim=dimension,
            target_mean=record['target_mean'].tolist(),target_std=record['target_std'].tolist(),
            state_dict_exact=True,manual_forward_max_abs=0.,normalization_roundtrip_max_abs=float((restored-actual).abs().max()),
            all_modules_eval=True,provenance_limitation='Original family/seed lineage incomplete; stored transforms verified, not reconstructed training provenance')
    return results


def absolute_summary(pred,truth,owners):
    """Family-balanced errors; retain x/y versus Euclidean definitions."""
    result={};unique=sorted(set(owners))
    for target,selection in TARGETS.items():
        error=pred[target].astype(float)-truth[:,selection]
        if target=='block_angle': error=wrap_angle(error)
        squared=[np.mean(np.sum(error[np.array(owners)==owner]**2,axis=1)) for owner in unique]
        result[target]=rms_bounds(squared)
        result[target]['signed_bias']=np.mean([error[np.array(owners)==owner].mean(axis=0) for owner in unique],axis=0).tolist()
        result[target]['coordinate_rmse']=np.sqrt(np.mean([np.mean(error[np.array(owners)==owner]**2,axis=0) for owner in unique],axis=0)).tolist()
        result[target]['coordinate_pooled_rmse']=result[target]['rms']/np.sqrt(error.shape[1])
    return result


def run_readout_audit(args,repo_root):
    cfg,section,results,artifacts,manifest,parents,model,loaded=repair_context(args,repo_root,'phase1_readout_r0_audit')
    manifest['r0_gate_passed']=False;manifest['implementation_bug_found']=False;manifest['corrections']=[]
    try:
        images=[];truths=[];owners=[];identities=[];geometry=[];cached=[]
        for name,parent in parents.items():
            directory=repo_root/section[f'{name}_manifest'];families=json.loads(directory.with_name('family_metrics.json').read_text())
            with h5py.File(parent['artifacts']['natural_diagnostics.h5']['path']) as saved:
                for family in families:
                    for d,scene in enumerate(family_scenes(family)):
                        env,_=create_fresh_scene(scene,SimulatorSettings())
                        try:
                            image=env.render().copy();before=body_vector(env).copy()
                            group=saved[scene.scene_id];old_image=group['images'][0];old_state=group['body'][0]
                            if not np.array_equal(image,old_image) or not np.allclose(before,old_state,atol=1e-10,rtol=0):
                                raise SimulatorValidationError('Saved image/body correspondence failed')
                            check=geometry_audit(env,image)
                            if not np.array_equal(body_vector(env),before): raise SimulatorValidationError('Render audit changed physics')
                            label=env._get_obs()[:5]
                            if not np.allclose(label,before[[0,1,9,10,11]],atol=1e-10,rtol=0):
                                raise SimulatorValidationError('Public labels/body fields mismatch')
                            # Exercise repository target builders independently.
                            fake=dict(emb=torch.zeros((1,192)),state=torch.tensor(env._get_obs()[None]),episode_idx=torch.tensor([0]),step_idx=torch.tensor([0]))
                            for target,selection in TARGETS.items():
                                if not np.array_equal(TARGET_BUILDERS[target](fake)['target'].numpy()[0],label[selection].astype(np.float32)):
                                    raise SimulatorValidationError('Target builder coordinate contract mismatch')
                            images.append(image);truths.append(label);owners.append(family['family_id'])
                            identities.append(dict(domain='prospective',family_id=family['family_id'],dose=[0,10,20,30][d],stratum=family['stratum']))
                            geometry.append(dict(**identities[-1],**check))
                            cached.append(saved[f"{family['family_id']}_model/natural_latents"][d,0])
                        finally: env.close()
        current_count=len(images)
        old=json.loads(Path(parents['development']['compatibility_parent']['path']).read_text())
        source_case_path=Path(old['provenance_parent']['path']).parents[1]/old['run_id']/'case_metrics.json'
        source_cases=json.loads(source_case_path.read_text())
        with h5py.File(old['artifacts']['compatibility_arrays.h5']['path']) as saved, h5py.File(repo_root/cfg.experiment.cache_dir/'pusht_expert_train.h5') as source:
            ids=json.loads(saved.attrs['image_identities'])
            indices={(i['family_id'],i['condition']):n for n,i in enumerate(ids) if i['condition']=='hdf5_source'}
            for case in source_cases:
                row=case['raw_row'];label=source['state'][row].astype(float);image=source['pixels'][row]
                oldidx=indices[(case['family_id'],'hdf5_source')]
                if not np.array_equal(saved['images'][oldidx],image): raise SimulatorValidationError('Source RGB identity changed')
                images.append(image);truths.append(label[:5]);owners.append(case['family_id'])
                identities.append(dict(domain='source',family_id=case['family_id'],raw_row=row,match_scope='stored render-relevant state; NOT complete historical dynamics'))
                cached.append(saved['embeddings'][oldidx])
                scene=SceneSpec(f"r0_source_pose_{row}",20261015,tuple(label[:2]),tuple(label[2:4]),float(label[4]))
                env,_=create_fresh_scene(scene,SimulatorSettings())
                try:
                    fresh=env.render().copy();check=geometry_audit(env,fresh)
                    images.append(fresh);truths.append(env._get_obs()[:5]);owners.append(case['family_id'])
                    identities.append(dict(domain='source_pose_prospective',family_id=case['family_id'],raw_row=row,match_scope='render-relevant source labels; fixed current goal, unknown historical goal'))
                    geometry.append(dict(**identities[-1],**check));cached.append(None)
                finally: env.close()
        print(f'R0: encode/recheck {len(images)} natural images',flush=True)
        z=encode_images(model,images);repeat=encode_images(model,images)
        if not np.array_equal(z,repeat): raise SimulatorValidationError('R0 encoder repeat mismatch')
        maxima=[]
        for index,cache in enumerate(cached):
            if cache is not None:
                maxima.append(float(np.max(np.abs(z[index]-cache))))
                if not np.allclose(z[index],cache,atol=1e-5,rtol=1e-4): raise SimulatorValidationError('R0 saved latent identity failed')
        manifest['preprocessing']=preprocessing_parity(np.stack(images[:4]))
        manifest['original_readout_contracts']=verify_original_readouts(z,loaded)
        predictions=readout_latents(z,loaded);truths=np.array(truths);owners=np.array(owners)
        reports=[]
        for domain in ('prospective','source','source_pose_prospective'):
            mask=np.array([i['domain']==domain for i in identities])
            reports.append(dict(domain=domain,absolute=absolute_summary({k:v[mask] for k,v in predictions.items()},truths[mask],owners[mask])))
        concentrations=[]
        for stratum in ('free_space','near_contact'):
            mask=np.array([i['domain']=='prospective' and i['stratum']==stratum for i in identities])
            concentrations.append(dict(group=stratum,absolute=absolute_summary({k:v[mask] for k,v in predictions.items()},truths[mask],owners[mask])))
        for quadrant in range(4):
            mask=np.array([i['domain']=='prospective' and int(truths[n,4]//(np.pi/2))==quadrant for n,i in enumerate(identities)])
            if mask.any(): concentrations.append(dict(group=f'angle_quadrant_{quadrant}',absolute=absolute_summary({k:v[mask] for k,v in predictions.items()},truths[mask],owners[mask])))
        rows=[]
        for n,identity in enumerate(identities):
            row={**identity,'truth':truths[n].tolist(),'predictions':{k:v[n].tolist() for k,v in predictions.items()}}
            rows.append(row)
        write_json(results/'audit_measurements.json',rows);write_json(results/'geometry_audit.json',geometry)
        write_json(results/'pose_concentration.json',concentrations);write_json(results/'absolute_domain_comparison.json',reports)
        with h5py.File(artifacts/'audit_arrays.h5','x') as output:
            output.create_dataset('images',data=np.stack(images),compression='gzip',compression_opts=1)
            output.create_dataset('labels',data=truths);output.create_dataset('latents',data=z)
            for target,value in predictions.items(): output.create_dataset(f'readouts/{target}',data=value)
            output.attrs['identities']=json.dumps(identities)
        manifest['measurements']=dict(fresh_initial_rows=current_count,total_rows=len(images),source_families=len(source_cases),
            encoder_repeat_max_abs=0.,saved_latent_identity_max_abs=max(maxima),geometry_rows=len(geometry),absolute_domains=reports,
            geometry_transform_max_abs=max(g['analytic_transform_max_abs'] for g in geometry),
            geometry_mismatch_counts={k:sum(g['raster'][name][k] for g in geometry for name in ('agent','block')) for k in ('missing_deep_interior','outside_boundary_band')})
        manifest['regressions']=repair_regressions(repo_root,artifacts,('test_counterfactual_readout_audit.py',))
        manifest['r0_gate_passed']=True;manifest['run_status']='completed'
    except Exception as error:
        manifest['run_status']='failed';manifest['failure']=dict(type=type(error).__name__,message=str(error))
        (results/'failure_traceback.txt').write_text(traceback.format_exc())
        print(f'R0 failed: {error}',file=sys.stderr)
    finish_repair(manifest,results,artifacts)
    print(f"R0 {manifest['run_status']}; pipeline gate={manifest['r0_gate_passed']}; {results/'repair_manifest.json'}",flush=True)
    return 0 if manifest['run_status']=='completed' else 1
