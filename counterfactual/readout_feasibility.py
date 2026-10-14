"""R1 bounded natural-latent readout fit; no official H1/P or H2 promotion."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import traceback

import h5py
import numpy as np
from omegaconf import OmegaConf
import torch

from counterfactual.baseline import file_metadata,sha256_file,write_json
from counterfactual.domain_diagnostics import near_duplicate,reserved_families,rms_bounds,TARGETS
from counterfactual.observation_compatibility import encode_images,readout_latents
from counterfactual.readout_audit import finish_repair,repair_context,repair_regressions
from counterfactual.simulator import (SceneSpec,SimulatorSettings,SimulatorValidationError,
    canonical_json,create_fresh_scene,jsonable,physical_readback,wrap_angle)
from counterfactual.simulator_validation import validation_cases
from probe.evaluate import load_probe_checkpoint,probe_checkpoint_path
from probe.models import MLPRegressionProbe
from probe.train import seed_probe_training

HEADS=('agent_location','block_location','angle_scalar','angle_sincos')


def variants(family):
    """All independent physical changes remain in one family, including zero."""
    base=SceneSpec(f"{family['family_id']}_base",int(family['seed']),tuple(family['agent_position']),
        tuple(family['block_position']),float(family['block_angle']))
    direction=np.array(family['direction']);perpendicular=np.array([-direction[1],direction[0]])
    result=[(dict(key='base',kind='base',dose=0.,sign=0),base)]
    for dose in (10.,20.,30.):
        key=f'agent_{int(dose)}'
        result.append((dict(key=key,kind='agent',dose=dose,sign=1),replace(base,scene_id=f"{family['family_id']}_{key}",
            agent_position=tuple(np.array(base.agent_position)+dose*direction))))
    for dose in (1.,2.,5.,10.):
        for sign in (-1,1):
            key=f"block_{'m' if sign<0 else 'p'}{int(dose)}"
            result.append((dict(key=key,kind='block',dose=dose,sign=sign),replace(base,scene_id=f"{family['family_id']}_{key}",
                block_position=tuple(np.array(base.block_position)+sign*dose*perpendicular))))
    for dose in (.01,.025,.05,.10):
        for sign in (-1,1):
            key=f"angle_{'m' if sign<0 else 'p'}{dose}"
            result.append((dict(key=key,kind='angle',dose=dose,sign=sign),replace(base,scene_id=f"{family['family_id']}_{key}",
                block_angle=float((base.block_angle+sign*dose)%(2*np.pi)))))
    result.append((dict(key='zero_repeat',kind='zero',dose=0.,sign=0),replace(base,scene_id=f"{family['family_id']}_zero_repeat")))
    return result


def family_assignments(section,parents,repo_root):
    keys=('family_id','stratum','seed','agent_position','block_position','block_angle','direction')
    families=[]
    for name in ('development','validation'):
        path=Path(parents[name]['diagnostic_parents'][name]['path']) if 'diagnostic_parents' in parents[name] else repo_root/section[f'{name}_manifest']
        for row in json.loads(path.with_name('family_metrics.json').read_text()):
            families.append(dict(**{key:row[key] for key in keys},role='training',origin='previously_exposed_v2'))
    settings=OmegaConf.create(dict(development_seed=int(section.fresh_training_seed),validation_seed=int(section.fresh_pilot_seed),
        families_per_partition=int(section.new_training_families)))
    if int(section.new_training_families)!=int(section.pilot_validation_families): raise ValueError('Predeclared fresh partitions must both have16 families')
    for row in reserved_families(settings):
        role='training' if row['partition']=='development' else 'pilot_validation'
        row['family_id']=row['family_id'].replace('domain_v2_development','r1_train').replace('domain_v2_validation','r1_pilot')
        families.append(dict(**{key:row[key] for key in keys},role=role,origin='new_r1'))
    # Interleave deterministic strata permutations: every nested even count balanced.
    rng=np.random.default_rng(int(section.ordering_seed))
    ordered={}
    for stratum in ('free_space','near_contact'):
        members=sorted([f for f in families if f['role']=='training' and f['stratum']==stratum],key=lambda f:f['family_id'])
        ordered[stratum]=[members[i] for i in rng.permutation(len(members))]
    training=[f for pair in zip(ordered['free_space'],ordered['near_contact']) for f in pair]
    for index,f in enumerate(training): f['training_order']=index
    return training+sorted([f for f in families if f['role']=='pilot_validation'],key=lambda f:f['family_id'])


def check_family_isolation(families,historical):
    all_poses=[]
    for family in families:
        for _,scene in variants(family):
            pose=scene.public_state()
            if any(near_duplicate(pose,old) for old in historical): raise SimulatorValidationError('R1 family overlaps a historical/fixture pose')
            for owner,other in all_poses:
                if owner!=family['family_id'] and near_duplicate(pose,other):
                    raise SimulatorValidationError(f"Related R1 poses cross families: {owner}, {family['family_id']}")
            all_poses.append((family['family_id'],pose))


def collect_inputs(families,model,artifacts,results,deadline):
    images=[];truth=[];owners=[];roles=[];identities=[];attempts=[];accepted=[]
    with h5py.File(artifacts/'natural_inputs.h5','x') as output:
        for family in families:
            print(f"R1 natural family: {family['family_id']} ({family['role']})",flush=True)
            local_images=[];local_labels=[];reads=[];checks=[];metadata=[]
            try:
                for identity,scene in variants(family):
                    if time.monotonic()>deadline: raise TimeoutError('Readout feasibility hard timeout')
                    env,validation=create_fresh_scene(scene,SimulatorSettings())
                    try:
                        read=physical_readback(env);image=env.render().copy()
                        if not np.array_equal(physical_readback(env)['body'],read['body']): raise SimulatorValidationError('Rendering changed physical labels')
                        label=read['body'][[0,1,9,10,11]].copy();label[4]=float(label[4]%(2*np.pi))
                        if not np.allclose(label,read['public_state'][:5],atol=1e-10,rtol=0): raise SimulatorValidationError('Natural input/label mapping failed')
                        local_images.append(image);local_labels.append(label);reads.append(read);checks.append(validation);metadata.append(identity)
                    finally:env.close()
                if not np.array_equal(local_images[0],local_images[-1]) or not np.array_equal(reads[0]['body'],reads[-1]['body']):
                    raise SimulatorValidationError('Duplicate zero scene changed image/body')
                group=output.create_group(family['family_id'])
                group.create_dataset('body_readbacks',data=np.stack([r['body'] for r in reads]))
                group.attrs['checks']=canonical_json(checks);group.attrs['scene_specs']=canonical_json([vars(s) for _,s in variants(family)])
                group.attrs['goal_state']=canonical_json(reads[0]['goal_state']);group.attrs['goal_pose']=canonical_json(reads[0]['goal_pose'])
                images.extend(local_images);truth.extend(local_labels);owners.extend([family['family_id']]*21);roles.extend([family['role']]*21)
                identities.extend([dict(family_id=family['family_id'],role=family['role'],stratum=family['stratum'],**i) for i in metadata])
                accepted.append(family);attempts.append(dict(family_id=family['family_id'],status='valid'))
            except SimulatorValidationError as error:
                attempts.append(dict(family_id=family['family_id'],status='invalid',reason=str(error)))
            write_json(results/'attempts.json',attempts)
        if not images:raise SimulatorValidationError('No valid natural input families')
        print(f'R1 frozen encoder: {len(images)} natural images',flush=True)
        z=encode_images(model,images)
        if not np.array_equal(z,encode_images(model,images)):raise SimulatorValidationError('Frozen R1 encoding repeat mismatch')
        truth=np.array(truth)
        output.create_dataset('images',data=np.stack(images),compression='gzip',compression_opts=1)
        output.create_dataset('latents',data=z);output.create_dataset('physical_labels',data=truth)
        output.attrs['identities']=canonical_json(identities);output.attrs['scope']='R1 natural readout feasibility, NOT formal Step3 or H2'
    return dict(z=z,y=truth,owners=np.array(owners),roles=np.array(roles),identities=identities,families=accepted),attempts


def family_weights(owners):
    owners=np.asarray(owners);ids,counts=np.unique(owners,return_counts=True)
    lookup=dict(zip(ids,counts))
    return np.array([1/(len(ids)*lookup[owner]) for owner in owners],np.float64)


def target_values(labels,head):
    if head in ('agent_location','block_location'):return labels[:,TARGETS[head]]
    angles=labels[:,4:5]
    return angles if head=='angle_scalar' else np.c_[np.sin(angles[:,0]),np.cos(angles[:,0])]


def target_statistics(targets,weights):
    mean=np.sum(targets*weights[:,None],axis=0)
    std=np.sqrt(np.sum((targets-mean)**2*weights[:,None],axis=0))
    if not np.isfinite(std).all() or np.any(std<=0):raise ValueError('Constant/nonfinite target cannot be normalized; no epsilon repair')
    return mean,std


def fit_ridge(x,y,weights,penalty):
    """Weighted mean-loss objective, raw latent coordinates, unpenalized bias."""
    if penalty<=0 or not np.isfinite(penalty) or np.any(weights<=0) or not np.isclose(weights.sum(),1.):
        raise ValueError('Ridge requires positive mean-loss weights and penalty')
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    xm=np.sum(x*weights[:,None],axis=0);ym=np.sum(y*weights[:,None],axis=0)
    centered=x-xm;yc=y-ym
    coef=np.linalg.solve(centered.T@(centered*weights[:,None])+penalty*np.eye(x.shape[1]),centered.T@(yc*weights[:,None])).T
    return dict(coef=coef,intercept=ym-coef@xm)


def decode_head(output,head):
    output=np.asarray(output,np.float64)
    if head!='angle_sincos':return output.copy(),np.isfinite(output).all(axis=1)
    norm=np.linalg.norm(output,axis=1);valid=np.isfinite(output).all(axis=1)&(norm>1e-8)
    angle=np.full((len(output),1),np.nan)
    angle[valid,0]=np.arctan2(output[valid,0],output[valid,1])%(2*np.pi)
    return angle,valid


def head_error(predicted,labels,head):
    truth=labels[:,TARGETS[head]] if head in TARGETS else labels[:,4:5]
    error=predicted-truth
    return wrap_angle(error) if head.startswith('angle') or head=='block_angle' else error


def absolute_metrics(predicted,labels,owners,head,seed=2026101504,draws=10000):
    error=head_error(predicted,labels,head);ids=sorted(set(owners));values=[];means=[];moments=[];bias=[]
    if not np.isfinite(error).all():return dict(status='undefined',invalid_rows=int((~np.isfinite(error).all(axis=1)).sum()),n_families=len(ids))
    truth=labels[:,TARGETS[head]] if head in TARGETS else labels[:,4:5]
    for owner in ids:
        mask=owners==owner;values.append(np.mean(error[mask]**2,axis=0));means.append(truth[mask].mean(axis=0));moments.append(np.mean(truth[mask]**2,axis=0));bias.append(error[mask].mean(axis=0))
    values=np.array(values);result=rms_bounds(values.sum(axis=1),seed,draws)
    result.update(status='pilot_conditional',coordinate_rmse=np.sqrt(values.mean(axis=0)).tolist(),
        coordinate_pooled_rmse=float(np.sqrt(values.mean())),signed_bias=np.mean(bias,axis=0).tolist(),family_mse=values.tolist(),family_ids=ids)
    if head in ('agent_location','block_location'):
        means=np.array(means);moments=np.array(moments);variance=moments.mean(axis=0)-means.mean(axis=0)**2
        rng=np.random.default_rng(seed);sample=rng.integers(0,len(ids),(draws,len(ids)))
        vb=moments[sample].mean(axis=1)-means[sample].mean(axis=1)**2
        if np.any(variance<=0) or np.any(vb<=0):result['r2_status']='undefined_constant_coordinate'
        else:
            r2=1-values.mean(axis=0)/variance;boot=1-values[sample].mean(axis=1)/vb
            result.update(coordinate_r2=r2.tolist(),mean_coordinate_r2=float(r2.mean()),r2_ci95=np.quantile(boot.mean(axis=1),[.025,.975]).tolist(),r2_lower95=float(np.quantile(boot.mean(axis=1),.05)))
    return result


def validation_score(output,labels,owners,head):
    predicted,valid=decode_head(output,head)
    if not valid.all():return float('inf')
    error=head_error(predicted,labels,head)
    return float(np.sqrt(np.sum(family_weights(owners)*np.sum(error**2,axis=1))))


def fit_mlp(x,y,xval,yval,owners,valowners,labels,val_labels,head,seed,section,deadline):
    weights=family_weights(owners);wval=family_weights(valowners);mean,std=target_statistics(y,weights)
    seed_probe_training(seed)
    probe=MLPRegressionProbe(192,256,y.shape[1],1,.1)
    optimizer=torch.optim.AdamW(probe.parameters(),lr=float(section.mlp_lr),weight_decay=float(section.mlp_weight_decay))
    xt=torch.from_numpy(x).float();yt=torch.from_numpy(((y-mean)/std).astype(np.float32))
    xv=torch.from_numpy(xval).float();yv=torch.from_numpy(((yval-mean)/std).astype(np.float32))
    wt=torch.from_numpy(weights.astype(np.float32));wv=torch.from_numpy(wval.astype(np.float32))
    best=float('inf');best_state=None;best_epoch=None;stale=0;curves=[];every=int(section.mlp_evaluate_every)
    for epoch in range(1,int(section.mlp_epochs)+1):
        if time.monotonic()>deadline:raise TimeoutError('R1 training exceeded configured hard timeout')
        probe.train();prediction=probe(xt)
        loss=((prediction-yt)**2).mean(dim=1).mul(wt).sum()
        if not torch.isfinite(loss):raise SimulatorValidationError('Nonfinite MLP optimization loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if epoch%every==0 or epoch==int(section.mlp_epochs):
            probe.eval()
            with torch.inference_mode():
                tr=probe(xt);va=probe(xv);raw_tr=tr.numpy()*std+mean;raw_val=va.numpy()*std+mean
                tl=float(((tr-yt)**2).mean(dim=1).mul(wt).sum());vl=float(((va-yv)**2).mean(dim=1).mul(wv).sum())
            score=validation_score(raw_val,val_labels,valowners,head)
            curves.append(dict(epoch=epoch,train_normalized_mse=tl,validation_normalized_mse=vl,
                train_physical_rmse=validation_score(raw_tr,labels,owners,head),validation_physical_rmse=score if np.isfinite(score) else None))
            if score<best:
                best=score;best_epoch=epoch;best_state={k:v.detach().clone() for k,v in probe.state_dict().items()};stale=0
            else:stale+=every
            if stale>=int(section.mlp_patience):break
    if best_state is None:raise SimulatorValidationError('Every validation decode was degenerate; no usable MLP selected')
    probe.load_state_dict(best_state);probe.eval();probe.requires_grad_(False)
    return probe,dict(target_mean=mean,target_std=std,best_epoch=best_epoch,curves=curves)


def pair_metrics(predicted,truth,head,indices,seed,draws):
    """Within-family dose contrasts, with wrapped angular effects and residuals."""
    angular=head.startswith('angle') or head=='block_angle'
    pd=predicted[:,indices]-predicted[:,:1];gt=truth[:,indices]-truth[:,:1]
    if angular:pd=wrap_angle(pd);gt=wrap_angle(gt)
    residual=wrap_angle(pd-gt) if angular else pd-gt
    if not np.isfinite(residual).all():return dict(status='undefined',invalid_rows=int((~np.isfinite(residual).all(axis=-1)).sum()))
    error=np.linalg.norm(residual,axis=-1);prednorm=np.linalg.norm(pd,axis=-1);realnorm=np.linalg.norm(gt,axis=-1)
    result=rms_bounds(np.mean(error**2,axis=1),seed,draws)
    result.update(status='pilot_conditional',family_squared_errors=np.mean(error**2,axis=1).tolist(),
        coordinate_rmse=np.sqrt(np.mean(residual**2,axis=(0,1))).tolist(),signed_bias=residual.mean(axis=(0,1)).tolist(),
        magnitude_error=rms_bounds(np.mean((prednorm-realnorm)**2,axis=1),seed,draws),
        predicted_changes=pd.tolist(),true_changes=gt.tolist())
    valid=(prednorm>0)&(realnorm>0)
    cos=np.full(prednorm.shape,np.nan);cos[valid]=np.sum(pd[valid]*gt[valid],axis=-1)/(prednorm[valid]*realnorm[valid])
    result['direction_undefined_count']=int((~valid).sum())
    if valid.all():
        family_cos=cos.mean(axis=1);sample=np.random.default_rng(seed).integers(0,len(family_cos),(draws,len(family_cos)))
        b=family_cos[sample].mean(axis=1)
        result['direction']=dict(mean=float(family_cos.mean()),lower95=float(np.quantile(b,.05)),ci95=np.quantile(b,[.025,.975]).tolist(),
            positive_fraction=float((cos>0).mean()))
    return result


def resolution_status(metric,margin):
    if metric.get('status')=='undefined':return 'INCONCLUSIVE'
    if metric['lower95']>margin:return 'FAIL'
    if metric['upper95']<=margin:return 'PASS'
    return 'INCONCLUSIVE'


def evaluate_candidate(predicted,data,head,section):
    labels=data['y'];owners=data['owners'];mask=data['roles']=='pilot_validation'
    identities=data['identities'];distinct=np.array([i['kind']!='zero' for i in identities])
    absolute=absolute_metrics(predicted[mask&distinct],labels[mask&distinct],owners[mask&distinct],head,int(section.bootstrap_seed),int(section.bootstrap_draws))
    unique=sorted(set(owners[mask]));pr=np.stack([predicted[owners==f] for f in unique]);tr=np.stack([labels[owners==f] for f in unique])
    truth=tr[:,:,TARGETS[head]] if head in TARGETS else tr[:,:,4:5]
    template=[i for i in identities if i['family_id']==unique[0]];pairs={}
    for kind,doses in (('zero',(0.,)),('agent',(10.,20.,30.)),('block',(1.,2.,5.,10.)),('angle',(.01,.025,.05,.10))):
        for dose in doses:
            idx=[n for n,i in enumerate(template) if i['kind']==kind and i['dose']==dose]
            metric=pair_metrics(pr,truth,head,idx,int(section.bootstrap_seed),int(section.bootstrap_draws))
            target_kind='agent' if head=='agent_location' else 'block' if head=='block_location' else 'angle'
            angular=target_kind=='angle';cap=.10 if angular else 5.
            if kind==target_kind:
                margin=min(cap,dose/2)
                metric.update(proposed_margin=margin,error_status=resolution_status(metric,margin))
                direction=metric.get('direction')
                metric['resolution_status']='PASS' if metric['error_status']=='PASS' and direction is not None and direction['lower95']>0 else 'FAIL' if metric['error_status']=='FAIL' else 'INCONCLUSIVE'
            else:metric.update(non_target_cap=cap,non_target_status=resolution_status(metric,cap))
            pairs[f'{kind}_{dose:g}']=metric
    absolute['practical_cap']=.10 if head.startswith('angle') or head=='block_angle' else 5.
    absolute['cap_status']=resolution_status(absolute,absolute['practical_cap'])
    return dict(absolute=absolute,pairs=pairs,n_families=len(unique),scope='Pilot-validation, used for model/epoch selection; conditional CIs, NOT confirmatory')


def row_space(coef):
    _,s,vt=np.linalg.svd(coef,full_matrices=False)
    tolerance=max(coef.shape)*np.finfo(coef.dtype).eps*s[0]
    rank=int(np.sum(s>tolerance))
    return vt[:rank].T,rank


def subspace_angles(a,b):
    s=np.linalg.svd(a.T@b,compute_uv=False)
    if np.any((s<0)|(s>1)):return dict(status='undefined_numerical_range',singular_values=s.tolist())
    return dict(status='descriptive',degrees=np.rad2deg(np.arccos(s)).tolist())


def load_candidate_predictions(path,z):
    """R1-only schema; never install into the legacy official probe loader."""
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('schema_version')!='readout-r1-v1' or payload.get('official_instrument') is not False:
        raise ValueError('Not a separately versioned R1 candidate')
    mean=payload['target_mean'].numpy();std=payload['target_std'].numpy()
    if payload['method']=='ridge':
        output=(np.asarray(z,float)@payload['coef'].numpy().T+payload['intercept'].numpy())*std+mean
    else:
        probe=MLPRegressionProbe(192,256,len(mean),1,.1)
        probe.load_state_dict(payload['probe_state_dict']);probe.eval();probe.requires_grad_(False)
        with torch.inference_mode():output=probe(torch.as_tensor(z).float()).numpy()*std+mean
    return decode_head(output,payload['head'])[0]


def paired_rms_gain(left,right,seed=2026101504,draws=10000):
    """Positive means right improves; paired whole-family resampling."""
    left=np.asarray(left,float);right=np.asarray(right,float)
    if left.shape!=right.shape or left.ndim!=1:raise ValueError('Paired family vectors must align')
    sample=np.random.default_rng(seed).integers(0,len(left),(draws,len(left)))
    value=np.sqrt(left[sample].mean(axis=1))-np.sqrt(right[sample].mean(axis=1))
    return dict(gain=float(np.sqrt(left.mean())-np.sqrt(right.mean())),ci95=np.quantile(value,[.025,.975]).tolist(),
        lower95=float(np.quantile(value,.05)),upper95=float(np.quantile(value,.95)),n_families=len(left))


def summarize_feasibility(rows,selected,section):
    lookup={r['candidate_id']:r for r in rows};contrasts={};bundles={}
    def family_errors(row):return np.sum(row['metrics']['absolute']['family_mse'],axis=1)
    for head in HEADS:
        ridge=lookup[selected[f'n48_ridge_{head}']]
        mlps=[lookup[f'n48_mlp_{head}_seed{int(s)}'] for s in section.mlp_seeds]
        if all(r['metrics']['absolute'].get('status')!='undefined' for r in [ridge,*mlps]):
            combined=np.mean([family_errors(r) for r in mlps],axis=0)
            contrasts[f'{head}_ridge_to_mlp_seed_mean']=paired_rms_gain(family_errors(ridge),combined,int(section.bootstrap_seed),int(section.bootstrap_draws))
            for method in ('ridge','mlp_seed_mean'):
                a=lookup[selected[f'n16_ridge_{head}']] if method=='ridge' else None
                small=family_errors(a) if a else np.mean([family_errors(lookup[f'n16_mlp_{head}_seed{int(s)}']) for s in section.mlp_seeds],axis=0)
                large=family_errors(ridge) if a else combined
                contrasts[f'{head}_{method}_16_to_48']=paired_rms_gain(small,large,int(section.bootstrap_seed),int(section.bootstrap_draws))
    for method in ('ridge',*[f'mlp_seed{int(s)}' for s in section.mlp_seeds]):
        for angle_head in ('angle_scalar','angle_sincos'):
            heads={h:lookup[selected[f'n48_ridge_{h}']] if method=='ridge' else lookup[f'n48_mlp_{h}_seed{method.removeprefix("mlp_seed")}'] for h in ('agent_location','block_location',angle_head)}
            resolution={}
            for kind,doses,target in (('agent',(10.,20.,30.),'agent_location'),('block',(1.,2.,5.,10.),'block_location'),('angle',(.01,.025,.05,.10),angle_head)):
                for dose in doses:
                    key=f'{kind}_{dose:g}';self_status=heads[target]['metrics']['pairs'][key]['resolution_status']
                    controls=[r['metrics']['pairs'][key]['non_target_status'] for h,r in heads.items() if h!=target]
                    status='FAIL' if self_status=='FAIL' or 'FAIL' in controls else 'PASS' if self_status=='PASS' and all(c=='PASS' for c in controls) else 'INCONCLUSIVE'
                    resolution[key]=dict(status=status,target_status=self_status,non_target_statuses=controls)
            passed_blocks=[d for d in (1.,2.,5.,10.) if resolution[f'block_{d:g}']['status']=='PASS']
            bundles[f'{method}_{angle_head}']=dict(resolution=resolution,smallest_tested_block_scale=next(iter(passed_blocks),None),
                high_resolution_1px_objective=resolution['block_1']['status'],
                absolute_cap_statuses={h:r['metrics']['absolute']['cap_status'] for h,r in heads.items()},
                scope='Feasibility only; these grouped instrument combinations are NOT official or confirmed')
    return dict(paired_contrasts=contrasts,instrument_bundles=bundles,
        minimum_half_effect_margin_block_px={str(d):min(5.,d/2) for d in (1.,2.,5.,10.)},
        high_resolution_objective_is_universal_prerequisite=False,step3_one_px_feasibility_threshold_changed=False,
        criteria_status='Predeclared feasibility proposals; final confirmation needs fresh independent families')


def run_readout_feasibility(args,repo_root):
    if not args.audit_manifest:raise ValueError('R1 requires a completed passing R0 --audit-manifest')
    audit_path=Path(args.audit_manifest).resolve();audit=json.loads(audit_path.read_text())
    if audit.get('stage')!='phase1_readout_r0_audit' or audit.get('run_status')!='completed' or audit.get('r0_gate_passed') is not True:
        raise SimulatorValidationError('R0 did not pass; do not train readouts')
    for path,digest in audit['source_sha256'].items():
        if sha256_file(path)!=digest:raise SimulatorValidationError(f'R0 audited implementation/config changed: {path}')
    cfg,section,results,artifacts,manifest,parents,model,loaded=repair_context(args,repo_root,'phase1_readout_r1_feasibility')
    manifest['audit_parent']=file_metadata(audit_path,include_hash=True)
    deadline=time.monotonic()+float(section.timeout_seconds)
    try:
        families=family_assignments(section,parents,repo_root)
        old=json.loads(Path(parents['development']['compatibility_parent']['path']).read_text())
        historical=json.loads(Path(old['provenance_parent']['path']).with_name('initial_families.json').read_text())
        check_family_isolation(families,[f['initial_state'] for f in historical]+[s.public_state() for s,_,_ in validation_cases(20261014)])
        write_json(results/'family_assignments.json',families)
        data,attempts=collect_inputs(families,model,artifacts,results,deadline)
        manifest['dataset']=dict(natural_array=file_metadata(artifacts/'natural_inputs.h5',include_hash=True),attempts=attempts,
            training_families=sum(f['role']=='training' for f in data['families']),pilot_validation_families=sum(f['role']=='pilot_validation' for f in data['families']),
            images=len(data['z']),source='Verified natural fresh quiescent renders only; no predicted/patched latents',encoder_repeat_max_abs=0.)
        if len(data['families'])!=64:raise SimulatorValidationError('Predeclared family coverage incomplete; no replacement or fit on reduced convenient sample')
        input_hashes={k:__import__('hashlib').sha256(data[k].tobytes()).hexdigest() for k in ('z','y')}
        model_parameters_frozen=all(not p.requires_grad for p in model.parameters())
        if not model_parameters_frozen:raise SimulatorValidationError('LeWM became trainable')
        results_rows=[];curves=[];selected={};spaces={};validation_mask=data['roles']=='pilot_validation'
        distinct=np.array([i['kind']!='zero' for i in data['identities']])
        vm=validation_mask&distinct
        baseline=readout_latents(data['z'],loaded)
        output_predictions={}
        for head in TARGETS:
            name=f'original_mlp_{head}';output_predictions[name]=baseline[head]
            results_rows.append(dict(candidate_id=name,head=head,method='original_mlp',training_families=0,selected=True,
                metrics=evaluate_candidate(baseline[head],data,head,section)))
        weights_dir=artifacts/'readouts';weights_dir.mkdir()
        for count in section.training_family_counts:
            count=int(count);training=data['families'][:count]
            ids=[f['family_id'] for f in training]
            if any(f['role']!='training' for f in training):raise SimulatorValidationError('Validation families entered training')
            tm=np.isin(data['owners'],ids)&distinct
            x=data['z'][tm];labels=data['y'][tm];owners=data['owners'][tm]
            xv=data['z'][vm];vl=data['y'][vm];vo=data['owners'][vm];weights=family_weights(owners)
            for head in HEADS:
                y=target_values(labels,head);yv=target_values(vl,head);mean,std=target_statistics(y,weights)
                normalized=(y-mean)/std
                ridge_rows=[]
                for penalty in section.ridge_lambdas:
                    name=f'n{count}_ridge_{head}_lambda{float(penalty):g}'
                    fitted=fit_ridge(x,normalized,weights,float(penalty))
                    output=(data['z'].astype(float)@fitted['coef'].T+fitted['intercept'])*std+mean
                    pred,valid=decode_head(output,head)
                    score=validation_score(output[vm],vl,vo,head)
                    path=weights_dir/f'{name}.pt'
                    torch.save(dict(schema_version='readout-r1-v1',method='ridge',head=head,coef=torch.from_numpy(fitted['coef']),
                        intercept=torch.from_numpy(fitted['intercept']),target_mean=torch.from_numpy(mean),target_std=torch.from_numpy(std),
                        penalty=float(penalty),training_family_ids=ids,natural_input_sha256=manifest['dataset']['natural_array']['sha256'],
                        official_instrument=False),path)
                    if not np.array_equal(load_candidate_predictions(path,data['z']),pred,equal_nan=True):raise SimulatorValidationError('Saved Ridge transform/reload mismatch')
                    row=dict(candidate_id=name,head=head,method='ridge',training_families=count,penalty=float(penalty),selected=False,
                        validation_selection_score=score if np.isfinite(score) else None,invalid_decode_rows=int((~valid).sum()),artifact=file_metadata(path,include_hash=True),
                        train_absolute=absolute_metrics(pred[tm],labels,owners,head,int(section.bootstrap_seed),int(section.bootstrap_draws)),
                        metrics=evaluate_candidate(pred,data,head,section))
                    output_predictions[name]=pred;ridge_rows.append((score,float(penalty),row,fitted));results_rows.append(row)
                best_score=min(r[0] for r in ridge_rows)
                if not np.isfinite(best_score):raise SimulatorValidationError('No finite Ridge candidate')
                best=max([r for r in ridge_rows if r[0]<=best_score+1e-12],key=lambda r:r[1])
                best[2]['selected']=True;selected[f'n{count}_ridge_{head}']=best[2]['candidate_id']
                if head=='agent_location':spaces[count]=row_space(best[3]['coef'])[0]
                for seed in section.mlp_seeds:
                    seed=int(seed);name=f'n{count}_mlp_{head}_seed{seed}'
                    print(f'R1 fit: {name}',flush=True)
                    probe,meta=fit_mlp(x,y,xv,yv,owners,vo,labels,vl,head,seed,section,deadline)
                    with torch.inference_mode():output=probe(torch.from_numpy(data['z']).float()).numpy()*meta['target_std']+meta['target_mean']
                    pred,valid=decode_head(output,head);output_predictions[name]=pred
                    path=weights_dir/f'{name}.pt'
                    torch.save(dict(schema_version='readout-r1-v1',method='mlp256',head=head,probe_state_dict=probe.state_dict(),
                        target_mean=torch.from_numpy(meta['target_mean']),target_std=torch.from_numpy(meta['target_std']),
                        best_epoch=meta['best_epoch'],seed=seed,dropout=.1,hidden_dim=256,training_family_ids=ids,
                        natural_input_sha256=manifest['dataset']['natural_array']['sha256'],official_instrument=False),path)
                    if not np.array_equal(load_candidate_predictions(path,data['z']),pred,equal_nan=True):raise SimulatorValidationError('Saved MLP transform/reload mismatch')
                    row=dict(candidate_id=name,head=head,method='mlp256',training_families=count,seed=seed,selected=True,
                        best_epoch=meta['best_epoch'],invalid_decode_rows=int((~valid).sum()),artifact=file_metadata(path,include_hash=True),
                        train_absolute=absolute_metrics(pred[tm],labels,owners,head,int(section.bootstrap_seed),int(section.bootstrap_draws)),
                        metrics=evaluate_candidate(pred,data,head,section))
                    results_rows.append(row);curves.append(dict(candidate_id=name,head=head,count=count,seed=seed,epochs=meta['curves']))
                    write_json(results/'learning_curves.json',curves);write_json(results/'readout_comparison.json',jsonable(results_rows))
        with h5py.File(artifacts/'candidate_predictions.h5','x') as output:
            for name,prediction in output_predictions.items():output.create_dataset(name,data=prediction)
            output.attrs['scope']='Natural R1 readout predictions only; NOT LeWM predictor or patched latents'
        if any(__import__('hashlib').sha256(data[k].tobytes()).hexdigest()!=digest for k,digest in input_hashes.items()):raise SimulatorValidationError('Readout training mutated input latents/labels')
        manifest['selected_ridge_candidates']=selected
        manifest['feasibility_assessment']=summarize_feasibility(results_rows,selected,section)
        manifest['candidate_artifact_reloads_exact']=True
        manifest['h1_subspace_assessment']=dict(official_ridge_present=False,official_projector_created=False,
            scope='R1 Ridge fits are feasibility instruments, not H1 or causal/selectivity proof',
            numerical_rank={str(n):spaces[n].shape[1] for n in spaces},
            row_space_stability={f'{a}_to_{b}':subspace_angles(spaces[a],spaces[b]) for a,b in ((16,32),(32,48))})
        source_linear=probe_checkpoint_path(repo_root/cfg.experiment.cache_dir,'agent_location','linear')
        if source_linear.exists():
            original=load_probe_checkpoint(source_linear,'cpu');original['probe'].requires_grad_(False)
            with torch.inference_mode():p=(original['probe'](torch.from_numpy(data['z']))*original['target_std']+original['target_mean']).numpy()
            manifest['h1_subspace_assessment']['legacy_source_affine']=dict(artifact=file_metadata(source_linear,include_hash=True),
                type='AdamW affine readout, NOT Ridge',metrics=evaluate_candidate(p,data,'agent_location',section))
            basis,rank=row_space((original['probe'].weight.detach().numpy()*original['target_std'].numpy().T).astype(float))
            manifest['h1_subspace_assessment']['legacy_affine_to_r1_ridge_angles']=subspace_angles(basis,spaces[48])
        manifest['world_model_parameters_remain_frozen']=model_parameters_frozen and all(not p.requires_grad and p.grad is None for p in model.parameters())
        manifest['regressions']=repair_regressions(repo_root,artifacts,('test_counterfactual_readout_audit.py','test_counterfactual_readout_feasibility.py'))
        manifest['run_status']='completed';manifest['r1_collection_and_fits_completed']=True
        write_json(results/'repair_manifest.json',jsonable(manifest))
        from counterfactual.readout_repair_report import write_r1_report
        write_r1_report(results)
    except Exception as error:
        manifest['run_status']='failed';manifest['failure']=dict(type=type(error).__name__,message=str(error))
        (results/'failure_traceback.txt').write_text(traceback.format_exc());print(f'R1 failed: {error}',file=sys.stderr)
    finish_repair(manifest,results,artifacts)
    print(f"R1 {manifest['run_status']}; no instrument promotion/R2/Step3; {results/'repair_manifest.json'}",flush=True)
    return 0 if manifest['run_status']=='completed' else 1
