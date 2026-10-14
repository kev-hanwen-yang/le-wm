"""Direct-execution R1 tests; toy fits are not evidence for readout repair."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
import torch

from counterfactual.readout_feasibility import (check_family_isolation,decode_head,
    family_assignments,family_weights,fit_mlp,fit_ridge,load_candidate_predictions,
    paired_rms_gain,pair_metrics,resolution_status,row_space,target_statistics,
    target_values,variants)
from counterfactual.simulator import SimulatorSettings,create_fresh_scene


def family():
    return dict(family_id='toy',seed=42,agent_position=[149.,245.],block_position=[256.,230.],block_angle=0.,direction=[1.,0.],stratum='near_contact',role='training')


def test_variants_are_independent_and_have_one_family_and_fixed_goal():
    definitions=variants(family());assert len(definitions)==21
    assert len({s.seed for _,s in definitions})==1
    for row,s in definitions:
        assert s.goal_state==(70.,420.,350.,120.,0.,0.,0.)
        if row['kind']=='agent':assert s.block_position==(256.,230.) and s.block_angle==0.
        elif row['kind']=='block':assert s.agent_position==(149.,245.) and s.block_angle==0.
        elif row['kind']=='angle':assert s.agent_position==(149.,245.) and s.block_position==(256.,230.)
    assert definitions[-1][0]['kind']=='zero'


def test_variants_are_valid_fresh_quiescent_not_dynamic_restore():
    for row,s in variants(family()):
        env,check=create_fresh_scene(s,SimulatorSettings())
        try: assert check['quiescent'] and check['reset_physics_steps']==2 and check['clearances']['agent_block']>0
        finally: env.close()


def test_family_assignment_nested_counts_and_pilot_are_isolated():
    root=Path(__file__).parent;c=OmegaConf.load(root/'config/phase1/pusht.yaml').simulator.readout_repair
    f=family_assignments(c,{'development':{},'validation':{}},root)
    assert len(f)==64
    for count in (16,32,48):
        assert all(x['role']=='training' for x in f[:count])
        assert sum(x['stratum']=='near_contact' for x in f[:count])==count//2
    assert all(x['role']=='pilot_validation' for x in f[48:])
    assert len({x['family_id'] for x in f})==64


def test_related_poses_cannot_cross_families():
    first=family();second={**family(), 'family_id':'other','role':'pilot_validation'}
    try:check_family_isolation([first,second],[])
    except ValueError:pass
    else:raise AssertionError('Shared poses must be rejected')


def test_family_weights_prevent_frame_pseudoreplication():
    weights=family_weights(['a','a','b'])
    assert np.array_equal(weights,[.25,.25,.5])
    mean,std=target_statistics(np.array([[0.],[0.],[2.]]),weights)
    assert mean[0]==std[0]==1.
    try:target_statistics(np.zeros((3,1)),weights)
    except ValueError:pass
    else:raise AssertionError('Constant target cannot be epsilon repaired')


def test_weighted_mean_loss_ridge_matches_sklearn_without_whitening():
    rng=np.random.default_rng(42);x=rng.normal(size=(30,5));y=rng.normal(size=(30,2))
    w=family_weights(['a']*10+['b']*20)
    fitted=fit_ridge(x,y,w,.01)
    reference=Ridge(alpha=.01,solver='cholesky').fit(x,y,sample_weight=w)
    assert np.allclose(fitted['coef'],reference.coef_,atol=1e-10,rtol=0)
    assert np.allclose(fitted['intercept'],reference.intercept_,atol=1e-10,rtol=0)


def test_sincos_head_uses_correct_order_and_rejects_zero_vector():
    labels=np.zeros((3,5));labels[:,4]=[0.,np.pi/2,2*np.pi-.01]
    raw=target_values(labels,'angle_sincos')
    decoded,valid=decode_head(raw,'angle_sincos')
    assert valid.all() and np.allclose(decoded[:,0],labels[:,4])
    bad,valid=decode_head(np.array([[0.,0.],[1e-10,1e-10]]),'angle_sincos')
    assert not valid.any() and np.isnan(bad).all()


def test_differential_angle_wraps_effects_and_residuals():
    p=np.array([[[2*np.pi-.01],[.015],[2*np.pi-.02]]])
    t=np.array([[[2*np.pi-.01],[.01],[2*np.pi-.02]]])
    m=pair_metrics(p,t,'angle_scalar',[1,2],42,100)
    assert abs(m['rms']-.005/np.sqrt(2))<1e-10
    assert m['direction']['positive_fraction']==1.


def test_zero_change_has_no_fabricated_direction_or_error():
    p=np.array([[[100.,200.],[100.,200.]]]);m=pair_metrics(p,p,'agent_location',[1],42,100)
    assert m['rms']==0 and m['direction_undefined_count']==1 and 'direction' not in m


def test_scale_dependent_budget_is_not_universal_half_pixel_gate():
    metric=dict(status='pilot_conditional',lower95=.7,upper95=1.1)
    assert resolution_status(metric,.5)=='FAIL'
    assert resolution_status(metric,2.5)=='PASS'
    assert resolution_status(metric,1.)=='INCONCLUSIVE'
    assert resolution_status(dict(status='undefined'),5.)=='INCONCLUSIVE'


def test_paired_gain_uses_same_family_draws():
    result=paired_rms_gain([9.,16.],[1.,4.],42,1000)
    assert result['gain']>0 and result['lower95']>0 and result['n_families']==2
    assert result==paired_rms_gain([9.,16.],[1.,4.],42,1000)


def test_ridge_artifact_reload_preserves_training_transforms_and_lineage():
    x=np.zeros((4,192),np.float32);x[:,0]=np.arange(4)
    with TemporaryDirectory() as temp:
        path=Path(temp)/'r1.pt'
        torch.save(dict(schema_version='readout-r1-v1',official_instrument=False,method='ridge',head='agent_location',
            coef=torch.ones((2,192),dtype=torch.float64),intercept=torch.tensor([1.,2.],dtype=torch.float64),
            target_mean=torch.tensor([100.,200.]),target_std=torch.tensor([2.,3.]),training_family_ids=['train']),path)
        p=load_candidate_predictions(path,x)
        assert np.array_equal(p,[[102.,206.],[104.,209.],[106.,212.],[108.,215.]])


def test_mlp_training_is_bounded_reproducible_and_does_not_mutate_latents():
    rng=np.random.default_rng(42);x=rng.normal(size=(8,192)).astype(np.float32)
    labels=np.zeros((8,5));labels[:,:2]=x[:,:2]*5+100
    y=labels[:,:2];owners=np.array(['a']*4+['b']*4)
    section=OmegaConf.create(dict(mlp_lr=.001,mlp_weight_decay=.0001,mlp_epochs=20,mlp_patience=100,mlp_evaluate_every=10))
    original=x.copy();answers=[]
    for _ in range(2):
        probe,meta=fit_mlp(x,y,x,y,owners,owners,labels,labels,'agent_location',42,section,float('inf'))
        answers.append({k:v.clone() for k,v in probe.state_dict().items()})
        assert len(meta['curves'])==2 and meta['best_epoch']<=20
    assert np.array_equal(x,original)
    assert all(torch.equal(answers[0][k],answers[1][k]) for k in answers[0])


def test_position_row_space_rank_is_at_most_two_not_an_official_projector():
    w=np.zeros((2,192));w[0,0]=1;w[1,1]=1
    q,rank=row_space(w)
    assert rank==2 and q.shape==(192,2) and np.array_equal(q.T@q,np.eye(2))


if __name__=='__main__':
    tests=[fn for name,fn in sorted(globals().items()) if name.startswith('test_')]
    for test in tests:test();print(f'PASS {test.__name__}',flush=True)
    print(f'counterfactual readout feasibility tests passed ({len(tests)} tests)')
