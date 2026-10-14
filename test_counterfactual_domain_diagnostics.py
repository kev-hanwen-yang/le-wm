"""Direct-execution tests. Toy-model numbers are not scientific evidence."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import torch

from counterfactual.contracts import ActionNormalizer
from counterfactual.domain_diagnostics import (DOSES, H, bound_decision, calibration_from,
    effect_vectors, family_scenes, fixed_actions, gate_measurement, near_duplicate,
    physical_error, predict_natural, reserved_families, rms_bounds, validate_reservation)
from counterfactual.simulator import SimulatorSettings, create_fresh_scene
from scripts.run_phase1 import parse_args


def test_family_reservation_is_disjoint_and_all_doses_share_parent():
    section=OmegaConf.load(Path(__file__).parent/'config/phase1/pusht.yaml').simulator.domain_diagnostics
    families=reserved_families(section)
    assert len(families)==32 and families==reserved_families(section)
    validate_reservation(families,[])
    assert len(set(f['family_id'] for f in families))==32
    for f in families:
        scenes=family_scenes(f)
        assert len(scenes)==4 and all(s.seed==f['seed'] for s in scenes)
        assert all(s.goal_state==scenes[0].goal_state and s.block_position==scenes[0].block_position and s.block_angle==scenes[0].block_angle for s in scenes)
        assert np.allclose([np.linalg.norm(np.subtract(s.agent_position,scenes[0].agent_position)) for s in scenes],DOSES)


def test_cross_partition_overlap_is_rejected_including_related_doses():
    a=np.array([100.,100.,250.,230.,2*np.pi-.004,0.,0.])
    b=a.copy();b[4]=.004
    assert near_duplicate(a,b)
    b[0]+=2.;assert not near_duplicate(a,b)
    section=OmegaConf.create(dict(development_seed=42,validation_seed=43,families_per_partition=2))
    families=reserved_families(section)
    try: validate_reservation(families,[family_scenes(families[0])[2].public_state()])
    except ValueError: pass
    else: raise AssertionError('Historical pose overlap must fail')


def test_initial_labels_and_actual_clearances_are_independently_verified():
    section=OmegaConf.create(dict(development_seed=2026101401,validation_seed=2026101402,families_per_partition=2))
    for f in reserved_families(section):
        for scene in family_scenes(f):
            env,check=create_fresh_scene(scene,SimulatorSettings())
            try:
                assert np.allclose(env._get_obs(),scene.public_state(),atol=1e-10,rtol=0)
                assert check['clearances']['agent_block']>0 and check['quiescent']
                assert check['reset_physics_steps']==2
                assert np.array_equal(env.goal_pose,[350.,120.,0.])
            finally: env.close()


def test_fixed_actions_have_quiescent_context_and_exact_future_duration():
    f=dict(direction=[.6,.8])
    a=fixed_actions(f)
    assert a.shape==(35,2) and a.dtype==np.float32
    assert np.all(a[:10]==0) and np.all(a[25:]==0)
    assert np.allclose(a[10:25],[.06,.08])
    assert np.array_equal(H,[0,5,10,15,20,25,30,35])


class ToyModel:
    def action_encoder(self,tokens):
        out=torch.zeros((*tokens.shape[:2],192));out[:,:,0]=tokens[:,:,0];return out
    def eval(self): return self
    def predict(self,z,a): return z+a


def test_teacher_forcing_and_open_loop_use_sliding_aligned_seven_tokens():
    z=np.zeros((2,8,192),np.float32);z[:,:,0]=np.arange(8)*10
    actions=np.zeros((2,35,2),np.float32)
    for i in range(7): actions[:,i*5:(i+1)*5,0]=i+1
    normalizer=ActionNormalizer(mean=[0.,0.],std=[1.,1.])
    out=predict_natural(ToyModel(),z,actions,normalizer)
    assert out['standardized_tokens'].shape==(2,7,10)
    assert np.array_equal(out['open_loop'][0,:,0],[23,27,32,38,45])
    assert np.array_equal(out['teacher_forced'][0,:,0],[23,34,45,56,67])
    assert np.array_equal(out['open_loop'][:,0],out['teacher_forced'][:,0])


def test_raw_zero_is_not_standardized_zero_and_bad_histories_fail():
    norm=ActionNormalizer(mean=[.1,.2],std=[.2,.4])
    result=predict_natural(ToyModel(),np.zeros((1,8,192),np.float32),np.zeros((1,35,2),np.float32),norm)
    assert np.all(result['standardized_tokens']==-.5)
    try: predict_natural(ToyModel(),np.zeros((1,3,192)),np.zeros((1,35,2)),norm)
    except ValueError: pass
    else: raise AssertionError('Missing natural trajectory must fail')


def test_errors_use_euclidean_world_units_and_wrapped_angles():
    assert physical_error([3.,4.],[0.,0.],'agent_location')==5.
    assert abs(physical_error([2*np.pi-.01],[.01],'block_angle')-.02)<1e-10


def test_natural_effects_wrap_each_angle_effect_and_residual():
    real=np.array([[[2*np.pi-.01]],[[.02]]])
    pred=np.array([[[2*np.pi-.02]],[[.02]]])
    p,t,e=effect_vectors(pred,real,'block_angle')
    assert np.allclose(p,.04) and np.allclose(t,.03) and np.allclose(e,.01)
    real=np.array([[[0.,0.]],[[3.,4.]]]);pred=real.copy()
    p,t,e=effect_vectors(pred,real,'block_location');assert np.all(e==0)


def test_bootstrap_counts_families_and_has_two_sided_intervals():
    b=rms_bounds([1.,4.,9.])
    assert b['n_families']==3 and b['rms']==np.sqrt(14/3)
    assert b==rms_bounds([1.,4.,9.]) and b['ci95'][0]<=b['rms']<=b['ci95'][1]
    for bad in ([],[np.nan],[-1.]):
        try:rms_bounds(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid squared family values')


def test_uncertain_bounds_do_not_manufacture_pass():
    assert bound_decision(dict(lower95=1.,upper95=3.),2.)=='INCONCLUSIVE'
    assert bound_decision(dict(lower95=3.,upper95=4.),2.)=='FAIL'
    assert bound_decision(dict(lower95=0.,upper95=1.),2.)=='PASS'


def test_development_noise_cannot_enlarge_physical_caps():
    key='differential_agent_location_dose10'
    development={key:rms_bounds([100.,100.])}
    calibration=calibration_from(development)
    assert calibration[key]['cap']==5. and calibration[key]['margin']==10.
    gate=gate_measurement(development,calibration)
    assert gate['status']=='FAIL' and gate['checks'][key]['calibrated_status']=='PASS'
    assert gate['checks'][key]['cap_status']=='FAIL'


def test_cli_requires_explicit_diagnostic_partition_not_step3_stage():
    a=parse_args(['--stage','domain-diagnostics','--run-id','toy','--partition','validation','--calibration-manifest','parent.json'])
    assert a.partition=='validation' and a.calibration_manifest=='parent.json'


if __name__=='__main__':
    tests=[fn for name,fn in sorted(globals().items()) if name.startswith('test_')]
    for test in tests: test();print(f'PASS {test.__name__}',flush=True)
    print(f'counterfactual domain diagnostic tests passed ({len(tests)} tests)')
