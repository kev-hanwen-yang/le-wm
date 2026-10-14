"""Direct-execution R0 tests; toy numbers are not scientific findings."""
from pathlib import Path
import numpy as np
import torch
from counterfactual.readout_audit import absolute_summary,geometry_audit,polygon_area_centroid,verify_original_readouts
from counterfactual.simulator import SceneSpec,SimulatorSettings,create_fresh_scene
from probe.models import MLPRegressionProbe
from scripts.run_phase1 import parse_args


def test_polygon_area_centroid_is_not_body_origin():
    area,c=polygon_area_centroid([[-60,0],[60,0],[60,30],[-60,30]])
    assert area==3600. and np.array_equal(c,[0,15])
    a2,c2=polygon_area_centroid([[-15,30],[15,30],[15,120],[-15,120]])
    assert a2==2700. and np.array_equal(c2,[0,75])
    assert abs(np.average([c,c2],axis=0,weights=[area,a2])[1]-40.714285714285715)<1e-12


def test_circle_and_rotated_block_geometry_agree_without_probes():
    for angle in (0.,np.pi/4,1.,2*np.pi-.01):
        scene=SceneSpec('r0_test',20261015,(100.,100.),(256.,230.),angle)
        env,_=create_fresh_scene(scene,SimulatorSettings())
        try:
            result=geometry_audit(env,env.render())
            assert result['analytic_transform_max_abs']<1e-10
            assert result['canvas_resize_exact']
            assert np.allclose(result['block_body_origin'],[256.,230.],atol=1e-10,rtol=0)
            assert abs(np.linalg.norm(np.subtract(result['block_world_mass_center'],result['block_body_origin']))-45.)<1e-10
            assert all(r['missing_deep_interior']==r['outside_boundary_band']==0 for r in result['raster'].values())
        finally: env.close()


def test_geometry_rejects_rgb_swap_or_image_misalignment():
    env,_=create_fresh_scene(SceneSpec('bad_rgb',42,(100.,100.),(256.,230.),0.),SimulatorSettings())
    try:
        image=env.render()
        for bad in (image[...,::-1],np.roll(image,10,axis=0)):
            try:geometry_audit(env,bad)
            except ValueError:pass
            else:raise AssertionError('Misordered/misaligned images must fail')
    finally: env.close()


def test_independent_original_mlp_forward_and_normalization_contract():
    torch.manual_seed(42);p=MLPRegressionProbe(192,256,2).eval()
    loaded={'agent_location':dict(probe=p,target_name='agent_location',probe_type='mlp',
        target_mean=torch.tensor([[200.,250.]]),target_std=torch.tensor([[50.,70.]]),payload={'probe_state_dict':p.state_dict()})}
    result=verify_original_readouts(np.zeros((4,192),np.float32),loaded)
    assert result['agent_location']['manual_forward_max_abs']==0.
    p.train()
    try:verify_original_readouts(np.zeros((4,192),np.float32),loaded)
    except ValueError:pass
    else:raise AssertionError('Training dropout must not be accepted')


def test_position_coordinate_and_euclidean_definitions_and_family_balance():
    owners=['a','a','b'];truth=np.zeros((3,5))
    pred={'agent_location':np.array([[3,4],[3,4],[0,0]]), 'block_location':np.zeros((3,2)), 'block_angle':np.zeros((3,1))}
    result=absolute_summary(pred,truth,owners)['agent_location']
    assert abs(result['rms']-np.sqrt(12.5))<1e-12
    assert np.allclose(result['coordinate_rmse'],[3/np.sqrt(2),4/np.sqrt(2)])
    assert abs(result['coordinate_pooled_rmse']-2.5)<1e-12


def test_angle_errors_wrap_without_degree_conversion():
    pred={'agent_location':np.zeros((1,2)), 'block_location':np.zeros((1,2)), 'block_angle':np.array([[2*np.pi-.01]])}
    truth=np.array([[0,0,0,0,.01]])
    assert abs(absolute_summary(pred,truth,['a'])['block_angle']['rms']-.02)<1e-10


def test_cli_has_separate_audit_and_feasibility_gate():
    args=parse_args(['--stage','readout-feasibility','--run-id','toy','--audit-manifest','r0.json'])
    assert args.audit_manifest=='r0.json'
    assert parse_args(['--stage','readout-audit','--run-id','toy']).stage=='readout-audit'


if __name__=='__main__':
    tests=[fn for name,fn in sorted(globals().items()) if name.startswith('test_')]
    for test in tests: test();print(f'PASS {test.__name__}',flush=True)
    print(f'counterfactual readout audit tests passed ({len(tests)} tests)')
