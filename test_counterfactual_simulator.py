"""Direct-execution mechanics tests against the installed, unchanged Push-T."""

from copy import deepcopy
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

import numpy as np

from counterfactual.simulator import (
    AuditedPushT, SceneSpec, SimulatorSettings, SimulatorValidationError,
    body_vector, compare_replays, create_fresh_scene, physical_readback,
    replay_fixed_actions, restore_dynamic_state, simulator_contract, validate_actions,
)
from counterfactual.simulator_validation import (
    audit_unsafe_public_restore, validate_case_behavior, validation_cases,
)


def scene():
    return SceneSpec("fixture", 20261014, (130., 240.), (256., 230.), 0.)


def test_scene_inputs_are_finite_explicit_quiescent_and_immutable():
    for changes in ({"seed": None}, {"seed": -1}, {"seed": True},
                    {"agent_position": (1.,)}, {"block_position": (np.nan, 230.)},
                    {"block_angle": float("inf")}, {"goal_state": (70., 420., 350., 120., 0., 1., 0.)}):
        with unittest.TestCase().assertRaises(SimulatorValidationError):
            replace(scene(), **changes)
    original = np.array([130., 240.])
    spec = replace(scene(), agent_position=original)
    original[0] = 200.
    assert spec.agent_position == (130., 240.)


def test_fresh_reset_has_two_steps_and_complete_quiescent_readback():
    env, validation = create_fresh_scene(scene(), SimulatorSettings())
    try:
        assert validation["reset_physics_steps"] == env.physics_step_index == 2
        read = physical_readback(env)
        assert read["body"].shape == (18,)
        assert read["public_state"].shape == (7,)
        assert np.array_equal(read["public_state"], scene().public_state())
        assert np.array_equal(read["goal_pose"], scene().goal_state[2:5])
        assert read["legacy_contact_points"] == 0 and not read["action_present"]
        contract = simulator_contract(env)
        assert contract["physics_substeps_per_raw_action"] == 10
        assert contract["physics"]["damping"] == 0
        assert contract["bodies"]["block"]["cog"] == [0., 45.]
        assert contract["bodies"]["agent"]["shapes"][0]["radius"] == 15.
        assert contract["hidden_collision_cache"] == "not captured or restorable"
    finally:
        env.close()


def test_overlap_tangency_walls_and_goal_contact_are_rejected():
    invalid = (
        replace(scene(), agent_position=(256., 245.)),
        replace(scene(), agent_position=(181., 245.)),
        replace(scene(), agent_position=(20., 50.)),
        replace(scene(), block_position=(100., 100.), block_angle=np.pi / 2),
        replace(scene(), goal_state=(256., 245., 256., 230., 0., 0., 0.)),
    )
    for spec in invalid:
        with unittest.TestCase().assertRaises(SimulatorValidationError):
            create_fresh_scene(spec, SimulatorSettings())


def test_bad_reset_is_rejected_not_corrected_or_settled():
    original = AuditedPushT._set_state
    calls = []

    def contaminated(env, state):
        calls.append(np.array(state))
        original(env, state)
        env.block.velocity = (1., 0.)

    with patch.object(AuditedPushT, "_set_state", contaminated):
        with unittest.TestCase().assertRaises(SimulatorValidationError):
            create_fresh_scene(scene(), SimulatorSettings())
    assert len(calls) == 2


def test_actions_reject_invalid_inputs_without_clipping_or_input_mutation():
    for commands in (np.zeros((0, 2)), np.zeros((2, 3)), np.ones((3, 2)) * 1.01,
                     np.full((3, 2), np.nan), np.zeros(2), np.full((2, 2), .2j),
                     np.full((2, 2), True)):
        with unittest.TestCase().assertRaises(SimulatorValidationError):
            validate_actions(commands)
    values = np.array([[.2, -.1]], dtype=np.float64)
    assert validate_actions(values).dtype == np.float32
    assert np.array_equal(values, [[.2, -.1]])


def test_all_mechanics_cases_replay_and_observer_preserves_unmodified_dynamics():
    settings = SimulatorSettings()
    for spec, actions, expected in validation_cases(20261014):
        first = replay_fixed_actions(spec, actions, settings)
        second = replay_fixed_actions(spec, actions, settings)
        plain = replay_fixed_actions(spec, actions, settings, audited=False)
        assert compare_replays(first, second, settings.atol)["passed"], spec.scene_id
        assert compare_replays(first, plain, settings.atol, require_audit=False)["passed"], spec.scene_id
        assert validate_case_behavior(first, expected, settings.atol)["passed"], spec.scene_id
        assert first["body"].shape == (36, 18)
        assert first["physics_states"].shape == (352, 18)
        assert np.array_equal(first["physics_raw_steps"][:2], [0, 0])
        assert np.array_equal(first["physics_raw_steps"][2:], np.repeat(np.arange(1, 36), 10))
        assert np.array_equal(first["typed_contact_points"].sum(axis=1), first["legacy_contact_points"])


def test_both_goal_render_modes_repeat_but_are_not_identical_images():
    commands = np.zeros((1, 2), dtype=np.float32)
    shown = replay_fixed_actions(scene(), commands, SimulatorSettings(with_target=True))
    hidden = replay_fixed_actions(scene(), commands, SimulatorSettings(with_target=False))
    hidden_repeat = replay_fixed_actions(scene(), commands, SimulatorSettings(with_target=False))
    assert compare_replays(hidden, hidden_repeat, 1e-10)["passed"]
    assert np.array_equal(shown["body"], hidden["body"])
    assert not np.array_equal(shown["images"], hidden["images"])


def test_related_scene_fixture_shares_goal_style_and_physics():
    settings = SimulatorSettings()
    a = replace(scene(), scene_id="A", agent_position=(120., 245.))
    b = replace(a, scene_id="B", agent_position=(140., 245.), block_position=(256., 290.))
    c = replace(a, scene_id="C", agent_position=(140., 245.))
    records = [replay_fixed_actions(spec, np.zeros((10, 2)), settings) for spec in (a, b, c)]
    for record in records:
        assert np.array_equal(record["body"], np.broadcast_to(record["body"][0], (11, 18)))
        assert np.array_equal(record["goal_state"], records[0]["goal_state"])
        assert record["metadata"]["physics"] == records[0]["metadata"]["physics"]
        assert record["metadata"]["bodies"] == records[0]["metadata"]["bodies"]
        assert record["metadata"]["rendering"] == records[0]["metadata"]["rendering"]
    assert np.array_equal(records[0]["body"][:, 9:], records[2]["body"][:, 9:])
    assert np.array_equal(records[1]["body"][:, :9], records[2]["body"][:, :9])


def test_comparison_catches_nonfinal_state_contact_rng_image_and_action_corruption():
    commands = np.zeros((2, 2), dtype=np.float32)
    trace = replay_fixed_actions(scene(), commands, SimulatorSettings())
    for corruption in ("velocity", "nan", "substep", "pixels", "contact", "rng", "action", "physics"):
        other = deepcopy(trace)
        if corruption == "velocity":
            other["body"][1, 12] += .01
        elif corruption == "nan":
            other["body"][1, 12] = np.nan
        elif corruption == "substep":
            other["physics_states"][3, 0] += .01
        elif corruption == "pixels":
            other["images"][1, 0, 0, 0] = 0
        elif corruption == "contact":
            other["contact_events"].append({"pair": "agent-block"})
        elif corruption == "rng":
            other["rng_hash"][1] = "different"
        elif corruption == "action":
            other["actions"][0, 0] = .1
        else:
            other["metadata"]["physics"]["damping"] = 1.
        assert not compare_replays(trace, other, 1e-10)["passed"], corruption


def test_public_dynamic_restore_fails_and_production_api_rejects_it():
    settings = SimulatorSettings()
    for spec, actions, expected in validation_cases(20261014):
        if expected not in ("free_motion", "agent_block"):
            continue
        trace = replay_fixed_actions(spec, actions, settings)
        audit = audit_unsafe_public_restore(trace, spec, settings)
        assert audit["set_state_extra_physics_steps"] == 1
        assert not audit["matches_reference"]
        assert not audit["supported"]
        if expected == "agent_block":
            assert np.linalg.norm(audit["omitted_source_block_velocity"]) > 0
    with unittest.TestCase().assertRaises(SimulatorValidationError):
        restore_dynamic_state(np.zeros(7))


def test_deadline_and_comparison_tolerances_fail_closed():
    with unittest.TestCase().assertRaises(TimeoutError):
        replay_fixed_actions(scene(), np.zeros((1, 2)), SimulatorSettings(), deadline=time.monotonic() - 1)
    with unittest.TestCase().assertRaises(SimulatorValidationError):
        compare_replays({}, {}, float("nan"))


if __name__ == "__main__":
    tests = [function for name, function in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}", flush=True)
    print(f"counterfactual simulator tests passed ({len(tests)} tests)")
