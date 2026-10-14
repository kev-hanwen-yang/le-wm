"""Audited fresh-start Push-T replay, without replacing its dynamics.

Readbacks are NOT complete Chipmunk snapshots. The supported protocol constructs
a new environment, calls the installed reset once, and rejects nonquiescent or
contact-contaminated starts. No body corrections or extra settling steps occur.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

import numpy as np
import pymunk
from stable_worldmodel.envs.pusht.env import PushT


BODY_FIELDS = tuple(
    f"{body}_{field}"
    for body in ("agent", "block")
    for field in ("x", "y", "angle", "vx", "vy", "omega", "fx", "fy", "torque")
)
ANGLE_COLUMNS = (2, 11)


class SimulatorValidationError(ValueError):
    """A scene, action, backend, or replay violates the scientific contract."""


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="backslashreplace")
    if isinstance(value, float) and not math.isfinite(value):
        # Kinematic bodies have infinite mass/moment; never allow NaN readbacks.
        if math.isnan(value):
            raise SimulatorValidationError("NaN in audit metadata")
        return "inf" if value > 0 else "-inf"
    return value


def canonical_json(value) -> str:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def wrap_angle(value):
    return np.arctan2(np.sin(value), np.cos(value))


@dataclass(frozen=True)
class SceneSpec:
    """A quiescent scene, in 512-world-coordinate px and radians."""

    scene_id: str
    seed: int
    agent_position: tuple[float, float]
    block_position: tuple[float, float]
    block_angle: float
    goal_state: tuple[float, ...] = (70., 420., 350., 120., 0., 0., 0.)

    def __post_init__(self):
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise SimulatorValidationError("scene_id must be nonempty")
        if type(self.seed) is not int or self.seed < 0:
            raise SimulatorValidationError("Scene seed must be an explicit nonnegative integer")
        for name, size in (("agent_position", 2), ("block_position", 2), ("goal_state", 7)):
            if np.iscomplexobj(getattr(self, name)):
                raise SimulatorValidationError(f"{name} must be real-valued")
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise SimulatorValidationError(f"{name} must contain {size} finite values")
            object.__setattr__(self, name, tuple(float(item) for item in value))
        if not math.isfinite(float(self.block_angle)):
            raise SimulatorValidationError("Block angle must be finite")
        if self.goal_state[-2:] != (0., 0.):
            raise SimulatorValidationError("Goal initialization must also be quiescent")

    def public_state(self) -> np.ndarray:
        return np.array((*self.agent_position, *self.block_position, self.block_angle, 0., 0.), dtype=np.float64)

    def reset_options(self) -> dict:
        # Match setup and explicit state; disable all implicit variation sampling.
        return {
            "variation": [],
            "variation_values": {
                "agent.start_position": np.array(self.agent_position, dtype=np.float64),
                "block.start_position": np.array(self.block_position, dtype=np.float64),
                "block.angle": float(self.block_angle),
                "goal.position": np.array(self.goal_state[2:4], dtype=np.float64),
                "goal.angle": float(self.goal_state[4]),
            },
            "state": self.public_state(),
            "goal_state": np.array(self.goal_state, dtype=np.float64),
        }


@dataclass(frozen=True)
class SimulatorSettings:
    resolution: int = 224
    with_target: bool = True
    atol: float = 1e-10

    def __post_init__(self):
        if type(self.resolution) is not int or self.resolution < 1:
            raise SimulatorValidationError("Resolution must be a positive integer")
        if type(self.with_target) is not bool:
            raise SimulatorValidationError("with_target must be explicit boolean")
        if not math.isfinite(self.atol) or self.atol < 0:
            raise SimulatorValidationError("atol must be finite and nonnegative")

    def env_kwargs(self):
        # Leave all physics arguments at their installed defaults.
        return dict(resolution=self.resolution, with_target=self.with_target,
                    relative=True, render_action=False, render_mode="rgb_array")


def body_vector(env) -> np.ndarray:
    values = []
    for body in (env.agent, env.block):
        values.extend((*body.position, body.angle, *body.velocity,
                       body.angular_velocity, *body.force, body.torque))
    result = np.array(values, dtype=np.float64)
    if result.shape != (18,) or not np.isfinite(result).all():
        raise SimulatorValidationError("Nonfinite/incompatible physical body readback")
    return result


def rng_snapshot(env) -> dict:
    result = {"env_rng": env.rng.bit_generator.state,
              "gym_rng": env.np_random.bit_generator.state}

    def visit(space, path):
        result[path] = space.np_random.bit_generator.state
        for name, child in getattr(space, "spaces", {}).items():
            visit(child, f"{path}.{name}")

    visit(env.variation_space, "variation")
    return jsonable(result)


def physical_readback(env) -> dict:
    state = env._get_obs().copy()
    goal = np.array(env.goal_state, dtype=np.float64, copy=True)
    goal_pose = np.array(env.goal_pose, dtype=np.float64, copy=True)
    if state.shape != (7,) or goal.shape != (7,) or goal_pose.shape != (3,):
        raise SimulatorValidationError("Incompatible observation/goal shape")
    if not all(np.isfinite(item).all() for item in (state, goal, goal_pose)):
        raise SimulatorValidationError("Nonfinite observation/goal")
    rng = rng_snapshot(env)
    action = np.zeros(2) if env.latest_action is None else np.asarray(env.latest_action).copy()
    return dict(body=body_vector(env), public_state=state,
                proprio=np.concatenate((state[:2], state[-2:])), goal_state=goal,
                goal_pose=goal_pose, latest_action=action, action_present=env.latest_action is not None,
                legacy_contact_points=int(env.n_contact_points),
                rng_hash=hashlib.sha256(canonical_json(rng).encode()).hexdigest())


def shape_descriptor(shape):
    result = {"kind": type(shape).__name__, "radius": float(shape.radius),
              "friction": float(shape.friction), "elasticity": float(shape.elasticity),
              "collision_type": int(shape.collision_type), "sensor": bool(shape.sensor),
              "filter": list(shape.filter), "surface_velocity": list(shape.surface_velocity)}
    if isinstance(shape, pymunk.Poly):
        result["vertices"] = [list(point) for point in shape.get_vertices()]
    elif isinstance(shape, pymunk.Circle):
        result["offset"] = list(shape.offset)
    elif isinstance(shape, pymunk.Segment):
        result.update(a=list(shape.a), b=list(shape.b))
    else:
        raise SimulatorValidationError("Unaudited shape type")
    return result


def simulator_contract(env) -> dict:
    """Actual geometry/solver/controller/render values, not guessed constants."""
    bodies = {}
    for name, body in (("agent", env.agent), ("block", env.block)):
        bodies[name] = dict(body_type=int(body.body_type), mass=float(body.mass),
                            moment=float(body.moment), cog=list(body.center_of_gravity),
                            shapes=sorted([shape_descriptor(s) for s in body.shapes], key=canonical_json))
    walls = sorted([shape_descriptor(s) for s in env.space.shapes if s.body is env.space.static_body], key=canonical_json)
    return jsonable(dict(
        controller={name: getattr(env, name) for name in ("dt", "control_hz", "k_p", "k_v", "action_scale", "relative")},
        raw_action_dtype="float32", raw_action_bounds=[env.action_space.low, env.action_space.high],
        physics_substeps_per_raw_action=int(1 / (env.dt * env.control_hz)),
        physics=dict(gravity=list(env.space.gravity), damping=env.space.damping,
                     iterations=env.space.iterations, collision_slop=env.space.collision_slop,
                     collision_bias=env.space.collision_bias, collision_persistence=env.space.collision_persistence,
                     sleep_time_threshold=env.space.sleep_time_threshold,
                     idle_speed_threshold=env.space.idle_speed_threshold,
                     threaded=env.space.threaded, constraints=len(env.space.constraints)),
        bodies=bodies, walls=walls,
        static_body=dict(position=list(env.space.static_body.position), angle=env.space.static_body.angle,
                         velocity=list(env.space.static_body.velocity), omega=env.space.static_body.angular_velocity),
        variation=env.variation_space.value,
        rendering=dict(resolution=env.render_size, world_size=env.window_size,
                       with_target=env.with_target, render_action=env.render_action,
                       render_goal=bool(env.variation_space["rendering"]["render_goal"].value),
                       mode=env.render_mode),
        goal_state=env.goal_state, goal_pose=env.goal_pose,
        pymunk_version=pymunk.version, chipmunk_version=pymunk.chipmunk_version,
        readback_fields=BODY_FIELDS, angles="unwrapped body radians; circular comparison",
        hidden_collision_cache="not captured or restorable",
    ))


class AuditedPushT(PushT):
    """Observe existing stepping/post-solve callbacks; never change responses."""

    def __init__(self, **kwargs):
        self.audit_phase = "reset"
        self.raw_step = 0
        self.physics_step_index = 0
        self.physics_states = []
        self.physics_dt = []
        self.physics_raw_steps = []
        self.contact_events = []
        self.audit_errors = []
        super().__init__(**kwargs)

    def _setup(self):
        super()._setup()
        original_step = self.space.step

        def observed_step(dt):
            self.physics_step_index += 1
            original_step(dt)
            self.physics_states.append(body_vector(self))
            self.physics_dt.append(float(dt))
            self.physics_raw_steps.append(self.raw_step)

        self.space.step = observed_step

    def _handle_collision(self, arbiter, space, data):
        result = super()._handle_collision(arbiter, space, data)
        try:
            roles = ["agent" if s.body is self.agent else "block" if s.body is self.block else "wall"
                     if s.body is self.space.static_body else "unknown" for s in arbiter.shapes]
            pair = "-".join(sorted(roles))
            self.contact_events.append(dict(
                phase=self.audit_phase, raw_step=self.raw_step,
                physics_step=self.physics_step_index, pair=pair,
                point_count=len(arbiter.contact_point_set.points),
                points=[dict(a=list(p.point_a), b=list(p.point_b), distance=float(p.distance))
                        for p in arbiter.contact_point_set.points],
                normal=list(arbiter.normal), impulse=list(arbiter.total_impulse),
                first_contact=bool(arbiter.is_first_contact),
            ))
        except Exception as error:
            # CFFI can swallow callback exceptions; surface them after stepping.
            self.audit_errors.append(f"{type(error).__name__}: {error}")
        return result


def geometry_clearances(env) -> dict:
    circles = list(env.agent.shapes)
    if len(circles) != 1 or not isinstance(circles[0], pymunk.Circle):
        raise SimulatorValidationError("Only the released circular pusher is audited")
    agent_block = min(s.point_query(env.agent.position).distance - circles[0].radius
                      for s in env.block.shapes)
    walls = [s for s in env.space.shapes if s.body is env.space.static_body]
    vertical = [s for s in walls if s.a.x == s.b.x]
    horizontal = [s for s in walls if s.a.y == s.b.y]
    if len(vertical) != 2 or len(horizontal) != 2:
        raise SimulatorValidationError("Unaudited workspace wall geometry")
    left, right = sorted(vertical, key=lambda s: s.a.x)
    bottom, top = sorted(horizontal, key=lambda s: s.a.y)
    bounds = (left.a.x + left.radius, right.a.x - right.radius,
              bottom.a.y + bottom.radius, top.a.y - top.radius)
    clearances = {}
    for name, body in (("agent", env.agent), ("block", env.block)):
        clearances[name] = min(min(s.bb.left - bounds[0], bounds[1] - s.bb.right,
                                  s.bb.bottom - bounds[2], bounds[3] - s.bb.top) for s in body.shapes)
    return dict(agent_block=float(agent_block), wall=clearances, inner_bounds=list(bounds))


def validate_initial_readback(env, scene: SceneSpec, settings: SimulatorSettings) -> dict:
    read = physical_readback(env)
    target = scene.public_state()
    difference = read["public_state"] - target
    difference[4] = wrap_angle(difference[4])
    if np.any(np.abs(difference) > settings.atol):
        raise SimulatorValidationError(f"Reset changed intended scene: {difference.tolist()}")
    vector = read["body"]
    dynamic_columns = [3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]
    if np.any(np.abs(vector[dynamic_columns]) > settings.atol) or abs(vector[2]) > settings.atol:
        raise SimulatorValidationError("Reset is not quiescent: omitted velocity/force/angular state")
    if not np.array_equal(read["goal_state"], np.asarray(scene.goal_state)):
        raise SimulatorValidationError("Goal state changed during reset")
    if not np.array_equal(read["goal_pose"], np.asarray(scene.goal_state[2:5])):
        raise SimulatorValidationError("Rendered goal pose does not match explicit physical goal")
    if read["action_present"] or read["legacy_contact_points"]:
        raise SimulatorValidationError("Reset contains stale controller action or contacts")
    clearance = geometry_clearances(env)
    if clearance["agent_block"] <= settings.atol or min(clearance["wall"].values()) <= settings.atol:
        raise SimulatorValidationError(f"Initial contact/penetration/bounds violation: {clearance}")
    if isinstance(env, AuditedPushT):
        if env.audit_errors or env.contact_events:
            raise SimulatorValidationError("Reset goal/scene stepped through contacts")
        if env.physics_step_index != 2:
            raise SimulatorValidationError("Installed reset changed: expected exactly two audited physics steps")
        goal_read = env.physics_states[0]
        expected_goal = np.array(scene.goal_state)
        goal_difference = goal_read[[0, 1, 9, 10, 11]] - expected_goal[:5]
        goal_difference[4] = wrap_angle(goal_difference[4])
        if np.any(np.abs(goal_difference) > settings.atol) or np.any(np.abs(goal_read[dynamic_columns]) > settings.atol):
            raise SimulatorValidationError("Goal's reset physics step was not quiescent/exact")
    contract = simulator_contract(env)
    expected_control = dict(dt=.01, control_hz=10, k_p=100, k_v=20, action_scale=100, relative=True)
    if contract["controller"] != expected_control or contract["physics_substeps_per_raw_action"] != 10:
        raise SimulatorValidationError("Released controller/time contract changed")
    if contract["physics"]["constraints"] != 0:
        raise SimulatorValidationError("Uncaptured constraints in fresh scene")
    return dict(clearances=clearance, intended_state_max_abs=float(np.max(np.abs(difference))),
                quiescent=True, reset_physics_steps=2 if isinstance(env, AuditedPushT) else "uninstrumented")


def create_fresh_scene(scene: SceneSpec, settings: SimulatorSettings, *, audited=True):
    """Return owned environment; caller closes it. Never correct a bad reset."""
    factory = AuditedPushT if audited else PushT
    env = factory(**settings.env_kwargs())
    try:
        env.reset(seed=scene.seed, options=scene.reset_options())
        validation = validate_initial_readback(env, scene, settings)
        return env, validation
    except AssertionError as error:
        env.close()
        raise SimulatorValidationError(f"Scene/reset options violate installed variation bounds: {error}") from error
    except Exception:
        env.close()
        raise


def validate_actions(actions) -> np.ndarray:
    """Released action-space dtype, no clipping or numerical repairs."""
    array = np.asarray(actions)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 1:
        raise SimulatorValidationError("Actions require nonempty (T,2) shape")
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all() or np.any(np.abs(array) > 1):
        raise SimulatorValidationError("Raw relative commands must be finite and in [-1,1]")
    return np.array(array, dtype=np.float32, copy=True)


def replay_fixed_actions(scene: SceneSpec, actions, settings: SimulatorSettings, *, audited=True, deadline=None) -> dict:
    commands = validate_actions(actions)
    env, initial = create_fresh_scene(scene, settings, audited=audited)
    try:
        metadata = simulator_contract(env)
        initial_rng = rng_snapshot(env)
        reads, images, rewards, terminated, truncated, controller_targets = [], [], [], [], [], []

        def record():
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Simulator validation exceeded its configured hard timeout")
            reads.append(physical_readback(env))
            image = np.array(env.render(), copy=True)
            if image.shape != (settings.resolution, settings.resolution, 3) or image.dtype != np.uint8:
                raise SimulatorValidationError("Renderer violates uint8 HWC contract")
            images.append(image)

        record()
        for index, command in enumerate(commands, start=1):
            if audited:
                env.audit_phase, env.raw_step = "rollout", index
            controller_targets.append(np.asarray(env.agent.position) + command * env.action_scale)
            observation, reward, term, trunc, _ = env.step(command.copy())
            if not np.array_equal(observation["state"], env._get_obs()):
                raise SimulatorValidationError("Returned observation and body readback disagree")
            if not np.array_equal(observation["proprio"], np.concatenate((observation["state"][:2], observation["state"][-2:]))):
                raise SimulatorValidationError("Returned proprio and state disagree")
            rewards.append(reward); terminated.append(term); truncated.append(trunc)
            record()
        result = dict(scene=asdict(scene), initial_validation=initial, metadata=metadata,
                      actions=commands, raw_steps=np.arange(len(reads)),
                      images=np.stack(images), rewards=np.array(rewards, dtype=np.float64),
                      terminated=np.array(terminated), truncated=np.array(truncated),
                      controller_target_world=np.stack(controller_targets),
                      rng_initial=initial_rng, rng_final=rng_snapshot(env))
        result.update({name: np.stack([read[name] for read in reads]) for name in reads[0]})
        result["image_sha256"] = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
        if audited:
            if env.audit_errors:
                raise SimulatorValidationError(f"Contact audit failed: {env.audit_errors}")
            expected_steps = 2 + len(commands) * 10
            if env.physics_step_index != expected_steps:
                raise SimulatorValidationError("Raw action did not execute exactly ten physics substeps")
            expected_raw_steps = np.r_[0, 0, np.repeat(np.arange(1, len(commands) + 1), 10)]
            if not np.array_equal(env.physics_raw_steps, expected_raw_steps) or not np.all(np.asarray(env.physics_dt) == env.dt):
                raise SimulatorValidationError("Physics substep timing/count differs within a raw action")
            result.update(physics_states=np.stack(env.physics_states),
                          physics_dt=np.array(env.physics_dt),
                          physics_raw_steps=np.array(env.physics_raw_steps),
                          contact_events=env.contact_events)
            counts = np.zeros((len(commands) + 1, 3), dtype=np.int64)
            for event in env.contact_events:
                if event["pair"] not in ("agent-block", "block-wall", "agent-wall"):
                    raise SimulatorValidationError(f"Unknown contact pair: {event['pair']}")
                counts[event["raw_step"], ("agent-block", "block-wall", "agent-wall").index(event["pair"])] += event["point_count"]
            result["typed_contact_points"] = counts
            if not np.array_equal(counts.sum(axis=1), result["legacy_contact_points"]):
                raise SimulatorValidationError("Observer changed/missed original contact point count")
        return result
    finally:
        env.close()


def compare_replays(left: dict, right: dict, atol: float, *, require_audit=True) -> dict:
    """Compare every field independently; no mixed-unit physical distance."""
    if not math.isfinite(atol) or atol < 0:
        raise SimulatorValidationError("Comparison tolerance must be finite/nonnegative")
    checks, deltas = {}, {}
    numeric = ("body", "public_state", "proprio", "goal_state", "goal_pose", "latest_action", "rewards", "controller_target_world")
    if require_audit:
        numeric += ("physics_states", "physics_dt")
    for key in numeric:
        a, b = np.asarray(left[key]), np.asarray(right[key])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            checks[key], deltas[key] = False, "shape/nonfinite"
            continue
        delta = a - b
        if key in ("body", "physics_states"):
            delta[..., list(ANGLE_COLUMNS)] = wrap_angle(delta[..., list(ANGLE_COLUMNS)])
            deltas[key] = dict(zip(BODY_FIELDS, np.max(np.abs(delta), axis=0).tolist()))
        else:
            if key in ("public_state", "goal_state"):
                delta[..., 4] = wrap_angle(delta[..., 4])
            elif key == "goal_pose":
                delta[..., 2] = wrap_angle(delta[..., 2])
            deltas[key] = float(np.max(np.abs(delta))) if delta.size else 0.
        checks[key] = bool(np.all(np.abs(delta) <= atol))
    exact = ("scene", "metadata", "actions", "raw_steps", "action_present", "terminated", "truncated",
             "legacy_contact_points", "rng_hash", "rng_initial", "rng_final", "image_sha256")
    if require_audit:
        exact += ("physics_raw_steps", "typed_contact_points", "contact_events")
    for key in exact:
        checks[key] = canonical_json(left[key]) == canonical_json(right[key])
    checks["pixels"] = np.array_equal(left["images"], right["images"])
    return dict(passed=all(checks.values()), checks=checks, max_abs_by_field=deltas,
                atol=atol, rtol=0., raw_state_count=len(left["raw_steps"]),
                physics_state_count=len(left.get("physics_states", [])))


def restore_dynamic_state(*args, **kwargs):
    """Deliberately not an approximate implementation of arbitrary restoration."""
    raise SimulatorValidationError(
        "Arbitrary dynamic restore is unsupported: 7D state omits block dynamics, "
        "and Chipmunk contact/solver caches are not captured. Use verified fresh quiescent starts."
    )
