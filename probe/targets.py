"""Target builders for physical-quantity probing tasks."""


# Component: supervised target construction.
#
# This module turns cached encoder outputs into supervised probe pairs. The
# embedding cache stores generic aligned data (`emb`, `state`, `proprio`,
# `episode_idx`, `step_idx`). Target builders choose one physical quantity from
# that cached data and return explicit tensors for probe training.
#
# Pair format returned by every builder:
# - embeddings: emb(N, 192)
# - target: the selected physical quantity, always kept 2D as (N, target_dim)
# - target_name: string used by training, checkpoint naming, and reporting
#
# Current targets:
# - agent_location: target = state[0:2] = (agent_x, agent_y), shape (N, 2).
# - block_location: target = state[2:4] = (block_x, block_y), shape (N, 2).
# - block_angle: target = state[4:5] = block angle, shape (N, 1).


def _build_target_pairs(encoded_data, target, target_name):
    embeddings = encoded_data["emb"].float()
    return {
        "embeddings": embeddings,
        "target": target.float(),
        "target_name": target_name,
        "episode_idx": encoded_data["episode_idx"],  # Metadata for traceability.
        "step_idx": encoded_data["step_idx"],  # Metadata for exact frame alignment.
    }


def build_agent_location_pairs(encoded_data):
    # Data format: embeddings(N, 192) <-> target agent_location(N, 2).
    agent_location = encoded_data["state"][:, 0:2].float()
    return _build_target_pairs(encoded_data, agent_location, "agent_location")


def build_block_location_pairs(encoded_data):
    # Data format: embeddings(N, 192) <-> target block_location(N, 2).
    block_location = encoded_data["state"][:, 2:4].float()
    return _build_target_pairs(encoded_data, block_location, "block_location")


def build_block_angle_pairs(encoded_data):
    # Data format: embeddings(N, 192) <-> target block_angle(N, 1).
    block_angle = encoded_data["state"][:, 4:5].float()
    return _build_target_pairs(encoded_data, block_angle, "block_angle")


TARGET_BUILDERS = {
    "agent_location": build_agent_location_pairs,
    "block_location": build_block_location_pairs,
    "block_angle": build_block_angle_pairs,
}
