"""Target builders for physical-quantity probing tasks."""


# Component: supervised target construction.
#
# This module turns cached encoder outputs into supervised probe pairs. The
# embedding cache stores generic aligned data (`emb`, `state`, `proprio`,
# `episode_idx`, `step_idx`). Target builders choose one physical quantity from
# that cached data and return explicit tensors for probe training.
#
# Current target:
# - agent_location: state[0:2] = (agent_x, agent_y), paired with emb(N, 192).


def build_agent_location_pairs(encoded_data):
    # Data format: embeddings(N, 192) <-> agent_location(N, 2).
    embeddings = encoded_data["emb"].float()
    agent_location = encoded_data["state"][:, 0:2].float()
    return {
        "embeddings": embeddings,
        "agent_location": agent_location,
        "target_name": "agent_location",
        "episode_idx": encoded_data["episode_idx"],  # Metadata for traceability.
        "step_idx": encoded_data["step_idx"],  # Metadata for exact frame alignment.
    }
