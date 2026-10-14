# Phase I Step 1 — Frozen baseline

- Run status: `failed`
- Step 1 gate passed: `False`
- Git revision: `597cc334fd68b2f455c4a1ab007224faccb378f9`
- Checkpoint SHA-256: `100d6459b9198697c1c26ef47e3eef4d7548946401b7501213caf761495a2553`
- Device: `cpu`
- Repeated encoding max delta: `not run`
- Repeated prediction max delta: `not run`
- Repeated five-step rollout max delta: `not run`

## Cached-embedding identity

| Split | Status | Max abs. difference | RMSE |
|---|---|---|---|

## Official CEM smoke

- Status: `not_run`
- Return code: `n/a`
- Requested episodes: `10`
- Reported success rate (%): `n/a`

This is a functionality gate. Simulator replay is the separate Step 2 gate.
The official evaluator resets without dataset seeds; process/CEM seeding alone does not prove simulator determinism.

## Failure

`ValueError`: First rollout output disagrees with one-step prediction
