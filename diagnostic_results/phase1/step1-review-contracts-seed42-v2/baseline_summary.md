# Phase I Step 1 — Frozen baseline

- Run status: `verification_only`
- Step 1 gate passed: `False`
- Git revision: `597cc334fd68b2f455c4a1ab007224faccb378f9`
- Checkpoint SHA-256: `100d6459b9198697c1c26ef47e3eef4d7548946401b7501213caf761495a2553`
- Device: `cpu`
- Repeated encoding max delta: `0.0`
- Repeated prediction max delta: `0.0`
- Repeated five-step rollout max delta: `0.0`

## Cached-embedding identity

| Split | Status | Max abs. difference | RMSE |
|---|---|---|---|
| train | match | 3.337860107421875e-06 | 7.971741524670506e-07 |
| val | match | 3.337860107421875e-06 | 8.515514196005824e-07 |
| test | match | 3.5762786865234375e-06 | 7.641403954039561e-07 |

## Official CEM smoke

- Status: `skipped`
- Return code: `n/a`
- Requested episodes: `10`
- Reported success rate (%): `n/a`

This is a functionality gate. Simulator replay is the separate Step 2 gate.
The official evaluator resets without dataset seeds; process/CEM seeding alone does not prove simulator determinism.
