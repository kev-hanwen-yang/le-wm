# Phase I Step 1 — Frozen baseline

- Git revision: `a008317afbf4bfa94b1c32f0da121c3da18f4a86`
- Checkpoint SHA-256: `100d6459b9198697c1c26ef47e3eef4d7548946401b7501213caf761495a2553`
- Device: `cpu`
- Repeated encoding max absolute delta: `0.0`
- Repeated prediction max absolute delta: `0.0`
- Deterministic / finite: `True` / `True`

## Cached-embedding identity

| Split | Status | Max abs. difference | RMSE |
|---|---:|---:|---:|
| train | match | 3.4570693969726562e-06 | 7.271822255461302e-07 |
| val | match | 3.337860107421875e-06 | 7.570064326500869e-07 |
| test | match | 2.9802322387695312e-06 | 7.20864250070008e-07 |

## Official CEM smoke

- Status: `completed`
- Return code: `0`
- Fixed-seed episodes: `10`
- Reported success rate: `90.0`

This smoke uses the existing evaluator and is not a physical counterfactual result.
