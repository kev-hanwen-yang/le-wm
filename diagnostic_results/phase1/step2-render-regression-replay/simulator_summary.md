# Phase I Step 2 — Fresh-start simulator replay

- Status: `completed`
- Fresh-start Step 2 gate: `True`
- Encoder-facing dataset ready: `False`
- Completed cases: `24`
- Tolerance: `1e-10`, `rtol=0`

| Case | Repeat | Observer-neutral | History stationary | Agent–block points | Block–wall points |
|---|---|---|---|---|---|
| stationary_seed20261014 | True | True | True | 0 | 0 |
| free_motion_seed20261014 | True | True | True | 0 | 0 |
| near_contact_hold_seed20261014 | True | True | True | 0 | 0 |
| contact_push_seed20261014 | True | True | True | 145 | 0 |
| contact_release_seed20261014 | True | True | True | 90 | 0 |
| wall_contact_seed20261014 | True | True | True | 100 | 193 |
| rotated_hold_seed20261014 | True | True | True | 0 | 0 |
| angle_branch_hold_seed20261014 | True | True | True | 0 | 0 |
| stationary_seed20261015 | True | True | True | 0 | 0 |
| free_motion_seed20261015 | True | True | True | 0 | 0 |
| near_contact_hold_seed20261015 | True | True | True | 0 | 0 |
| contact_push_seed20261015 | True | True | True | 145 | 0 |
| contact_release_seed20261015 | True | True | True | 90 | 0 |
| wall_contact_seed20261015 | True | True | True | 100 | 193 |
| rotated_hold_seed20261015 | True | True | True | 0 | 0 |
| angle_branch_hold_seed20261015 | True | True | True | 0 | 0 |
| stationary_seed20261016 | True | True | True | 0 | 0 |
| free_motion_seed20261016 | True | True | True | 0 | 0 |
| near_contact_hold_seed20261016 | True | True | True | 0 | 0 |
| contact_push_seed20261016 | True | True | True | 145 | 0 |
| contact_release_seed20261016 | True | True | True | 90 | 0 |
| wall_contact_seed20261016 | True | True | True | 100 | 193 |
| rotated_hold_seed20261016 | True | True | True | 0 | 0 |
| angle_branch_hold_seed20261016 | True | True | True | 0 | 0 |

## Scope and unresolved assumptions

Only fresh quiescent starts are supported. No complete Chipmunk snapshot/restore is claimed.
Dynamic public-state restoration remains explicitly unsupported; saved negative diagnostics explain why.
Rendering is reproducible under the stated settings; exact compatibility with training rendering remains unresolved because source goal/render metadata is absent.
Do not collect encoder-facing ABC/H2 data until that convention is resolved. No H1/H2/H3 experiment ran here.
