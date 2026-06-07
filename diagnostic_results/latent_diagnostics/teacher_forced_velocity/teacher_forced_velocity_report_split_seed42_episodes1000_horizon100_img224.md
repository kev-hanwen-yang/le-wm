# Teacher-Forced Velocity Diagnostic

| Horizon | Teacher-forced ratio |
|---:|---:|
| 15 | 0.3247 |
| 30 | 0.3343 |
| 50 | 0.3269 |
| 100 | 0.3319 |

At h=50, ratio_mean=0.3269: under-shoot: the one-step map appears intrinsically smoothed or damped.

Charts:
- [Velocity by horizon](pusht_expert_train_pusht_lewm_split_seed42_episodes1000_horizon100_img224_tf_velocity_by_horizon.png)
- [Velocity ratio by horizon](pusht_expert_train_pusht_lewm_split_seed42_episodes1000_horizon100_img224_tf_velocity_ratio_by_horizon.png)
- [One-step error by horizon](pusht_expert_train_pusht_lewm_split_seed42_episodes1000_horizon100_img224_tf_one_step_error_by_horizon.png)

Caveat: this test isolates the one-step map and does not by itself diagnose the open-loop collapse mechanism; that comes from comparing the teacher-forced and open-loop ratio curves in chart 2.
