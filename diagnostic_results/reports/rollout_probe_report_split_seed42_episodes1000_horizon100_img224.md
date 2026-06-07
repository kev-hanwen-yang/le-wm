# Push-T Rollout Probe Report

## Setup

- Episodes: 1000
- Target steps: [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
- Probe types: ['linear', 'mlp']
- Targets: ['agent_location', 'block_location', 'block_angle']

## Charts

- Normalized MSE: `/Users/kevinyang/Desktop/LeWM/le-wm/.stable-wm/probes/reports/rollout_probe_report_split_seed42_episodes1000_horizon100_img224_charts/norm_mse_by_horizon.png`
- Pearson r: `/Users/kevinyang/Desktop/LeWM/le-wm/.stable-wm/probes/reports/rollout_probe_report_split_seed42_episodes1000_horizon100_img224_charts/pearson_r_by_horizon.png`
- Raw RMSE: `/Users/kevinyang/Desktop/LeWM/le-wm/.stable-wm/probes/reports/rollout_probe_report_split_seed42_episodes1000_horizon100_img224_charts/raw_rmse_by_horizon.png`

## Final Horizon

| Probe | Physical Quantity | Step | Norm MSE | Raw RMSE | Pearson r |
|---|---|---:|---:|---:|---:|
| linear | Agent Location | 100 | 1.301571 | 115.280067 | 0.201327 |
| linear | Block Location | 100 | 1.605499 | 92.057526 | 0.353234 |
| linear | Block Angle | 100 | 2.622731 | 3.105649 | 0.066174 |
| mlp | Agent Location | 100 | 1.201429 | 110.816231 | 0.205892 |
| mlp | Block Location | 100 | 1.555220 | 90.636665 | 0.354842 |
| mlp | Block Angle | 100 | 2.347990 | 2.938486 | 0.087339 |
