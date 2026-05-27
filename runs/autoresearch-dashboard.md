# Autoresearch Dashboard: Optimize Diffusion Training Speed

**Runs:** 3 | **Kept:** 3 | **Discarded:** 0 | **Crashed:** 0
**Baseline:** training_time_s: 118.634s (#1)
**Best:** training_time_s: 109.285s (#3, -7.9%)

| # | commit | training_time_s | status | description |
|---|--------|---------------|--------|-------------|
| 1 | c4ffbb3 | 118.634s | keep | baseline |
| 2 | 6d3cb50 | 110.615s (-6.8%) | keep | hardtanh instead of tanh logit softcap |
| 3 | 931fd51 | 109.285s (-7.9%) | keep | remove dead code in diffusion forward, --num-cpus 1 |
