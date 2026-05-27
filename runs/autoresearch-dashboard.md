# Autoresearch Dashboard: Optimize Diffusion Training Speed

**Runs:** 2 | **Kept:** 2 | **Discarded:** 0 | **Crashed:** 0
**Baseline:** training_time_s: 118.634s (#1)
**Best:** training_time_s: 110.615s (#2, -6.8%)

| # | commit | training_time_s | status | description |
|---|--------|---------------|--------|-------------|
| 1 | c4ffbb3 | 118.634s | keep | baseline |
| 2 | 6d3cb50 | 110.615s (-6.8%) | keep | hardtanh instead of tanh logit softcap |
