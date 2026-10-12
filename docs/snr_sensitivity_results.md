# SNR sensitivity results

## Protocol

The 5, 10 and 15 dB points use the same two-user directional-Rician channel,
20° angular separation, relative CSIT error 0.2, `n=128` and 256 bits/user.
Each SNR has its own oracle dataset, three independently pretrained actors and
three 65,536-step PPO runs. This tests whether the method can be trained for
each operating load; it is not a zero-shot generalization test across SNR.

Final evaluation uses deterministic policies, the same six held-out
channel/CSIT/ACK seeds at every SNR, 8,192 slots per trajectory and a 2,048-slot
burn-in. Learned-policy intervals use training seeds as the uncertainty unit.

## Results

| SNR | Objective-warm RSMA-PPO | SDMA-power PPO | Relative reduction | Seed wins | Trajectory wins |
|---:|---:|---:|---:|---:|---:|
| 5 dB | **2.4803** ± 0.0693 | 3.2489 ± 0.6992 | 23.66% | 3/3 | 6/6 |
| 10 dB | **1.5440** ± 0.0093 | 1.6709 ± 0.1463 | 7.59% | 3/3 | 6/6 |
| 15 dB | **1.0342** ± 0.0234 | 1.2164 ± 0.0314 | 14.98% | 3/3 | 6/6 |

The `±` values are training-seed standard deviations, not standard errors.
The 10 dB value uses only seeds 0–2 and a new common evaluation-seed set, so it
is deliberately separate from the five-seed headline value of 1.5511.

## Learned common-stream behaviour

| SNR | Common power | Common payload scale `rho` | Common-on fraction | Private BLER |
|---:|---:|---:|---:|---:|
| 5 dB | 0.0518 | 0.00051 | 0.2325 | 0.5802 |
| 10 dB | 0.3527 | 0.3394 | 0.4606 | 0.4856 |
| 15 dB | 0.4248 | 0.4213 | 0.6544 | 0.0199 |

At 5 dB, `rho` is effectively zero. The numerical gain there therefore cannot
be presented as evidence that carrying a substantial common payload is useful.
It is consistent with objective-aware initialization and the larger RSMA
parameterization providing a more stable private-power policy; SDMA also has
large training-seed variance, and its confidence interval overlaps the RSMA
interval. The third matched seed is nearly tied (2.4517 versus 2.4749).

At 10 and 15 dB, the policy assigns substantial power and payload to the common
stream. At 15 dB it raises the mean success probability from 0.8434 for SDMA to
0.9727 and operates close to the AoI lower bound of one. The percentage gain is
therefore less informative than the absolute difference and success rates.

The simple fixed allocation (`common power=0.25`, `rho=0.30`) produces no ACKs
in the 5 dB trajectories. This is a finite-blocklength threshold failure of that
fixed action, not evidence that all RSMA allocations fail at low SNR.

## Supported conclusion and limits

The trained objective-warm policy is lower than the trained SDMA-power policy
at all three tested SNR points under matched information, optimization budget
and evaluation trajectories. Common-stream use changes strongly with SNR, so a
fixed split is not a credible baseline in this regime.

The non-monotone relative gains (23.66%, 7.59%, 14.98%) should not be fitted to a
general SNR trend with only three points and three training seeds. The 5 dB
result is primarily an optimization-stability result because common payload is
negligible. These experiments also do not establish one checkpoint's robustness
to an unseen SNR, because every point was retrained.

Raw results and the plotting inputs are in
`results/thesis_snr{05,10,15}_longrun_v1/` and
`results/thesis_snr_sensitivity_v1/`.
