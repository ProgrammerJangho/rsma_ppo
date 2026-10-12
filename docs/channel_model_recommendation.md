# RSMA–FBL–AoI–PPO 채널 모델 권고안

**작성일:** 2026-09-07  
**결론:** 기존 temporally correlated Rayleigh 채널을 control로 유지하고,
mean-preserving Rician-ULA 채널을 주 실험축으로 추가한다. Previous의 channel을
통째로 복사하거나 3GPP 38.901 전체 모델을 도입하지 않는다.

## 1. 연구 질문에 맞는 채널의 조건

현재 논문의 중심은 PPO가 매 slot의 CSIT와 AoI를 보고 common/private power split과
사용자별 common-rate allocation을 결정하는 것이다. 따라서 채널은 다음 효과를
서로 분리할 수 있어야 한다.

1. 사용자 채널 방향의 유사성 또는 spatial correlation
2. 사용자별 평균 채널 세기 차이
3. 시간 상관과 정책이 활용할 수 있는 예측성
4. CSIT 품질
5. 사용자 수와 안테나 수의 load

RSMA가 좋아지기 쉬운 조건을 하나로 묶은 channel보다, 이 다섯 축을 독립적으로
바꿀 수 있는 모델이 성능 역전의 원인을 설명하기에 적합하다.

## 2. 문헌이 실제로 통제하는 채널 요인

### RSMA 대표 연구

Mao, Clerckx, Li의 대표 연구는 2-user MU-MISO 채널을

\[
\mathbf h_1=[1,1,\ldots]^H,\qquad
\mathbf h_2=\gamma[1,e^{j\theta},e^{j2\theta},\ldots]^H
\]

처럼 두고 `gamma`로 strength disparity, `theta`로 channel direction을 따로
통제한다. 채널이 정렬되면 MU linear precoding이 어렵고, 직교에 가까워질수록
SDMA가 강해진다. Imperfect-CSIT 실험에서는
`h=true estimate + Gaussian error` 형식과 SNR-dependent error variance를 사용한다.

- https://doi.org/10.1186/s13638-018-1104-7

FBL-RSMA 연구도 angle, channel strength, load와 blocklength를 분리한다. 짧은
blocklength와 특정 angle에서는 common stream을 끄고 SDMA 경계로 가는 해가
선택되며, `N_t=4,K=8` 같은 overloaded random channel도 별도로 평가한다.

- https://arxiv.org/abs/2105.06198
- https://doi.org/10.1109/TVT.2022.3191085

실제 RSMA prototype 연구 역시 두 사용자 LoS 채널의 **relative pathloss, spatial
correlation, CSIT quality**를 독립적으로 변화시켜 RSMA/SDMA/NOMA를 비교한다.
이는 Rician이라는 분포 이름보다 이 세 관측량을 보고하는 것이 중요하다는 가장
직접적인 실험 근거다.

- https://doi.org/10.1109/TWC.2024.3367891

Statistical-CSIT RSMA 연구는 spatially correlated Rayleigh와 ULA의 amplitude/phase
statistics를 별도 시나리오로 다룬다. 즉 Rayleigh와 directional ULA는 서로 대체재가
아니라 다른 channel-information regime이다.

- https://arxiv.org/abs/2104.00220

### RSMA + RL + AoI 인접 연구

PPO 기반 RSMA power-allocation 연구는 특정 Rician geometry를 RSMA의 필수 전제로
두지 않고 동적인 channel uncertainty를 MDP state로 사용한다.

- https://arxiv.org/abs/2107.00238

RSMA–FBL–AoI satellite 연구는 응용에 맞춰 shadowed-Rician을 사용하며, aerial
RSMA 연구는 Rician channel을 사용한다. 이는 LoS 응용에서 Rician이 타당하다는
근거지만 terrestrial MU-MISO에 satellite shadowed-Rician을 그대로 쓰라는 근거는
아니다.

- https://doi.org/10.1109/VTC2023-Fall60731.2023.10333373
- https://arxiv.org/abs/2306.12977

2026 V2X short-packet RSMA 연구는 mobility, outdated CSIT, antenna count, power와
rate splitting이 AoI에 미치는 영향을 함께 본다. V2X 또는 Digital Twin mobility를
주장하려면 단순 i.i.d. channel보다 시간 상관과 CSIT age를 명확히 정의해야 한다.

- https://arxiv.org/abs/2604.13691

3GPP TR 38.901은 pathloss, shadowing, delay/angle spread, K-factor, cluster와 spatial
consistency를 포함하는 광범위한 0.5–100 GHz 모델이다. 현재 단일 narrowband channel
vector와 FBL link abstraction에 전체 모델을 넣으면 많은 새 가정이 생기므로 최종
외부-validity 실험이 아니라면 과도하다.

- https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/18.01.00_60/tr_138901v180100p.pdf

## 3. 현재 구현 평가

현재 channel은

\[
\mathbf h_k(t+1)=\alpha\mathbf h_k(t)
+\sqrt{1-\alpha^2}\mathbf z_k(t),\qquad
\mathbf z_k\sim\mathcal{CN}(0,\beta_k\mathbf I)
\]

인 stationary Gauss–Markov Rayleigh block fading이다.

| 요소 | 평가 | 판정 |
|---|---|---|
| `CN(0,beta I)` Rayleigh | 표준적이고 명확한 control | 유지 |
| Gauss–Markov `alpha` | RL 환경에 필요한 시간 상관을 간단히 제공 | 유지 |
| true channel / `h_hat` 분리 | 정보 누출을 막는 올바른 구조 | 유지 |
| explicit near/far `beta` | strength disparity를 정확히 분리 | 유지 |
| 거리 pathloss·shadowing | 외부-validity용으로 충분 | 선택 실험 |
| spatial direction 제어 | i.i.d. Rayleigh에서는 직접 제어 불가 | 추가 필요 |
| Rician LoS mean | 없음 | 추가 필요 |
| `alpha`의 물리적 시간 단위 | 현재 정의되지 않음 | 문서화 필요 |
| CSIT error의 physical estimator | correlation-quality abstraction | 명칭 명확화 필요 |

현재 모델은 틀린 것이 아니다. 다만 i.i.d. antenna Rayleigh에서는 user channel
correlation이 realization에 따라 우연히 결정되므로, RSMA와 SDMA의 성능 역전을
channel alignment의 함수로 설명하기 어렵다.

## 4. 권장 channel family

### 4.1 Rayleigh control

기존 모델을 그대로 둔다.

\[
\mathbf h_k(t)=\sqrt{\beta_k}\mathbf g_k(t),\qquad
\mathbf g_k(t+1)=\alpha\mathbf g_k(t)
+\sqrt{1-\alpha^2}\mathbf z_k(t).
\]

이는 기존 실험과 backward compatibility를 보장하고 Rician 결과가 LoS 구조 때문인지
확인하는 control이 된다.

### 4.2 Controlled Rician-ULA main channel

권장 모델은

\[
\mathbf h_k(t)=\mathbf m_k+\widetilde{\mathbf h}_k(t),
\]

\[
\mathbf m_k=
\sqrt{\beta_k\frac{\kappa}{\kappa+1}}\,\mathbf a(\phi_k),
\qquad
\widetilde{\mathbf h}_k(t)\sim
\mathcal{CN}\!\left(0,\frac{\beta_k}{\kappa+1}\mathbf I\right),
\]

\[
[\mathbf a(\phi)]_n=
\exp\!\left(j2\pi d_\lambda n\sin\phi\right),
\quad n=0,\ldots,N_t-1,
\]

\[
\mathbf h_k(t+1)=\mathbf m_k+
\alpha\bigl(\mathbf h_k(t)-\mathbf m_k\bigr)
+\sqrt{1-\alpha^2}\mathbf z_k(t),
\]

\[
\mathbf z_k(t)\sim
\mathcal{CN}\!\left(0,\frac{\beta_k}{\kappa+1}\mathbf I\right).
\]

Steering vector는 `1/sqrt(Nt)`로 정규화하지 않고 각 원소의 크기를 1로 둔다. 그러면

\[
\mathbb E\|\mathbf h_k\|^2=N_t\beta_k
\]

가 되어 기존 Rayleigh channel의 `beta_k`와 SNR 의미가 정확히 유지된다.

### 4.3 Rician에서의 CSIT error

현재 relative model을 LoS mean 주위에 적용한다.

\[
\widehat{\mathbf h}_k=\mathbf m_k+
\sqrt{1-\tau^2}(\mathbf h_k-\mathbf m_k)
+\tau\mathbf e_k,
\quad
\mathbf e_k\sim
\mathcal{CN}\!\left(0,\frac{\beta_k}{\kappa+1}\mathbf I\right).
\]

이 정의는 `h`와 `h_hat`의 mean과 covariance를 동일하게 유지하고 `kappa=0`에서 현재
relative-CSIT 모델과 정확히 같아진다. LoS angle 또는 array calibration error는
포함하지 않는다. 이를 연구하려면 별도의 angle-error 변수로 두어야 하며 첫
학부논문 실험에는 넣지 않는다.

현재 `absolute`와 `normalized` CSIT mode는 Rayleigh robustness 실험용으로 남기되,
Rician 주 결과에는 위 mean-preserving relative mode를 사용한다.

## 5. Previous channel을 그대로 이식하면 안 되는 이유

Previous는 다음을 동시에 사용한다.

- `K=8,N_t=4`
- Rician K-factor 8 dB
- 모든 angle을 30도 구간 안에 배치
- 거리 30–100 m와 pathloss exponent 3
- episode별 CSIT quality 변화
- AR(1) NLoS correlation
- residual SIC 0.08
- maxmin common beam

이 설정에서는 alignment, overload, pathloss, imperfect CSIT, imperfect SIC와 beam
변경이 모두 섞인다.

정규화도 현재와 다르다. Previous는 `a(phi)/sqrt(Nt)`와
`g~CN(0,I/Nt)`를 사용하므로

\[
\mathbb E\|\mathbf h_k\|^2=\beta_k.
\]

현재는 `CN(0,beta_k I)`이므로

\[
\mathbb E\|\mathbf h_k\|^2=N_t\beta_k.
\]

그대로 복사하면 `N_t=4`에서 평균 channel power가 4배, 즉 6.02 dB 달라진다.
RSMA/SDMA 차이 이전에 FBL operating point 자체가 이동한다.

또한 Previous의 “Jakes channel”은 실제 sum-of-sinusoids Jakes process가 아니라
`alpha=J0(2 pi f_D T)` 한 값을 사용한 AR(1) process다. 현재 Gauss–Markov 구조와
본질적으로 같으므로 별도 channel 구현을 복사할 필요가 없다.

## 6. 추가할 configuration

| 변수 | 의미 | 기본값 |
|---|---|---|
| `small_scale_model` | `rayleigh`, `rician_ula` | `rayleigh` |
| `rician_k_db` | LoS/scatter power ratio | `8.0` |
| `user_angles_deg` | 사용자별 물리적 AoD | 명시 배열 |
| `antenna_spacing_wavelength` | ULA 간격 | `0.5` |
| existing `alpha` | centered scatter AR(1) coefficient | `0.8` |
| existing `path_gains_db` | controlled strength disparity | `[0,0]` 또는 `[0,-5]` |
| existing `csit_error_std` | normalized CSIT decorrelation `tau` | `0` 또는 `0.3` |

`rician_k_db`는 첫 구현에서는 scalar로 둔다. 사용자별 K-factor, angle spread,
clustered multipath와 shadowing dynamics는 추가하지 않는다.

## 7. 추천 실험 순서

### Channel unit/PHY validation

1. `kappa=0` Rician이 Rayleigh와 통계적으로 일치
2. 모든 `kappa`에서 `E||h_k||^2=Nt beta_k`
3. centered lag-1 correlation이 `alpha`
4. ULA steering correlation이 이론값과 일치
5. Rician CSIT estimate의 mean/covariance 보존
6. perfect CSIT에서 `h_hat==h`
7. 모든 MA가 동일한 true/estimated channel realization 사용

### Phase A: spatial alignment

- `K=N_t=2`
- `kappa=8 dB`
- angles `[-Delta/2,+Delta/2]`
- `Delta={5,20,60}` degrees
- homogeneous `beta=[1,1]`
- perfect CSIT
- `alpha=0.8`, SNR 10 dB, `n=128`, `D=256` 유지

`N_t=2,d_lambda=0.5`에서 LoS steering-vector correlation은 대략 다음과 같다.

| angle separation | normalized LoS correlation |
|---:|---:|
| 5° | 0.981 |
| 20° | 0.731 |
| 60° | 0.000 |

이 세 점은 aligned/intermediate/orthogonal regime을 명확히 분리한다.

### Phase B: strength disparity

- Phase A의 `Delta={5,20,60}` 유지
- path-gain gap `{0,5}` dB
- 먼저 perfect CSIT만 사용

10 dB gap은 5 dB 결과를 확인한 뒤 선택 실험으로 둔다. 첫 factorial에 넣으면
FBL outage가 지배해 spatial effect가 보이지 않을 수 있다.

### Phase C: CSIT robustness

- 대표 angle `Delta={5,60}`
- gap `{0,5}` dB
- `tau={0,0.3}`

이 단계에서 principal/maxmin beam factor를 포함한다. Imperfect-CSIT allocator가
conditional uncertainty를 평균하지 않는 한 결과는 “robust RSMA 최적값”이 아니라
plug-in PPO/beam design의 결과로 표현한다.

### Phase D: load

- main result가 안정된 후 `N_t=2,K=3` 또는 `N_t=4,K={4,6}` 한 쌍만 추가
- scheduling을 켠 경우 실제 active-user count와 overload 빈도를 반드시 보고

## 8. 시간 상관의 표현

현재 `alpha=0.8`은 비교용으로 유지할 수 있지만 물리적 속도라고 부르면 안 된다.
논문에서는 “Gauss–Markov lag-one correlation coefficient”라고 정의한다.

속도와 연결하고 싶다면 별도 표에서

\[
\alpha=J_0(2\pi f_D T_s),\qquad f_D=v f_c/c
\]

로 calibration한다. 이것은 Jakes ACF의 lag-one 값을 AR(1)에 맞춘 근사이며 전체
Jakes process와 동일하지 않다. 현재 논문에서는 `alpha={0,0.8,0.98}` robustness
sweep이면 충분하고 carrier frequency와 slot duration을 새 핵심 변수로 만들 필요는
없다.

## 9. 반드시 보고할 channel diagnostics

각 결과 표에는 평균 AoI만 두지 않고 다음을 기록한다.

- realized normalized channel correlation의 mean/median/p05/p95
- configured LoS correlation과 realized composite correlation
- per-user average received SNR와 path-gain gap
- CSIT normalized MSE
- lag-one centered channel correlation
- channel condition number 또는 RZF effective interference
- common/private BLER와 common-on 비율

Angle separation만 보고 실제 composite channel correlation을 보고하지 않으면
Rician scatter 때문에 operating regime가 의도대로 형성됐는지 확인할 수 없다.

## 10. 최종 선택

논문용 channel 구성은 다음으로 확정하는 것이 가장 설득력 있다.

> **Single-cell narrowband MU-MISO block fading with a ULA, controlled
> Rician LoS directions, Gauss–Markov correlated scattering, per-user
> large-scale gains, and mean-preserving imperfect CSIT.**

Rayleigh는 반드시 control로 유지한다. 주 분석은 `K=N_t=2`에서 angle correlation,
5 dB near/far와 CSIT quality의 interaction을 본다. Overload와 distance/shadowing은
일반화 실험으로 제한한다.

이 구성은 채널을 RSMA에 유리하게 고정하지 않고 aligned부터 orthogonal까지 모두
포함한다. 동시에 PPO가 power split과 common-rate allocation을 왜 다르게 선택하는지
해석할 수 있으며, 현재 발견한 RSMA–SDMA 성능 역전을 모델 오류, optimizer 문제,
operating regime 효과로 분리할 수 있다.

## 11. 구현 및 통계 검증 결과

위 권고안을 코드에 반영했다. 기본값은 계속 `rayleigh`이며, `rician_ula`를 선택한
경우에만 `rician_k_db`, `user_angles_deg`, `antenna_spacing_wavelength`를 사용한다.
Rician에서는 물리적 estimator를 별도로 가정하지 않은 `absolute`/`normalized` CSIT를
금지하여, LoS mean을 뜻하지 않게 훼손하는 조합이 조용히 실행되지 않도록 했다.

기존 Rayleigh의 `sample_initial`, `evolve`, `estimate(relative)`는 별도 legacy 경로를
유지한다. 같은 seed에 대해 이전 식과 배열이 원소별로 정확히 일치하는 회귀 테스트를
통과했다. 따라서 기존 실험 결과가 새 기본값 때문에 달라지지 않는다.

`20,000`개의 독립 표본으로 세 angle config를 검증한 결과는 다음과 같다. 표의 LoS
correlation은 정규화된 squared inner product이다.

| angle separation | theoretical/empirical LoS correlation | empirical centered time corr. | target |
|---:|---:|---:|---:|
| 5° | 0.981339 | 0.800710 | 0.8 |
| 20° | 0.730771 | 0.799256 | 0.8 |
| 60° | ~0 | 0.799142 | 0.8 |

`tau=0.3`에서 centered `corr(h,h_hat)`은 세 config에서
`0.953992`, `0.954182`, `0.954183`이었고 이론값은
`sqrt(1-0.3^2)=0.953939`이다. Channel total-power 최대 상대오차는 `0.312%`, estimate
total-power 최대 상대오차는 `0.416%`였다. 검증 원자료는
`results/channel_model_validation_rician_v1/channel_moments.csv`에 저장했다.

전체 test suite `541`개가 통과했다. 이 검증은 distribution과 environment integration을
확인한 것이며, PPO 성능 결론은 아직 아니다. 다음 실험에서는 동일 channel trace에서
RSMA/SDMA 및 beam/allocation을 먼저 비교한 뒤 PPO를 학습해야 한다.
