# 불완전 CSIT 및 유한 블록길이 환경에서 AoI 최소화를 위한 상태 적응형 RSMA-PPO 자원 배분

## 초록

짧은 상태정보 패킷을 전송하는 무선 시스템에서는 전송률뿐 아니라 수신 정보의 신선도를 나타내는 Age of Information(AoI)이 중요하다. Rate-Splitting Multiple Access(RSMA)는 공통 스트림과 사용자별 private 스트림을 결합하여 다중사용자 간섭을 부분적으로 복호하고 나머지를 잡음으로 처리한다. 본 연구는 finite blocklength(FBL), 불완전한 channel state information at the transmitter(CSIT), 장기 AoI가 결합된 환경에서 전력과 payload 분할을 학습한다.

사용자 3명과 송신 안테나 2개인 overloaded multi-user multiple-input single-output(MU-MISO) 하향링크에서 proximal policy optimization(PPO)이 common 전력 비율, private 전력 비율 및 사용자별 common payload 비율을 직접 결정한다. 정책은 추정 채널에서 계산한 causal PHY feature만 관측한다. FBL 복호 임계영역의 불안정한 탐색을 줄이기 위해 one-step expected-AoI 목적을 사용해 actor를 초기화한 뒤 mixed-regime 학습과 목표 조건 미세조정을 수행한다.

Geometry span 20도, 15 dB signal-to-noise ratio(SNR), blocklength 128에서 두 방법의 사전학습을 동일한 objective-only 손실로 맞췄을 때 RSMA-PPO의 q-only PPO-SDMA 대비 seed별 평균 AoI 개선율은 16.01%였다. 95% 신뢰구간 half-width는 3.40 percentage point였고 30개 paired 평가 궤적 모두에서 RSMA-PPO가 낮은 AoI를 보였다. 매 slot 최대 AoI 사용자 한 명만 MRT로 전송하는 기준선 대비 기존 RSMA-PPO의 개선율은 26.87%였다. 낮은 SNR 또는 짧은 blocklength의 stress 조건은 사전학습에는 포함됐지만 최종 PPO curriculum에서는 제외됐다. 이 조건에서는 단일 사용자 방식이 더 낮은 AoI를 보였으며, PHY 난도·학습 분포 변화·활성 집합 제약의 영향은 분리하지 않았다.

**주요어:** RSMA, Age of Information, finite blocklength, PPO, imperfect CSIT, MU-MISO, resource allocation

---

## 1. 서론

### 1.1 연구 배경

상태 모니터링, 차량 제어, 산업 자동화 및 Digital Twin 동기화에서는 수신기가 보유한 정보가 얼마나 최근에 생성됐는지가 중요하다. AoI는 현재 시각과 가장 최근 성공적으로 수신된 update의 생성 시각 차이로 이를 정량화한다 [@kaul2012realtime; @yates2021aoi]. 높은 throughput이나 낮은 packet delay가 항상 낮은 AoI를 보장하지 않으므로 상태 update 시스템은 별도의 신선도 목적함수를 필요로 한다.

MU-MISO 하향링크에서는 제한된 안테나, 사용자 채널 정렬, 불완전 CSIT로 다중사용자 간섭이 발생한다. SDMA는 송신 precoding으로 간섭을 억제하지만 사용자의 공간 채널이 정렬되거나 시스템이 overloaded되면 성능이 제한된다. RSMA는 사용자 메시지 일부를 하나의 common stream에 포함하고 나머지를 private stream으로 전송한다. 각 사용자는 common stream을 먼저 복호한 후 successive interference cancellation(SIC)을 수행하고 자신의 private stream을 복호한다 [@mao2018bridging; @joudeh2016partial; @mao2022survey].

짧은 상태 패킷에서는 같은 SINR이라도 blocklength와 payload에 따라 block error rate(BLER)가 크게 변한다. 복호 임계값 부근에서는 작은 allocation 변화가 성공확률을 급격히 바꾼다 [@polyanskiy2010fbl]. FBL-RSMA 연구는 precoder와 common rate를 함께 최적화할 필요성을 보여주지만 [@xu2022fblrsma], 장기 AoI 정책은 현재 AoI와 시간에 따라 변하는 채널도 고려해야 한다.

### 1.2 연구 질문

> 불완전 CSIT를 가진 overloaded FBL MU-MISO 하향링크에서 하나의 PPO 정책이 현재 AoI와 추정 PHY 상태를 이용해 RSMA의 common/private 전력 및 payload 분할을 안정적으로 학습하고, 어떤 운용 조건에서 SDMA 휴리스틱보다 낮은 장기 AoI를 달성하는가?

### 1.3 연구 기여

1. **Causal PHY-aware 상태와 제약을 만족하는 연속 행동:** 추정 채널에서 계산한 beam gain과 nominal signal-to-interference-plus-noise ratio(SINR)를 상태에 포함하고, common 전력 alpha, private 전력 simplex q_k, 사용자별 common payload fraction phi_k를 직접 출력한다.
2. **장기 AoI 기반 상태 적응 결과:** 다섯 독립 학습 seed에서 동일한 channel, CSIT 오차 및 acknowledgement(ACK) 난수를 사용해 평가한다. 정책은 geometry와 SNR뿐 아니라 사용자 AoI 순위에 따라 전력과 payload를 바꾼다.
3. **q-only PPO-SDMA 대조 비교:** 데이터 상태 집합, objective-only 사전학습 손실·epoch, PPO curriculum·학습량과 all-user 조건을 맞추고 alpha와 phi_k를 0으로 고정한 PPO-SDMA를 비교한다.
4. **AoI-aware 규칙 기준선:** Max-AoI 단일 사용자 MRT와 validation으로 조정한 고정 규칙 RSMA를 추가해 동시 전송 이득과 상태 적응 allocation 이득을 구분한다.
5. **유효 영역과 실패 영역의 동시 보고:** feasible 영역의 개선과 함께 최종 PPO curriculum 밖의 stress 조건에서 관측된 열세를 보고하고, 그 원인이 분리되지 않았음을 명시한다.

One-step objective 사전학습과 staged curriculum은 본 실험의 학습 절차다. Scratch PPO 및 curriculum 제거 조건과의 matched ablation이 없으므로 독립적인 성능 기여로 주장하지 않는다. RSMA, FBL, AoI 또는 PPO의 최초 결합도 주장하지 않는다.

---

## 2. 관련 연구

### 2.1 RSMA와 FBL

1-layer RSMA는 한 개의 common stream과 사용자별 private stream을 중첩한다. Mao 등은 RSMA가 SDMA와 NOMA의 간섭 관리 전략을 연결하며 채널 방향, 채널 세기 및 CSIT 품질에 따라 다른 운용점이 선택됨을 보였다 [@mao2018bridging]. Joudeh와 Clerckx는 partial CSIT 하에서 sample-average approximation과 weighted minimum mean-square error(WMMSE)를 사용한 robust rate-splitting을 제시했다 [@joudeh2016partial]. 이 결과는 최적화된 precoder와 weighted sum-rate 목적에 기반하므로 고정 beam과 AoI 목적을 사용하는 본 연구에 그대로 적용할 수 없다.

Polyanskiy 등의 normal approximation은 유한한 blocklength에서 rate, SINR 및 BLER의 관계를 근사한다 [@polyanskiy2010fbl]. Xu 등은 short-packet MU-MISO RSMA에서 precoder와 common rate를 successive convex approximation(SCA)으로 최적화했으며 일부 조건에서 common stream을 끄는 SDMA 경계가 선택될 수 있음을 보였다 [@xu2022fblrsma].

### 2.2 AoI와 강화학습

Kaul 등은 상태 update queue의 AoI를 정식화했고 [@kaul2012realtime], Kadota 등은 broadcast wireless network의 AoI scheduling을 분석했다 [@kadota2018scheduling]. 본 연구는 generate-at-will traffic을 사용해 queue와 재전송을 제거한다. 따라서 성공 시 AoI는 한 slot으로 reset된다. HARQ와 RSMA를 결합한 AoI 최소화 연구도 존재하지만 [@ali2025harqrsma], 본 연구는 재전송 상태를 포함하지 않는다.

Yan 등은 satellite IoT의 RSMA short-packet AoI 문제를 다뤘고 [@yan2023satellite], Kaya는 FBL 시스템에서 RL 기반 rate splitting을 연구했다 [@kaya2025thesis]. Hieu 등은 RSMA power allocation에 deep reinforcement learning(DRL)을 적용했다 [@hieu2021drl]. 따라서 RSMA, AoI 및 DRL의 결합이나 PPO 사용 자체가 본 연구의 신규성은 아니다. 본 연구는 overloaded terrestrial MU-MISO에서 causal PHY state, 직접적인 alpha-q-phi 행동, 공통 난수 기반 five-seed 평가와 실패 영역 분석에 초점을 둔다.

---

## 3. 시스템 모델

### 3.1 채널과 불완전 CSIT

송신 안테나 N_t=2개인 기지국이 단일 안테나 사용자 K=3명에게 동시에 상태 update를 전송한다. K>N_t이므로 all-user transmission은 overloaded 상태다. 사용자 k의 directional Rician uniform linear array(ULA) 채널은

\[
\mathbf h_k(t)=\mathbf m_k+\mathbf g_k(t)
\]

\[
\mathbf m_k=\sqrt{\beta_k\frac{\kappa}{\kappa+1}}\,\mathbf a(\theta_k),\qquad
\mathbf g_k(t)\sim\mathcal{CN}\left(\mathbf 0,\frac{\beta_k}{\kappa+1}\mathbf I\right)
\]

이다. Homogeneous large-scale gain과 Rician factor kappa=8 dB를 사용한다. Geometry span g는 양끝 사용자 각도의 차이이며 기본 배치는 (-g/2, 0, g/2)다. 따라서 5/20/60도 조건의 각도는 각각 (-2.5, 0, 2.5), (-10, 0, 10), (-30, 0, 30)도이며, 각 배치의 사용자 순열 6개를 사용한다. 산란 성분은 eta=0.8인 1차 Gauss-Markov process로 진화한다.

\[
\mathbf h_k(t+1)=\mathbf m_k+\eta[\mathbf h_k(t)-\mathbf m_k]+\sqrt{1-\eta^2}\,\mathbf z_k(t)
\]

송신기는 sigma_e=0.2인 다음 추정 채널만 사용한다.

\[
\hat{\mathbf h}_k=\mathbf m_k+\sqrt{1-\sigma_e^2}[\mathbf h_k-\mathbf m_k]+\sigma_e\mathbf e_k
\]

Beam과 PPO observation은 추정 채널로 만들고 실제 SINR와 ACK는 실제 채널로 계산한다.

### 3.2 1-layer RSMA PHY

송신 신호와 총전력 제약은

\[
\mathbf x=\mathbf p_c s_c+\sum_{k=1}^{K}\mathbf p_k s_k,\qquad
\|\mathbf p_c\|^2+\sum_k\|\mathbf p_k\|^2\le P_{\max}
\]

이다. Private 방향은 추정 채널 기반 regularized zero-forcing(RZF), common 방향은 활성 사용자 추정 채널 행렬의 principal singular direction으로 고정한다. 사용자 k의 common 및 SIC 이후 private SINR는

\[
\gamma_{c,k}=\frac{|\mathbf h_k^H\mathbf p_c|^2}{\sum_j|\mathbf h_k^H\mathbf p_j|^2+\sigma_n^2}
\]

\[
\gamma_{p,k}=\frac{|\mathbf h_k^H\mathbf p_k|^2}{\sum_{j\ne k}|\mathbf h_k^H\mathbf p_j|^2+\sigma_n^2}
\]

이다.

### 3.3 FBL 성공확률

\[
C(\gamma)=\log_2(1+\gamma),\qquad
V(\gamma)=[1-(1+\gamma)^{-2}](\log_2 e)^2
\]

\[
\epsilon(\gamma,n,d)\approx Q\left(\frac{\sqrt n[C(\gamma)-d/n]}{\sqrt{V(\gamma)}}\right)
\]

를 사용한다. Capacity-dispersion 항만 포함하며 third-order log2(n)/(2n) 보정항은 생략한다. Header, pilot, CRC overhead와 residual SIC는 모델링하지 않는다.

Common과 private 부분이 모두 있는 사용자의 packet 성공확률은

\[
p_k^{\mathrm{succ}}=(1-\epsilon_{c,k})(1-\epsilon_{p,k})
\]

이다. 전체 common payload가 양수이면 자신의 common share가 0인 활성 사용자도 전체 common codeword를 복호해야 한다. 성공한 SIC 이후의 private SINR를 적용하며, 전체 common payload가 0일 때만 common 복호를 생략한다. 다른 사용자의 common 복호 실패는 사용자 k의 ACK를 강제로 실패시키지 않는다.

### 3.4 Generate-at-will AoI

매 slot 각 사용자의 새로운 256-bit update가 준비된다. ACK를 A_k(t)라 하면

\[
\Delta_k(t+1)=1 \quad \mathrm{if}\ A_k(t)=1
\]

\[
\Delta_k(t+1)=\Delta_k(t)+1 \quad \mathrm{if}\ A_k(t)=0
\]

이다. 목적은 장기 평균 AoI를 최소화하는 것이다.

\[
\bar\Delta=\limsup_{T\to\infty}\frac{1}{TK}\mathbb E\left[\sum_{t=0}^{T-1}\sum_{k=1}^{K}\Delta_k(t)\right]
\]

---

## 4. 제안 방법

### 4.1 PPO 행동과 상태

모든 세 사용자를 활성화한 allocation-only 문제에서 행동은

\[
a_t=(\alpha_t,\mathbf q_t,\boldsymbol\phi_t)
\]

\[
P_c=\alpha P_{\max},\qquad P_k=(1-\alpha)q_kP_{\max},\qquad q_k\geq0,\quad\sum_kq_k=1
\]

\[
D_{c,k}=\phi_kD_k,\qquad D_{p,k}=(1-\phi_k)D_k,\qquad 0\leq\phi_k\leq1
\]

이다. 전체 common codeword는 D_c=sum_k D_c,k bits이며 모든 활성 사용자가 이를 복호한다. 표준 RSMA의 사용자별 common rate는 C_k=D_c,k/n이다. phi_k의 합은 1일 필요가 없다.

alpha와 phi_k에는 endpoint를 정확히 허용하는 h(z)=clip(1/2+z/6,0,1)을 적용한다. q_k는 logit의 masked softmax로 simplex를 만든다. 따라서 alpha=0과 모든 phi_k=0인 SDMA 경계를 정확히 표현한다. Common on/off는 별도의 이진 변수로 결정하지 않고 연속 변수 alpha가 0에 도달할 수 있도록 표현한다.

정책 상태는 정규화 AoI, 이전 ACK와 allocation, 추정 채널의 사용자별 gain과 pairwise correlation, common 및 private beam gain, 이전 allocation의 nominal common/private SINR, 현재 SNR과 blocklength를 포함한다. 모든 PHY feature는 추정 채널만 사용한다.

### 4.2 보상과 PPO objective

환경 보상은 sampled ACK로 갱신된 다음-slot 평균 AoI의 음수다.

\[
u_t=-\frac{1}{K}\sum_k\Delta_k(t+1)
\]

PPO는 generalized advantage estimation(GAE)을 사용하며 할인율 gamma_RL=0.99, GAE 계수 lambda=0.95다. 환경은 사용자 AoI 합의 음수에 reward_scale=1/3을 곱해 위 평균 보상을 만들고, 그 episode 합을 return으로 기록한다. PPO 갱신에는 이후 running statistics로 정규화한 보상을 사용한다. 따라서 그림 4의 logged return은 PPO 내부 보상 정규화 이전의 값이다. Clipped surrogate는

\[
L^{\mathrm{clip}}(\theta)=\mathbb E_t\left[\min\{\rho_t(\theta)\hat A_t,\operatorname{clip}(\rho_t(\theta),1-\epsilon_{\mathrm{clip}},1+\epsilon_{\mathrm{clip}})\hat A_t\}\right]
\]

이며 epsilon_clip=0.2다. Actor와 critic은 각각 128-unit tanh hidden layer 두 개를 사용한다.

### 4.3 Objective-aware 사전학습과 curriculum

수치 label은 추정 채널을 실제 채널의 대용값으로 사용하는 cross-entropy method(CEM) 탐색으로 생성한다. CEM은 상태당 4 iterations, 128 population, 1 restart를 사용해 다음 one-step expected AoI를 최소화한다. 아래 H는 성공확률 계산에 사용하는 채널이며, beam은 추정 채널로 생성한다. Label 탐색에서는 H=hat H를 대입하며 posterior 평균을 사용하지 않는다.

\[
J_1(a;H,\hat H)=\frac1K\sum_k\left[\Delta_k+1-p_k^{\mathrm{succ}}(a;H,\hat H)\Delta_k\right]
\]

\[
\hat J_1(a;\hat H)=\frac{1}{8}\sum_{m=1}^{8}J_1(a;H^{(m)},\hat H),\qquad H^{(m)}\sim p(H\mid\hat H)
\]

Posterior channel 8개에 대한 sample-average approximation(SAA)은 label 생성이 아니라 actor의 differentiable FBL-AoI 손실에 사용한다. 구현 손실은 posterior 평균의 음의 AoI-weighted 성공 합이다. 이는 위 hat J_1에 행동과 무관한 항을 빼고 K를 곱한 값이다. 최종 matched 비교의 두 정책은 같은 objective-only 손실을 사용한다.

\[
L_{\mathrm{pre}}^{\mathrm{RSMA}}=L_{\mathrm{pre}}^{\mathrm{SDMA}}=L_{\mathrm{FBL-AoI}}
\]

두 정책은 같은 데이터 상태 집합에서 80 epochs 사전학습한다. 데이터는 geometry span 5/20/60도, SNR 5/10/15/20 dB, n=64/128/256의 전체 격자에서 생성돼 stress 조건도 포함한다. q-only는 common-off 행동 공간에서, RSMA는 alpha-q-phi 행동 공간에서 같은 AoI 목적을 직접 최적화한다. 별도 진단 정책은 CEM label 행동 모방과 목적 손실을 결합한 `0.5 L_action + L_FBL-AoI`를 사용했으며, 목표 조건의 최종 AoI는 objective-only RSMA와 통계적으로 구분되지 않았다.

사전학습 후 Stage 1 PPO는 geometry span 5/20/60도, SNR 10/15/20 dB, n=128/256에서 524,288 steps 학습한다. 이어 20도, 15 dB, n=128에서 524,288 steps 미세조정한다. 최종 정책의 PPO 단계에는 stress 조건이 포함되지 않는다. 별도의 full-grid Stage 2 진단 checkpoint는 최종 정책의 초기화로 사용하지 않았다. Scratch PPO 및 curriculum 제거 ablation이 없으므로 이 절차의 독립적인 이득은 정량화하지 않는다.

---

## 5. 실험 설계

### 5.1 주요 파라미터

| 항목 | 값 |
|---|---|
| 사용자/안테나 | K=3, N_t=2 |
| 채널 | Directional Rician ULA, kappa=8 dB |
| 시간 상관 | Gauss-Markov, eta=0.8 |
| CSIT | Relative error sigma_e=0.2, perfect CSIR |
| Geometry span g | 5/20/60도; (-g/2, 0, g/2), 각 6 user permutations |
| SNR | 5/10/15/20 dB |
| Blocklength | 64/128/256 |
| Payload | 사용자당 256 bits |
| Beam | RZF private, principal common |
| Traffic | Generate-at-will, queue/HARQ 없음 |
| Episode 길이 | 512 slots |
| Training seeds | 0, 1, 2, 3, 4 |

### 5.2 기준선과 공정성

주 SDMA 기준선은 매 slot 현재 AoI가 가장 큰 두 사용자를 선택하고 두 private stream에 전력을 균등 분배한다. common stream은 사용하지 않는다. 이 기준선은 PPO처럼 q_k를 학습하지 않으므로 최종 PPO와의 차이는 common stream의 효과만을 뜻하지 않는다. 모든 방법은 같은 총전력, blocklength, 추정 채널 정보와 FBL 모델을 사용하며 channel, CSIT 오차 및 ACK 난수를 공유한다.

AoI scheduling의 기본 기준선으로 매 slot 최대 AoI 사용자 한 명에게 전체 전력을 주는 정책을 사용한다. 단일 사용자 RZF 방향은 MRT와 동일한 방향이다. 이 정책은 대칭적인 link 조건에서 maximum-age-first greedy scheduling에 해당한다 [@kadota2018scheduling]. 매 slot 최대 한 명만 성공할 수 있고 성공확률이 1인 K=3 시스템의 최소 평균 AoI는 순환 서비스에서 (K+1)/2=2다.

고정 규칙 RSMA는 세 사용자를 모두 활성화하고 최대 AoI 사용자의 256-bit update 전체를 common stream에 배치한다. 이 사용자는 phi_k=1, private 전력 0으로 설정한다. 나머지 두 사용자는 phi_k=0으로 private stream만 사용하고 잔여 private 전력을 절반씩 나눈다. PPO와 같은 principal common beam과 RZF private beam을 사용한다. Common 전력 비율 alpha는 test와 분리된 3개 validation repeat와 6개 사용자 순열에서 0.05 간격으로 탐색했으며 alpha=0.70이 선택됐다. 이 기준선은 PPO가 학습한 oldest-user common allocation을 간단한 수작업 규칙이 재현할 수 있는지 검사한다.

Common stream의 기여는 별도의 paired 투영으로 검사한다. 최종 PPO가 출력한 q_k와 all-user schedule은 유지하고 alpha와 phi_k만 0으로 고정한다. 이 투영은 별도로 최적화한 SDMA가 아니며 최종 actor에서 common을 제거한 반사실적 대조군이다. CEM은 사전학습 label 생성에 사용하지만 최종 성능 주장의 중심 기준선으로 사용하지 않는다.

q-only PPO-SDMA를 별도로 학습해 비교한다. 이 정책은 RSMA-PPO와 동일하게 세 사용자를 모두 활성화하고 동일한 RZF beam, state, 데이터 상태 집합, objective-only 사전학습 손실·epoch, mixed Stage 1, target fine-tuning step과 training seed를 사용한다. 행동 공간에서는 alpha=0과 모든 phi_k=0을 고정하고 private 전력 simplex q_k만 학습한다. 따라서 matched 비교에서 남는 체계적 차이는 RSMA의 common 전력·payload 제어 자유도다. 다만 행동 차원과 최적화 난도까지 같지는 않으므로 common stream 하나의 인과 효과를 완전히 분리한 것으로 해석하지 않는다.

### 5.3 평가 protocol

각 checkpoint는 여섯 user permutation에서 768 slots 평가하고 처음 256 slots를 burn-in으로 제외한다. 다섯 독립 training seed 각각에 대해 permutation 평균을 먼저 계산한다. 95% 신뢰구간은 다섯 training-seed 평균을 독립 표본으로 하는 Student-t 구간이다.

표의 성공확률은 burn-in 이후 각 사용자와 slot에 대해 PHY가 계산한 packet 성공확률 p_k^succ의 평균이다. 실제 ACK는 이 확률에서 Bernoulli 표본화하며 AoI 갱신과 별도로 기록한다. p95 AoI는 각 평가 궤적에서 burn-in 이후 모든 사용자-slot AoI 표본의 경험적 95백분위수를 구한 뒤 해당 궤적들을 평균한 값이다.

q-only 대비 16.01%는 각 training seed b에서 objective-only RSMA와 objective-only SDMA 각각의 6개 순열 평균 AoI를 먼저 계산하고, 다음 seed별 비율 G_b를 구한 뒤 5개 seed에 걸쳐 평균한 값이다.

\[
\bar\Delta_{j,b}=\frac{1}{6}\sum_{p=1}^{6}\bar\Delta_{j,b,p}
\]

\[
G_b=100\,\frac{\bar\Delta_{\mathrm{qonly},b}-\bar\Delta_{\mathrm{PPO},b}}{\bar\Delta_{\mathrm{qonly},b}},\qquad \bar G=\frac{1}{5}\sum_{b=0}^{4}G_b
\]

이 개선율의 신뢰구간도 5개 G_b에서 계산한다. 고정 규칙·단일 사용자·top-2 기준선과의 비교에서는 궤적 s의 paired gain을 먼저 계산하고 seed 내에서 평균한 뒤 seed 간에 집계한다. 아래 ref는 해당 비교 기준선이다.

\[
G_s=100\,\frac{\bar\Delta_{\mathrm{ref},s}-\bar\Delta_{\mathrm{PPO},s}}{\bar\Delta_{\mathrm{ref},s}}
\]

두 집계 모두 전체 평균 AoI 두 개의 비율과 일치할 필요가 없다. 본문은 저장된 seed 평균 선계산 결과인 16.01%를 사용한다. Feasible 구간은 SNR 10/15/20 dB와 blocklength 128/256의 18개 regime이다. 5 dB 또는 blocklength 64는 stress 구간이다. Stress는 사전학습에는 포함됐지만 최종 PPO curriculum에서는 제외됐다. 격자 평가는 action+objective 사전학습을 사용한 기존 최종 checkpoint로 수행했으며, objective-only RSMA는 목표 조건의 matched 감사에 사용했다.

---

## 6. 결과 및 분석

### 6.1 목표 조건의 정책별 성능 비교

**표 1. 목표 조건: 20도, 15 dB, n=128**

| 방법 | 평균 AoI | seed SD | 95% CI half-width | p95 AoI | 성공확률 | 평균 alpha |
|---|---:|---:|---:|---:|---:|---:|
| Objective-pretrained actor | 2.481 | 0.270 | 0.336 | 7.508 | 0.463 | 0.737 |
| RSMA-PPO, objective-only matched | **1.463** | 0.012 | 0.015 | 2.033 | **0.577** | 0.761 |
| RSMA-PPO, action+objective 진단 | 1.463 | 0.022 | 0.027 | **2.000** | 0.568 | 0.762 |
| 고정 규칙 RSMA, alpha=0.70 | 1.656 | 0.046* | - | 3.000 | 0.502 | 0.700 |
| q-only PPO-SDMA | 1.744 | 0.060 | 0.074 | 3.167 | 0.447 | 0.000 |
| Max-AoI 단일 사용자 MRT | 2.000 | 0.000* | - | 3.000 | 0.333 | 0.000 |
| Common-off 투영 | 2.650 | 0.125 | 0.155 | - | - | 0.000 |
| Max-AoI top-2 SDMA | 2.822 | - | - | 7.500 | 0.330 | 0.000 |

별표가 붙은 표준편차는 학습 seed가 없는 규칙 정책의 6개 test permutation 간 변동이다. 동일한 objective-only 사전학습을 사용한 RSMA-PPO의 q-only PPO-SDMA 대비 seed별 paired gain은 16.01%이고 95% 신뢰구간 half-width는 3.40 percentage point다. 평균 AoI 절대 차이는 0.280 ± 0.072(95% CI half-width)이며 RSMA-PPO는 30개 paired 궤적 모두에서 이겼다. Objective-only RSMA와 기존 action+objective RSMA의 차이는 -0.08% ± 2.26%p로 구분되지 않았다. 따라서 기존 16.05% 격차를 사전학습 손실 불일치가 만들었다는 가설은 지지되지 않는다. q-only PPO-SDMA가 max-AoI top-2 equal-power SDMA보다 낮은 AoI를 보인 결과에는 학습된 q_k와 세 사용자 all-user 전송의 차이가 함께 반영된다. Common-off 투영의 43.76% 역시 q_k를 재최적화하지 않은 제거 실험이므로 순기여 추정치로 사용하지 않는다.

고정 규칙 RSMA 대비 PPO의 paired gain은 11.63% ± 1.63 percentage point이고 Max-AoI 단일 사용자 대비 gain은 26.87% ± 1.34 percentage point다. 두 비교 모두 30개 paired 궤적 전부에서 PPO가 이겼다. 고정 규칙도 q-only PPO-SDMA보다 강했지만 PPO와의 격차가 남았으므로, 목표 조건에서는 validation으로 조정한 하나의 alpha와 고정 payload 규칙이 상태 적응 행동을 모두 대체하지 못했다. 단일 사용자 AoI 2.0을 넘은 것은 동시 전송의 이점이며 RSMA common stream만의 이점은 아니다.

![목표 조건의 matched 및 AoI-aware 기준선 비교](../results/thesis_rule_baselines_target_v1/figure_target_rule_baselines.png)

**그림 1.** 학습 정책의 error bar는 다섯 training seed의 95% Student-t 신뢰구간이다. 고정 규칙 RSMA의 error bar는 6개 test permutation의 표준편차다. 붉은 점선은 K=3에서 slot당 최대 한 명이 완벽하게 갱신될 때의 하한 2.0이다.

![동일 objective-only 사전학습 조건에서 RSMA-PPO와 q-only PPO-SDMA의 paired 비교](../results/matched_pretraining_target_v1/figure_matched_pretraining.png)

**그림 2.** 첫 두 열의 점과 선은 같은 training seed의 objective-only RSMA/SDMA paired 비교다. 세 번째 열은 기존 action+objective RSMA 진단 정책이다. 사각형과 error bar는 seed 평균과 95% Student-t 신뢰구간이다.

### 6.2 Regime별 성능과 alpha

**표 2. 사전 정의한 구간별 PPO와 두 scheduling 기준선 비교**

| 구간 | PPO 평균 AoI | top-2 SDMA 평균 AoI | 단일 사용자 평균 AoI | PPO의 단일 사용자 대비 paired gain |
|---|---:|---:|---:|---:|
| Feasible: SNR >= 10 dB, n >= 128 | **1.701** | 2.798 | 2.000 | **14.97% ± 1.65%p** |
| Stress: SNR=5 dB 또는 n=64 | 151.388 | 198.967 | **67.933** | 백분율 평균으로 요약하지 않음 |

Feasible 영역에서 단일 사용자 전송은 거의 항상 성공해 AoI 2.0에 머물지만 PPO가 5개 training seed 모두에서 더 낮았다. Stress 영역에서는 방법별 AoI가 수백까지 증가하는 셀이 있어 분모를 사용하는 궤적별 gain 평균이 극단값에 민감하다. 절대 AoI 평균에서는 Max-AoI 단일 사용자 전송이 현재 all-user PPO보다 낮다. Stress는 사전학습에서 관측했지만 최종 PPO 단계에서는 제외된 조건이므로, 이 열세를 scheduling만의 효과로 볼 수 없다. PHY 난도, 학습 분포 변화 및 행동 공간 제약의 영향은 분리하지 않았다.

**표 3. 학습된 common-power fraction의 조건부 평균**

| 변화 축 | 조건 | 평균 alpha | 95% CI half-width |
|---|---:|---:|---:|
| Geometry | 5도 | 0.619 | 0.068 |
| Geometry | 20도 | 0.589 | 0.063 |
| Geometry | 60도 | 0.397 | 0.061 |
| SNR | 5 dB | 0.250 | 0.104 |
| SNR | 10 dB | 0.453 | 0.066 |
| SNR | 15 dB | 0.659 | 0.035 |
| SNR | 20 dB | 0.778 | 0.026 |
| Blocklength | 64 | 0.500 | 0.132 |
| Blocklength | 128 | 0.542 | 0.047 |
| Blocklength | 256 | 0.562 | 0.032 |

각 행은 나머지 두 축을 평균한 marginal 결과다. Geometry와 SNR에 대한 방향성이 다섯 seed에서 유지됐다는 점은 alpha가 하나의 상수로 고정되지 않았다는 근거다.

![Geometry, SNR 및 blocklength에 따른 common-power fraction](../results/thesis_mixed_phy_final_v1/figure_2_alpha_response.png)

**그림 3.** Error bar는 다섯 training seed 평균의 95% Student-t 신뢰구간이며 각 panel은 나머지 조건을 평균한 marginal response다.

### 6.3 AoI에 따른 행동 적응

**표 4. 목표 조건에서 현재 AoI 순위별 평균 행동**

| AoI 순위 | 평균 q_k | 평균 phi_k |
|---|---:|---:|
| 가장 낮음 | 0.271 | 0.215 |
| 중간 | 0.291 | 0.263 |
| 가장 높음 | 0.438 | 0.713 |

정책은 더 오래된 사용자에게 private 전력과 common payload를 더 많이 배분했다. 최대 AoI가 2에서 3 이상으로 증가할 때 평균 alpha는 0.766에서 0.699로 감소했다. 이는 긴급한 사용자의 private 전달을 보호하는 행동과 양립하지만 관측 상관만으로 인과적 최적성을 뜻하지 않는다.

Pretraining actor의 target AoI는 2.481이고 PPO는 1.463이다. One-step differentiable objective와 장기 sampled AoI는 같은 목적함수가 아니며 PPO는 실제 ACK가 만드는 AoI state distribution에서 초기 actor를 수정했다. 다만 이 비교는 사전학습이 scratch PPO보다 낫다는 것을 분리해 입증하지 않는다.

![다섯 training seed의 target fine-tuning AoI와 episodic return](../results/thesis_mixed_phy_final_v1/figure_3_training_curves.png)

**그림 4.** 각 가는 선은 독립 training seed의 8-point moving average이고, 검은 선과 음영은 seed 평균과 ±1 표준편차다. (a) On-policy mean AoI는 다섯 seed 모두 약 1.91–2.00에서 1.54–1.63으로 감소했다. (b) 환경에 기록된 episodic return은 약 -1027~-980에서 -828~-789로 증가했다. 이 return은 reward_scale=1/3이 적용된 환경 보상의 episode 합이며 PPO 내부 running reward normalization 이전에 기록된 값이다. AoI도 학습 중 stochastic action과 방문 state에서 수집했으므로 frozen deterministic 평가 AoI 1.463과 직접 같은 통계량은 아니다.

### 6.4 실패 영역

n=64에서 사용자당 256 bit를 한 block에 담으려면 4 bit/channel-use의 coding rate가 필요하다. 본 채널 정규화에서는 사용자당 평균 채널 norm 제곱이 N_t=2다. 모든 전력을 한 사용자에게 주고 이상적인 MRT를 적용해도 평균 수신 SNR을 사용한 상계는 log2(1+2*10^(5/10))=2.87 bit/channel-use로 4보다 작다. 실제 시스템에는 다중사용자 간섭, 불완전 CSIT 및 FBL penalty가 추가되므로 일부 stress 조합은 물리적으로 매우 어렵다.

전체 격자에서 Max-AoI 단일 사용자 전송의 stress 평균 AoI는 67.933으로 PPO의 151.388보다 낮았다. 이 조건은 사전학습에는 포함됐으나 최종 PPO curriculum에서는 제외됐다. 활성 사용자가 바뀌면 RZF와 principal common beam, 간섭 및 payload 구조도 함께 달라진다. 따라서 이 비교만으로 학습 분포 변화와 고정 활성 집합의 영향을 분리할 수 없으며 scheduling의 인과 효과를 확정하지 않는다.

이 절대 AoI는 256-slot burn-in 이후 512-slot 측정 창에서 얻은 유한-horizon 값이다. 성공확률이 거의 0이면 AoI가 측정 길이에 따라 계속 증가하므로 이를 정상상태 평균으로 해석할 수 없다. 18개 stress regime 가운데 평균 성공확률이 10^-3 미만인 regime은 PPO와 단일 사용자 방식 모두 3개였고, 평균 AoI가 100을 넘은 regime은 각각 6개와 3개였다. 따라서 67.933과 151.388은 실패 심각도를 비교하는 제한된 평가값이며 안정된 장기 평균의 추정치는 아니다. 고정 payload, common bottleneck과 raw-AoI objective의 개별 기여도 분리하지 않았다. 1M-step full-grid 진단은 학습 분포와 절차가 달라 학습량 자체의 효과로 해석하지 않는다.

---

## 7. 한계

1. RZF와 principal common beam을 고정해 WMMSE/SCA precoder 최적화를 하지 않는다.
2. Generate-at-will traffic만 사용하고 packet arrival, queue, waiting time 및 HARQ를 모델링하지 않는다.
3. FBL 모델에서 pilot/header/CRC overhead, channel estimation overhead, residual SIC와 third-order 보정항을 제외한다.
4. 사용자당 payload를 256 bit로 고정하고 K=3, N_t=2만 최종 five-seed 학습했다.
5. 사전학습 손실까지 맞춘 q-only 비교는 목표 조건에서만 수행했다. 16.01%를 다른 geometry, SNR 및 blocklength로 일반화하지 않는다.
6. 고정 규칙 RSMA의 alpha는 목표 조건에서만 validation했으므로 다른 regime에서의 성능을 일반화하지 않는다.
7. CSIT 오차는 sigma_e=0.2 한 점에서만 평가해 불완전 CSIT 수준 변화에 대한 강건성을 입증하지 않았다.
8. 평가 channel, CSIT error 및 ACK seed는 공정한 paired 비교를 위해 한 frozen set을 공유한다. 따라서 신뢰구간은 training-seed 변동만 반영하며 평가 궤적 표본의 불확실성을 포함하지 않는다.
9. Scratch PPO와 curriculum 제거 조건이 없어 사전학습과 staged curriculum의 독립 기여를 정량화하지 못했다. Stress는 사전학습에만 포함됐으므로 최종 PPO 학습 분포 변화와 행동 제약의 영향도 분리하지 못했다.
10. Full-precoder WMMSE나 다른 DRL 알고리즘과 matched 재학습하지 않았다.
11. AoI는 update freshness를 나타낼 뿐 Digital Twin의 실제 state-estimation error나 source dynamics를 직접 모델링하지 않는다.

---

## 8. 결론

본 연구는 overloaded FBL MU-MISO 하향링크에서 common on/off를 이진 변수로 두지 않고 연속 alpha로 표현하는 RSMA-PPO allocation을 구현하고 평가했다. 정책은 추정 채널 기반 PHY-aware state로부터 common 전력, private 전력 및 common payload fraction을 출력한다. 동일한 objective-only 사전학습과 PPO 예산을 사용한 목표 조건 비교에서 q-only PPO-SDMA 대비 seed별 평균 AoI 개선율은 16.01%였고, 기존 action+objective RSMA와 objective-only RSMA의 최종 성능은 구분되지 않았다. Validation으로 조정한 고정 규칙 RSMA 대비 기존 RSMA-PPO 개선율은 11.63%였다. Max-AoI 단일 사용자 기준선 대비 26.87% 개선은 동시 update의 이점과 양립한다. Stress는 사전학습에는 포함됐으나 최종 PPO curriculum에서는 제외됐고, 이 영역에서는 단일 사용자 방식이 더 낮은 AoI를 보였다. 학습 분포 변화와 행동 제약의 영향은 분리하지 않았다.

결과의 범위는 고정 beam, generate-at-will traffic 및 K=3, N_t=2 시스템이다. 후속 연구에서는 matched q-only 비교를 전체 regime으로 확장하고 stochastic arrival, HARQ, adaptive scheduling 또는 blocklength, uncertainty-aware precoder를 순차적으로 추가할 수 있다.
