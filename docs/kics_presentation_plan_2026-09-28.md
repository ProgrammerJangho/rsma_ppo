# RSMA-PPO AoI 교수님 발표용 PPT 구성안

## 발표 방향

이 발표의 핵심은 **RSMA가 항상 우수하다는 주장**이 아니다. 다음 한 문장을 실험으로 뒷받침하는 것이 목적이다.

> 사용자 3명, 송신 안테나 2개의 과부하 MU-MISO 목표 조건에서, 동일한 사전학습과 PPO 학습 예산을 사용한 RSMA-PPO가 SDMA-PPO보다 장기 평균 AoI를 15.34% 낮췄으며, 정책은 AoI 긴급도와 채널 정렬도에 따라 전력과 payload를 다르게 배분했다.

발표 본문에서는 학습하지 않은 SNR 5 dB 및 blocklength 64 stress 결과를 제외한다. 해당 결과는 RSMA의 구조적 한계와 PPO의 분포 밖 일반화 실패를 구분할 수 없으므로 핵심 결론으로 사용하지 않는다.

권장 발표 시간은 8~10분이며, 본문 9장과 질의응답용 보조 슬라이드 4장으로 구성한다.

---

## 슬라이드 1. 제목과 핵심 결과

### 제목

**유한 블록길이 과부하 MU-MISO 하향링크에서  
AoI 최소화를 위한 PPO 기반 RSMA 자원 할당**

영문 제목은 논문과 동일하게 작은 글씨로 배치한다.

### 화면 구성

중앙에 다음 흐름을 한 줄로 표시한다.

```text
AoI·채널 상태 → PPO → α, q_k, φ_k → RSMA 전송 → ACK·AoI 갱신
```

하단에 핵심 결과를 크게 표시한다.

```text
RSMA-PPO 평균 AoI: 1.401
SDMA-PPO 대비 15.34% 감소
```

### 발표 멘트

> 사용자 3명과 송신 안테나 2개의 과부하 환경에서, PPO가 RSMA의 common 전력, private 전력과 common payload를 결정하도록 설계했습니다. RSMA와 SDMA에 동일한 사전학습과 누적 2.10M target fine-tuning step을 적용한 결과, RSMA-PPO가 SDMA-PPO보다 평균 AoI를 15.34% 낮췄습니다.

### 발표 시간

약 30초.

---

## 슬라이드 2. 연구 동기와 문제 정의

### 왼쪽: AoI 개념

```text
새 정보 수신          전송 실패             새 정보 수신
AoI = 1      →      AoI = 2, 3, 4      →      AoI = 1
```

차량 위치, 센서 상태 또는 로봇 제어 상태와 같은 아이콘을 사용한다.

### 오른쪽: 과부하 MU-MISO

```text
기지국 송신 안테나: N_t = 2
사용자: K = 3
K > N_t
```

기지국에서 사용자 세 명으로 향하는 beam을 그린다. beam이 완전히 분리되지 않고 겹치는 모습으로 간섭을 표현한다.

### 핵심 문장

- 기존 RSMA 연구는 주로 sum-rate 또는 spectral efficiency를 최적화한다.
- 상태 업데이트 시스템에서는 전송률보다 정보의 신선도인 AoI가 중요하다.
- `K=3 > N_t=2`인 과부하 조건에서는 RZF만으로 간섭을 완전히 제거하기 어렵다.
- finite blocklength에서는 SINR과 payload 변화에 따라 BLER이 급격히 달라진다.

### 발표 멘트

> 상태 업데이트 시스템에서는 데이터를 많이 보내는 것보다 수신 정보가 얼마나 최신인지가 중요합니다. 이를 AoI로 측정합니다. 현재 시스템은 두 개의 송신 안테나로 세 사용자를 동시에 서비스하므로 공간 자유도가 부족합니다. 따라서 common stream으로 일부 간섭을 복호하는 RSMA가 AoI를 낮출 수 있는지 검토했습니다.

### 발표 시간

약 1분.

---

## 슬라이드 3. 1-layer RSMA 시스템

### 시스템 그림

```text
                           ┌─ common + private 1 → 사용자 1
기지국 ── common stream ───┼─ common + private 2 → 사용자 2
2 Tx antennas              └─ common + private 3 → 사용자 3

각 사용자: common 복호 → SIC → 자기 private 복호
```

색상은 다음과 같이 통일한다.

- common stream: 주황색
- private stream: 파란색
- PPO: 초록색
- AoI 또는 실패: 빨간색

### 송신 신호

화면에는 다음 수식 하나만 제시한다.

\[
\mathbf{x}
=
\sqrt{\alpha P}\mathbf{u}_c s_c
+
\sum_k\sqrt{(1-\alpha)q_kP}\mathbf{u}_ks_k.
\]

### PPO 행동의 의미

| 행동 | 의미 |
|---|---|
| \(\alpha\) | 전체 전력 중 common stream에 사용하는 비율 |
| \(q_k\) | 남은 private 전력을 사용자별로 나누는 비율, \(\sum_k q_k=1\) |
| \(\phi_k\) | 사용자 \(k\)의 패킷을 common으로 보내는 비율 |

### Payload 그림

```text
사용자 k의 D = 256-bit 상태 update

┌──── common: φ_k D ────┬──── private: (1−φ_k)D ────┐
```

사용자별 common 부분은 하나의 common codeword로 결합된다.

\[
D_c=\sum_k \phi_kD.
\]

### 꼭 설명할 내용

- 모든 사용자가 동일한 common stream 하나를 먼저 복호한다.
- common 제거 후 각 사용자는 자기 private stream만 복호한다.
- 다른 사용자의 private stream은 간섭으로 남는다.
- NOMA처럼 사용자 신호를 채널 순서대로 여러 번 SIC하지 않는다.
- \(\alpha=0\), \(\phi_k=0\)이면 common stream이 없어져 SDMA 경계가 된다.

### 발표 멘트

> 1-layer RSMA에서는 모든 사용자가 복호하는 common stream 하나와 사용자별 private stream을 동시에 보냅니다. 사용자 패킷의 \(\phi_kD\) bit는 common으로, 나머지는 private으로 전송합니다. \(\alpha\)는 common 전력, \(q_k\)는 private 전력 배분입니다. 모든 사용자가 common을 먼저 제거한 뒤 자기 private만 복호하며, 사용자별 오름차순 SIC는 수행하지 않습니다.

### 발표 시간

약 1분 30초.

---

## 슬라이드 4. FBL 성공확률과 AoI 갱신

### 패킷 성공확률

common과 private 부분이 모두 필요한 사용자의 성공확률은 다음과 같이 모델링한다.

\[
p_k=
[1-\epsilon(\gamma_{c,k},D_c)]
[1-\epsilon(\gamma_{p,k},(1-\phi_k)D)].
\]

FBL BLER의 전체 normal approximation 식은 본문 슬라이드에 넣지 않고 보조 슬라이드에 둔다.

### AoI 갱신

\[
\Delta_k(t+1)=
\begin{cases}
1, & \text{전송 성공},\\
\Delta_k(t)+1, & \text{전송 실패}.
\end{cases}
\]

### 화면용 흐름

```text
common 성공?
      ↓
private 성공?
   ┌──┴──┐
 성공    실패
AoI=1   AoI+1
```

### 발표 멘트

> 짧은 패킷에서는 Shannon capacity 대신 finite-blocklength normal approximation으로 BLER을 계산합니다. common과 private 부분이 모두 필요한 경우 두 부분을 모두 성공해야 패킷 성공으로 처리합니다. 성공하면 AoI가 1로 reset되고 실패하면 1 증가합니다.

### 발표 시간

약 50초.

---

## 슬라이드 5. PPO의 상태·행동·보상

### 화면 구성

```text
┌──────────── 상태 ────────────┐
│ 사용자별 AoI                 │
│ 이전 ACK·이전 전력 할당       │
│ 채널 이득·사용자 상관도       │
│ beam 이득·nominal SINR       │
│ SNR·blocklength              │
└──────────────┬───────────────┘
               ↓
          ┌─────────┐
          │   PPO   │
          │ 2×128   │
          └────┬────┘
               ↓
┌──────────── 행동 ────────────┐
│ α : common 전력              │
│ q_k: 사용자별 private 전력    │
│ φ_k: 사용자별 common payload  │
└──────────────────────────────┘
               ↓
          RSMA PHY·ACK
               ↓
            다음 AoI
```

### 보상

\[
r_t=-\frac{1}{K}\sum_k\Delta_k(t+1).
\]

### 왜 PPO인가

- \(\alpha,q_k,\phi_k\)는 연속 행동이다.
- AoI와 이전 ACK 때문에 상태가 시간에 따라 누적된다.
- PPO는 학습 후 한 번의 신경망 추론으로 행동을 출력한다.
- PPO가 SAC나 TD3보다 항상 우수하다고 주장하지 않는다.

### 발표 멘트

> PPO는 현재 AoI와 추정 채널 정보를 보고 common 전력, 사용자별 private 전력과 common payload를 결정합니다. 보상은 다음 slot 평균 AoI의 음수입니다. 연속 행동을 안정적으로 학습하고, 학습 후에는 반복 수치 최적화 없이 빠르게 행동을 출력할 수 있어 PPO를 사용했습니다.

### 발표 시간

약 1분 20초.

---

## 슬라이드 6. 학습 및 평가 설계

### 학습 과정

```text
One-step expected AoI objective 사전학습
                  ↓
Mixed-regime PPO: 524,288 step
각도 5°/20°/60°, SNR 10/15/20 dB, n=128/256
                  ↓
Target fine-tuning: 누적 2,097,152 step
20°, 15 dB, n=128
```

오른쪽에 논문의 학습 return 곡선을 배치한다.

### 공정성 조건

- RSMA-PPO와 SDMA-PPO에 동일한 사전학습 방식 적용
- 동일한 mixed curriculum
- 동일한 누적 target fine-tuning step
- 학습 seed 5개
- 학습 seed마다 독립 평가 seed 5개
- 평가 seed마다 사용자 순열 6개
- 방법별 총 150개 평가 궤적
- 같은 채널·CSIT 오차·ACK 난수 사용

### 통계 단위

```text
6개 사용자 순열 평균
        ↓
5개 평가 seed 평균
        ↓
학습 seed별 성능
        ↓
5개 학습 seed로 평균·95% Student-t CI 계산
```

150개 궤적을 독립 표본으로 처리하지 않았음을 강조한다.

### Learning-rate restart

목표 조건 학습은 세 구간으로 수행했다.

```text
0 → 0.52M → 1.05M → 2.10M
           ↑        ↑
       LR annealing restart
```

actor, critic, optimizer, 관측·보상 정규화 상태는 복원했다. RSMA와 SDMA 모두 같은 절차를 사용했다.

### 발표 멘트

> 비교의 공정성을 위해 RSMA와 SDMA 모두 동일한 사전학습, curriculum과 누적 2.10M step을 사용했습니다. 학습 seed 5개 각각을 평가 seed 5개와 사용자 순열 6개에서 평가했습니다. 신뢰구간은 150개 궤적을 독립 표본으로 사용하지 않고, 최종 학습 seed 다섯 개를 통계 단위로 계산했습니다.

### 발표 시간

약 1분 20초.

---

## 슬라이드 7. 목표 조건 성능

### 권장 그래프

평균 AoI 막대그래프를 크게 배치한다. AoI는 낮을수록 좋다는 화살표를 표시한다.

| 방법 | 평균 AoI | 95% CI |
|---|---:|---:|
| RSMA-PPO | **1.401** | ±0.007 |
| 고정 규칙 RSMA | 1.645 | ±0.015 |
| SDMA-PPO | 1.655 | ±0.033 |
| Max-AoI 단일 사용자 MRT | 2.000 | - |

### 오른쪽 핵심 수치

```text
SDMA-PPO 대비 15.34% 감소

학습 seed        5/5 승리
학습·평가 seed  25/25 승리
전체 궤적      150/150 승리
```

### 학습량 증가 효과

슬라이드 하단에 작은 글씨로 표시한다.

- RSMA-PPO: 0.52M → 2.10M에서 AoI 4.70% 감소
- SDMA-PPO: 0.52M → 2.10M에서 AoI 5.10% 감소
- RSMA의 SDMA 대비 개선율: 0.52M에서 15.68%, 2.10M에서 15.34%

추가 학습이 RSMA의 상대적 우위를 새로 만든 것이 아니라는 근거다.

### 발표 멘트

> 목표 조건에서 RSMA-PPO의 평균 AoI는 1.401, 동일한 학습량의 SDMA-PPO는 1.655로 15.34% 차이가 났습니다. 이 우위는 학습 seed 5개, 학습·평가 seed 쌍 25개와 전체 150개 궤적에서 모두 나타났습니다. 다만 행동 차원과 최적화 난도가 다르기 때문에 15.34% 전체를 common stream만의 순수 인과효과라고 단정하지는 않습니다.

### 발표 시간

약 1분 30초.

---

## 슬라이드 8. PPO가 학습한 행동

### \(q_k\): AoI 긴급도 대응

```text
AoI가 가장 큰 사용자
        ↓
private 전력 q_k 집중
```

안정적인 4개 seed에서 최대 AoI 사용자 선택률은 97.7~99.0%였다.

### \(\phi_k\): 채널 정렬도 대응

```text
사용자 채널 방향이 비슷함
        ↓
사용자 간 간섭 증가
        ↓
common payload 증가
```

안정적인 4개 seed에서 평균 \(\phi_k\)와 채널 정렬도의 Spearman 상관은 0.68~0.82였다.

### \(\alpha\): common 전력 운용점

- 평균 \(\alpha\): 약 0.81
- 4개 seed는 높은 common 전력점 주변에서 작은 폭으로 보정
- seed 1은 \(\alpha\)를 넓게 사용하는 다른 해를 찾았지만 AoI는 1.404로 비슷함

### Figure 2 활용

논문의 common power 그래프를 함께 배치한다.

- SNR 10→20 dB: 평균 \(\alpha\) 0.53→0.84
- 사용자 각도 폭 60°: 평균 \(\alpha\) 0.62로 감소

위 조건은 모두 mixed-regime 학습 범위에 포함된 조건이다.

### Counterfactual 해석

학습된 행동을 보정된 상수나 단순 규칙으로 대체하면 모든 학습 seed에서 AoI가 증가했다. 다만 seed별 증가 폭이 크므로 각 행동의 중요도가 모든 seed에서 동일하다고 주장하지 않는다.

### 발표 멘트

> 행동을 분석하면 \(q_k\)는 AoI가 오래된 사용자에게 private 전력을 집중했습니다. \(\phi_k\)는 사용자 채널이 서로 정렬될수록 common payload를 증가시켰습니다. \(\alpha\)는 평균 약 0.81의 높은 common 전력 운용점 주변에서 보정됐습니다. 네 개 seed에서는 유사한 역할 분리 패턴이 나타났고 seed 1은 다른 행동 해를 찾았지만 성능은 비슷했습니다.

### 발표 시간

약 1분 20초.

---

## 슬라이드 9. 결론과 연구 범위

### 결론

- 목표 조건에서 RSMA-PPO 평균 AoI: **1.401**
- 동일 학습 예산의 SDMA-PPO 대비: **15.34% 감소**
- 5개 학습 seed와 5개 평가 seed에서 일관된 개선
- \(q_k\): AoI 긴급도와 강한 관계
- \(\phi_k\): 채널 정렬도와 강한 관계
- \(\alpha\): 높은 common 전력 운용점 주변의 보정

### 연구 범위

- \(K=3,N_t=2\) 과부하 MU-MISO
- generate-at-will traffic
- CSIT 오차 \(\sigma_e=0.2\)
- RZF private beam 및 principal common beam 고정
- 학습 조건: SNR 10/15/20 dB, blocklength 128/256, 각도 폭 5°/20°/60°
- 핵심 평가 조건: 20°, 15 dB, \(n=128\)

### 후속 연구

- 사용자 scheduling을 PPO 행동에 추가
- CSIT 오차 수준 sweep
- stochastic arrival 및 queue 확장
- SAC 또는 TD3와 알고리즘 비교
- common beam 또는 precoder 공동 최적화

후속 연구를 모두 반드시 해야 하는 것처럼 말하지 않는다. 현재 연구에서 가장 직접적인 확장은 사용자 scheduling이다.

### 마지막 발표 멘트

> 동일한 학습과 평가 조건에서 RSMA 행동 공간을 사용한 PPO가 SDMA-PPO보다 낮은 AoI를 달성했습니다. 또한 정책은 AoI 긴급도와 채널 정렬도에 따라 해석 가능한 행동 패턴을 보였습니다. 현재 결론은 고정 beam과 지정된 학습 범위에 한정되며, 이후에는 사용자 scheduling을 포함하는 방향으로 확장할 수 있습니다.

### 발표 시간

약 50초.

---

# 질의응답용 보조 슬라이드

## 보조 슬라이드 A. 1-layer RSMA와 NOMA의 SIC 차이

| 1-layer RSMA | Power-domain NOMA |
|---|---|
| 모든 사용자가 common stream 하나를 먼저 복호 | 강한 사용자가 약한 사용자 신호부터 복호 |
| common 제거 후 자기 private만 복호 | 사용자별 순서에 따라 여러 신호를 순차 복호 |
| private stream 사이의 decoding order 없음 | 채널 또는 전력 기반 decoding order 필요 |

### 질문 대응

> NOMA에서는 사용자 신호 사이의 SIC 순서가 필요하지만, 본 연구의 1-layer RSMA에서는 모든 사용자가 동일한 common stream 하나를 먼저 제거한 뒤 자기 private stream만 복호합니다. 따라서 사용자별 오름차순 SIC는 사용하지 않습니다.

---

## 보조 슬라이드 B. SINR과 FBL 상세식

\[
\gamma_{c,k}
=
\frac{P_c|\mathbf h_k^H\mathbf u_c|^2}
{\sum_jP_j|\mathbf h_k^H\mathbf u_j|^2+\sigma^2},
\]

\[
\gamma_{p,k}
=
\frac{P_k|\mathbf h_k^H\mathbf u_k|^2}
{\sum_{j\ne k}P_j|\mathbf h_k^H\mathbf u_j|^2+\sigma^2}.
\]

\[
\epsilon(\gamma,d)
=Q\!\left(
\frac{n\log_2(1+\gamma)-d}{\sqrt{nV(\gamma)}}
\right).
\]

private SINR 분모에 다른 사용자의 private stream이 남는다는 점을 표시한다.

---

## 보조 슬라이드 C. 학습 seed별 결과

| 학습 seed | RSMA-PPO 2.10M | SDMA-PPO 2.10M |
|---:|---:|---:|
| 0 | 1.4082 | 1.6316 |
| 1 | 1.4038 | 1.6365 |
| 2 | 1.3932 | 1.6915 |
| 3 | 1.4012 | 1.6416 |
| 4 | 1.3980 | 1.6742 |

모든 학습 seed에서 RSMA-PPO가 SDMA-PPO보다 낮다.

---

## 보조 슬라이드 D. 예상 질문과 답변

### Q1. 15.34%가 common stream만의 순수 이득인가?

> 아닙니다. 동일한 학습량에서 RSMA 행동 공간을 사용한 PPO와 SDMA 행동 공간을 사용한 PPO의 시스템 수준 차이입니다. 행동 차원과 최적화 난도가 다르기 때문에 전체 차이를 common stream만의 인과효과로 단정하지 않습니다.

### Q2. 왜 PPO를 사용했는가?

> 행동이 \(\alpha,q_k,\phi_k\)의 연속 변수이고 AoI가 시간에 따라 누적되는 문제이기 때문입니다. PPO는 연속 행동을 비교적 안정적으로 학습하며, 학습 후에는 신경망 한 번의 추론으로 행동을 결정할 수 있습니다.

### Q3. 왜 SAC나 TD3와 비교하지 않았는가?

> PPO가 항상 최적이라고 주장하지 않습니다. 본 연구에서는 시스템과 행동 설계의 타당성을 먼저 검증하기 위해 PPO를 사용했으며, 알고리즘 간 비교는 후속 범위입니다.

### Q4. 왜 WMMSE baseline이 없는가?

> WMMSE는 일반적으로 weighted sum-rate 목적의 precoder와 rate allocation에 사용됩니다. 본 연구는 finite-blocklength 성공확률을 거쳐 장기 AoI를 최소화하며, 고정 beam 아래 전력과 payload를 상태에 따라 제어합니다. 학부 연구 범위에서는 동일 학습 예산의 SDMA-PPO와 AoI 규칙을 직접 기준선으로 사용했습니다.

### Q5. 왜 \(\alpha\)가 약 0.8로 높은가?

> 과부하되고 채널 방향이 비교적 정렬된 목표 조건에서는 common stream을 이용해 간섭을 복호하는 높은-common 운용점을 찾았습니다. 다만 각도 폭 60°에서는 평균 \(\alpha\)가 0.62로 감소했고, seed 1은 \(\alpha\)를 넓게 사용하는 다른 해를 찾았습니다.

### Q6. PPO가 장기적인 제어를 배웠다는 근거가 있는가?

> 학습된 행동을 별도 보정 궤적에서 구한 상수나 단순 순위 규칙으로 바꾸면 모든 학습 seed에서 closed-loop AoI가 증가했습니다. 따라서 완전한 고정 정책으로 수렴한 것은 아닙니다. 다만 one-step 사전학습과 PPO의 순수 기여를 완전히 분리한 결과는 아닙니다.

### Q7. precoder는 왜 PPO 행동에 포함하지 않았는가?

> precoder까지 학습하면 행동 차원이 크게 증가하고 성능 원인 분석이 어려워집니다. 본 연구에서는 RZF private beam과 principal common beam을 고정하고 전력과 payload 제어에 범위를 한정했습니다.

### Q8. 학습하지 않은 저 SNR 또는 매우 짧은 blocklength 결과는 어떻게 해석하는가?

> 해당 조건은 PPO 학습 분포 밖이므로 현재 결과만으로 RSMA의 구조적 한계와 정책의 일반화 실패를 구분할 수 없습니다. 해당 조건을 검증하려면 curriculum에 포함해 다시 학습한 뒤 동일한 기준선과 비교해야 합니다. 따라서 본 발표의 핵심 성능 주장에는 포함하지 않았습니다.

---

# PPT 디자인 원칙

## 색상

- RSMA/common: 주황색
- SDMA/private: 파란색
- PPO: 초록색
- AoI 또는 주의 사항: 빨간색
- 배경: 흰색

## 글자 크기

- 슬라이드 제목: 28~32 pt
- 핵심 수치: 28 pt 이상
- 본문: 18~22 pt
- 표·그림 축: 최소 15 pt
- 각주: 최소 12~13 pt

## 작성 원칙

- 한 슬라이드에 핵심 문장은 3~5개만 둔다.
- 논문 문단을 그대로 붙여 넣지 않는다.
- 수식은 슬라이드당 하나 또는 두 개만 사용한다.
- 표보다 막대그래프와 흐름도를 우선한다.
- `1.401`, `1.655`, `15.34%`, `5/5·25/25·150/150`을 가장 눈에 띄게 표시한다.
- 실험 범위 밖 stress 결과는 본문에 넣지 않고 질문이 있을 때만 답한다.

---

# 발표 직전 기억할 네 가지

1. **무엇을 했는가:** PPO가 \(\alpha,q_k,\phi_k\)를 결정한다.
2. **왜 RSMA인가:** \(K=3>N_t=2\) 과부하에서 private beam만으로 간섭 제거가 어렵다.
3. **무엇이 좋아졌는가:** RSMA-PPO 1.401, SDMA-PPO 1.655, AoI 15.34% 감소.
4. **어디까지 주장하는가:** 지정된 채널·CSIT·학습 범위에서의 시스템 수준 비교이며, common stream만의 순수 인과효과나 모든 조건에서의 우월성은 주장하지 않는다.
