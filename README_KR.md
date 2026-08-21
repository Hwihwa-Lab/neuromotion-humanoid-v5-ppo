# 🤖 NEUROMOTION 3.0 // Humanoid-v5 Telemetry & RL

[![Language: English](https://img.shields.io/badge/Language-English-blue)](README.md)
[![Language: 한국어](https://img.shields.io/badge/Language-한국어-green)](README_KR.md)
[![Hugging Face Model Hub](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20Hub-orange)](https://huggingface.co/hwihwalab/neuromotion-humanoid-v5-ppo)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=flat&logo=github)](https://github.com/Hwihwa-Lab/neuromotion-humanoid-v5-ppo)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-MuJoCo%20Humanoid--v5-0080FF)](https://gymnasium.farama.org/environments/mujoco/humanoid/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-brightgreen)](https://stable-baselines3.readthedocs.io)

> **MuJoCo Humanoid-v5 PPO Continuous Control & Real-time Engineering Telemetry Dashboard**
> *[ 🌐 English Documentation ](README.md) | [ 🇰🇷 한국어 매뉴얼 ](README_KR.md)*

천재디자인에이전트 디자인 시스템과 **인터랙티브 드롭다운 메뉴(Dropdown Menu)**가 장착된 **`NEUROMOTION 3.0`** 공식 텔레메트리 매뉴얼입니다.

---

## 🚀 Quick Start & Launch Modes (실행 방법)

### 1. 🖥️ 메인 GUI 대시보드 실행 (추천)
Pygame 기반 실시간 다크 사이버펑크 텔레메트리 대시보드를 실행합니다:
```powershell
python run_gui.py
```

### 2. 🤗 Hugging Face 모델 허브 배포 (`deploy_to_hf.py`)
학습된 모델 가중치(`humanoid_ppo_model.zip`)와 실행 코드를 Hugging Face에 원클릭으로 업로드합니다:
```powershell
# 전체 모델 및 코드 업로드
python deploy_to_hf.py

# 파일 업로드 없이 리포지토리(hwihwalab/neuromotion-humanoid-v5-ppo)만 생성할 때
python deploy_to_hf.py --create-only
```

> 💡 **향후 모델 업데이트 방법**
> 앞으로 추가 학습을 진행한 후 가중치나 코드를 다시 배포하고 싶으실 때는 언제든 아래 명령어 한 줄만 실행하시면 됩니다:
> ```powershell
> python deploy_to_hf.py
> ```

### 3. 💻 터미널 다중 모드 실행기 (`mujoko_humanoid.py`)
터미널 대화형 메뉴 또는 명령줄 옵션을 통해 다양한 목적별 학습/시연을 실행합니다:
```powershell
# 대화형 번호 선택 메뉴 실행
python mujoko_humanoid.py

# 또는 원하는 모드를 직접 지정하여 실행
python mujoko_humanoid.py --mode train_live   # [모드 1] 실시간 3D 시각화 + 강화학습
python mujoko_humanoid.py --mode train_fast   # [모드 2] 고속 백그라운드 학습 + 주기적 3D 평가
python mujoko_humanoid.py --mode play         # [모드 3] 학습된 모델(humanoid_ppo_model.zip) 3D 보행 시연
python mujoko_humanoid.py --mode random       # [모드 4] 초기 무작위(Random) 행동 관찰
```

---

## 🖱️ Top Control Header (상단 인터랙티브 버튼 & 드롭다운 메뉴)

상단 버튼들은 마우스 호버 시 **손가락 모양 커서(👆)**와 앰비언트 하이라이트가 적용되며, 드롭다운 메뉴를 통해 원하는 옵션을 즉시 콕 찍어 선택할 수 있습니다.

| Button Name | Type | Options & Function Description |
| :--- | :---: | :--- |
| **`[ ● PPO TRAINING ]` / `[ ⏸ PAUSED ]`** | **Action** | **Telemetry Pause / Resume**: AI 학습 및 시뮬레이션 일시정지 / 재개 (단축키: `Space`) |
| **`[ SPEED: 1X ▼ ]`** | **Dropdown** | **클릭 시 배속 선택 메뉴 팝업**:<br>• `1X  Normal (1x)` : 기본 세밀 관찰 속도<br>• `2X  Fast (2x)` : 2배속 학습<br>• `4X  Hyper (4x)` : 4배속 고속 학습<br>• `8X  Ultra (8x)` : 8배속 초고속 집중 학습 (단축키: `F` 로 빠른 순환 가능) |
| **`[ PUSH FORCE ▼ ]`** | **Dropdown** | **클릭 시 외란 충격 세기 선택 팝업**:<br>• `Light Push  (15 N)` : 가벼운 툭 밀기<br>• `Medium Push (30 N)` : 중간 균형 흔들림 (기본 외란)<br>• `Heavy Force (50 N)` : 강한 충격파 테스트 (단축키: `P` 로 기본 30N 인가) |
| **`[ SAVE MODEL ]`** | **Action** | **Save Checkpoint Weights**: 현재 학습된 신경망 가중치를 `humanoid_ppo_model.zip`으로 저장 (단축키: `S`) |
| **`[ LOAD MODEL ]`** | **Action** | **Load Checkpoint Weights**: 저장된 모델 가중치 즉시 불러오기 (단축키: `L`) |

---

## 🖥️ Telemetry Layout & Panels (7대 핵심 텔레메트리 패널)

1. **MAIN 3D SIMULATION VIEW**:
   * MuJoCo Humanoid-v5 실시간 3D 물리 렌더링 화면
   * 상단 인포바: `EPISODE` // `ALIVE STEPS` // `NOISE SCALE (σ)`
2. **TOP BENTO STATS CARDS**:
   * `EPISODE REWARD`: 이번 에피소드 실시간 누적 보상
   * `PEAK REWARD`: 역대 최고 기록 보상 점수
   * `20-EP MOVING AVG`: 최근 20 에피소드 이동 평균선
   * `TOTAL TIMESTEPS`: 환경 상호작용 누적 스텝수
3. **LEARNING DIAGNOSTICS**:
   * `Policy Gradient Loss`: 정책 신경망 손실 지표
   * `Value Function Loss`: 가치 평가 신경망 손실 지표
   * `Entropy`: 탐색 무작위성 척도 및 학습 업데이트 횟수
4. **ACTUATOR TORQUES (17 DOF)**:
   * 17개 관절 모터(복부, 고관절, 무릎, 어깨, 팔꿈치 등)의 실시간 힘 방향 및 세기 듀얼톤 게이지
5. **REWARD TELEMETRY CURVE**:
   * 최근 100개 에피소드 보상 꺾은선 차트 (하단 에어리어 필 & 동적 스케일링)
6. **SYSTEM CONSOLE LOG**:
   * 에피소드 완료 및 텔레메트리 이벤트 실시간 모노스페이스 로그 (100% 박스 내부 클리핑)
7. **FOOTER SHORTCUTS**:
   * 키보드 단축키 가이드 바 (`[Space]` `[T]` `[F]` `[P]` `[S]` `[L]` `[R/E]`)

---

## ⌨️ Keyboard Shortcuts Reference

| Key | Action | Description |
| :---: | :--- | :--- |
| **`Space`** | **Pause / Resume** | 학습 및 3D 물리 시뮬레이션 정지/재개 |
| **`F`** | **Cycle Speed** | 1X ➡️ 2X ➡️ 4X ➡️ 8X 배속 순환 |
| **`P`** | **Push Robot** | 로봇에게 외란 충격(30.0 N) 인가 |
| **`S`** | **Save Model** | 최신 가중치를 `humanoid_ppo_model.zip`으로 저장 |
| **`L`** | **Load Model** | 저장된 가중치 불러오기 |
| **`T`** | **Toggle 3D View** | 3D 렌더링 ON / OFF |
| **`R` / `E`** | **Reset Episode** | 환경 즉시 초기화 |

---

## 📊 Model Evaluation & Benchmarks

100,000 스텝 추가 훈련 후 10개 에피소드 정밀 평가 결과:
* **평균 생존 스텝**: `88.5 steps` (최고: `109 steps`)
* **평균 누적 보상**: `455.52` (최고: `549.18`)
