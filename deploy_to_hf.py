"""
NeuroMotion // Hugging Face Model Hub Deployer
------------------------------------------------
Hugging Face에 MuJoCo Humanoid-v5 PPO 강화학습 모델 및 코드를 자동 업로드하는 스크립트입니다.

사용법:
    # 1. 리포지토리 생성 및 전체 파일 업로드 (기본)
    python deploy_to_hf.py

    # 2. 파일 업로드 없이 빈 리포지토리만 먼저 생성할 때
    python deploy_to_hf.py --create-only

    # 3. 비공개(Private) 리포지토리로 생성할 때
    python deploy_to_hf.py --private
"""

import os
import sys
import argparse
from huggingface_hub import HfApi, get_token, login


DEFAULT_REPO_NAME = "neuromotion-humanoid-v5-ppo"


HF_MODEL_CARD_TEMPLATE = """---
language:
- en
- ko
tags:
- reinforcement-learning
- stable-baselines3
- ppo
- continuous-control
- mujoco
- humanoid-v5
- robotics
- robot
- bipedal-robot
- neuromotion
pipeline_tag: reinforcement-learning
library_name: stable-baselines3
---

# 🤖 NeuroMotion // Humanoid-v5 PPO Continuous Control

[![Hugging Face Hub](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20Hub-orange)](https://huggingface.co/{repo_id})
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=flat&logo=github)](https://github.com/Hwihwa-Lab/neuromotion-humanoid-v5-ppo)

> **MuJoCo Humanoid-v5 Continuous Control Telemetry & PPO Reinforcement Learning System**  
> *[ 🌐 English Documentation ](https://huggingface.co/{repo_id}/blob/main/README.md) | [ 🇰🇷 한국어 매뉴얼 ](https://huggingface.co/{repo_id}/blob/main/README_KR.md)*

This repository contains an advanced continuous reinforcement learning system (PPO) and a real-time engineering telemetry dashboard for bipedal robotic continuous control in [Gymnasium](https://gymnasium.farama.org/environments/mujoco/humanoid/) MuJoCo `Humanoid-v5`.

---

## 📊 Model Specifications & Benchmark Results

| Parameter | Specification |
| :--- | :--- |
| **Environment** | Gymnasium MuJoCo `Humanoid-v5` |
| **Observation Space** | 378-dimensional continuous vector |
| **Action Space** | 17-dimensional continuous joint torques (Box[-1.0, 1.0]) |
| **Algorithm** | Proximal Policy Optimization (PPO) |
| **Framework** | Stable-Baselines3 / PyTorch |
| **Architecture** | Actor-Critic MLP Policy (MlpPolicy) |
| **Average Survival Steps** | **`88.5 steps`** *(Peak: `109 steps`)* |
| **Average Cumulative Reward** | **`+455.52`** *(Peak: `+549.18`)* |

---

## 🚀 Quick Start (Inference & Evaluation)

### 1. Download & Load with Stable-Baselines3

```python
import gymnasium as gym
from stable_baselines3 import PPO
from huggingface_hub import hf_hub_download

# Download model weights from Hugging Face Hub
model_file = hf_hub_download(
    repo_id="{repo_id}",
    filename="humanoid_ppo_model.zip"
)

# Initialize MuJoCo Humanoid-v5 environment
env = gym.make("Humanoid-v5", render_mode="human")
model = PPO.load(model_file, env=env)

# Run evaluation episodes
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

---

## 🖥️ Real-time Telemetry Dashboard (NeuroMotion Studio)

The repository also includes a native 60fps Pygame telemetry cockpit:

```powershell
python run_gui.py
```

### Features:
* **7-Panel Telemetry Layout**: 17-DOF Actuator Torque Bars, Learning Diagnostics (Policy/Value Loss, Entropy), Reward Telemetry Curves, Top Bento KPI Cards.
* **Interactive Controls**: Real-time Physics Speed Multipliers (1x, 2x, 4x, 8x), Disturbance Force Injection (15N, 30N, 50N), Checkpoint Save/Load.

---

## 📂 Repository Contents

* `README.md`: Hugging Face model card, specifications, and telemetry manual.
* `README_KR.md`: Dedicated Korean manual ([한국어 매뉴얼](https://github.com/Hwihwa-Lab/neuromotion-humanoid-v5-ppo/blob/main/README_KR.md)).
* `humanoid_ppo_model.zip`: Pre-trained PPO neural network weights.
* `simulation_engine.py`: Gymnasium MuJoCo physical engine & 17-DOF torque extractor.
* `train_rl.py`: Stable-Baselines3 PPO incremental trainer & diagnostics.
* `run_gui.py`: Pygame real-time telemetry cockpit.
* `mujoko_humanoid.py`: Multi-mode CLI execution and visualization script.

---

*Trained and deployed with [NeuroMotion Studio](https://huggingface.co/{repo_id}) by **hwihwalab**.*
"""


def get_authenticated_api(token_arg: str = None) -> tuple[HfApi, str]:
    """Hugging Face API 인증 및 사용자명을 확인합니다."""
    token = token_arg or os.environ.get("HF_TOKEN") or get_token()

    if not token:
        print("\n[!] Hugging Face 인증 토큰을 찾을 수 없습니다.")
        print("    토큰을 등록하려면 터미널에 아래 명령어를 실행하거나:")
        print("    > huggingface-cli login")
        print("    또는 본 스크립트 실행 시 --token 옵션으로 전달해주세요.\n")
        user_input = input("Hugging Face Access Token을 입력하세요 (Enter=취소): ").strip()
        if user_input:
            token = user_input
            login(token=token)
        else:
            sys.exit(1)

    api = HfApi(token=token)
    try:
        user_info = api.whoami()
        username = user_info.get("name")
        print(f"[+] Hugging Face 로그인 성공! (계정: {username})")
        return api, username
    except Exception as e:
        print(f"[!] 인증 실패: {e}")
        sys.exit(1)


def deploy(repo_name: str, create_only: bool = False, private: bool = False, token_arg: str = None):
    print("=" * 65)
    print(" 🚀 NeuroMotion // Hugging Face Hub Deployment")
    print("=" * 65)

    api, username = get_authenticated_api(token_arg)

    # 리포지토리 ID 구성 (예: hwihwalab/neuromotion-humanoid-v5-ppo)
    repo_id = f"{username}/{repo_name}" if "/" not in repo_name else repo_name

    print(f"\n[1/3] Hugging Face Model Repository 생성 확인 중...")
    print(f"      Target Repo: https://huggingface.co/{repo_id}")

    try:
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        print(f"[+] 리포지토리 준비 완료: {repo_url}")
    except Exception as e:
        print(f"[!] 리포지토리 생성 중 오류: {e}")
        sys.exit(1)

    if create_only:
        print("\n[+] --create-only 옵션이 지정되어 파일 업로드 없이 리포지토리를 생성했습니다.")
        print(f"    🔗 저장소 주소: https://huggingface.co/{repo_id}\n")
        return

    # 업로드 대상 파일 목록
    print(f"\n[2/3] 업로드 대상 파일 준비 및 모델 카드 생성 중...")
    workspace_dir = os.path.dirname(os.path.abspath(__file__))

    files_to_upload = [
        "README_KR.md",
        "humanoid_ppo_model.zip",
        "simulation_engine.py",
        "train_rl.py",
        "run_gui.py",
        "mujoko_humanoid.py",
    ]

    # Hugging Face용 README(Model Card) 임시 생성 후 업로드
    model_card_content = HF_MODEL_CARD_TEMPLATE.format(repo_id=repo_id)
    temp_readme_path = os.path.join(workspace_dir, "HF_MODEL_CARD.md")
    with open(temp_readme_path, "w", encoding="utf-8") as f:
        f.write(model_card_content)

    print(f"\n[3/3] Hugging Face Hub로 파일 업로드 진행 중...")

    # README.md 업로드
    try:
        api.upload_file(
            path_or_fileobj=temp_readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="docs: add NeuroMotion Humanoid-v5 PPO model card & metadata",
        )
        print("  ✓ README.md (Hugging Face Model Card) 업로드 완료")
    except Exception as e:
        print(f"  ✗ README.md 업로드 실패: {e}")

    # 개별 파일 업로드
    for filename in files_to_upload:
        file_path = os.path.join(workspace_dir, filename)
        if os.path.exists(file_path):
            file_size_kb = os.path.getsize(file_path) / 1024
            print(f"  → 업로드 중: {filename} ({file_size_kb:.1f} KB)...", end="", flush=True)
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"feat: upload {filename}",
                )
                print(" [완료]")
            except Exception as e:
                print(f" [실패: {e}]")
        else:
            print(f"  - 스킵 (파일 없음): {filename}")

    # 임시 파일 정리
    if os.path.exists(temp_readme_path):
        os.remove(temp_readme_path)

    print("\n" + "=" * 65)
    print(" 🎉 NeuroMotion Hugging Face 배포가 성공적으로 완료되었습니다!")
    print(f" 🔗 모델 허브 URL: https://huggingface.co/{repo_id}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="NeuroMotion Hugging Face Model Hub Deployer")
    parser.add_argument(
        "--repo-name",
        type=str,
        default=DEFAULT_REPO_NAME,
        help=f"Hugging Face 리포지토리 이름 (기본값: {DEFAULT_REPO_NAME})",
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="파일 업로드 없이 빈 리포지토리만 생성합니다.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="비공개(Private) 리포지토리로 생성합니다.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face User Access Token (생략 시 로컬 인증 토큰 자동 사용)",
    )
    args = parser.parse_args()

    deploy(
        repo_name=args.repo_name,
        create_only=args.create_only,
        private=args.private,
        token_arg=args.token,
    )


if __name__ == "__main__":
    main()
