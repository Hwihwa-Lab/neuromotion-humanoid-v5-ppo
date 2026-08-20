import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ==========================================================
# 1. 실시간 그래프 시각화 콜백
# ==========================================================
class RealtimePlotCallback(BaseCallback):
    """학습 도중 실시간으로 에피소드 보상 및 에피소드 길이를 그래프로 업데이트합니다."""

    def __init__(self, plot_freq_episodes: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.plot_freq = plot_freq_episodes
        self.episode_rewards = []
        self.episode_lengths = []
        self.moving_avg_rewards = []
        self.current_ep_reward = 0.0
        self.current_ep_length = 0
        self.fig = None
        self.ax1 = None
        self.ax2 = None

    def _on_training_start(self) -> None:
        # 대화형 모드(interactive mode) 활성화
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.canvas.manager.set_window_title("Humanoid-v5 학습 실시간 통계")
        self.fig.tight_layout(pad=3.0)
        plt.show(block=False)

    def _on_step(self) -> bool:
        # 보상 및 스텝 카운트 누적
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if rewards is not None:
            self.current_ep_reward += float(rewards[0])
            self.current_ep_length += 1

        if dones is not None and dones[0]:
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)

            # 이동 평균 계산 (최근 10 에피소드)
            window = min(10, len(self.episode_rewards))
            moving_avg = np.mean(self.episode_rewards[-window:])
            self.moving_avg_rewards.append(moving_avg)

            # 그래프 업데이트
            if len(self.episode_rewards) % self.plot_freq == 0:
                self._update_plot()

            self.current_ep_reward = 0.0
            self.current_ep_length = 0

        return True

    def _update_plot(self):
        if not plt.fignum_exists(self.fig.number):
            return

        self.ax1.clear()
        self.ax2.clear()

        eps = range(1, len(self.episode_rewards) + 1)
        self.ax1.plot(eps, self.episode_rewards, label="Episode Reward", color="#4CAF50", alpha=0.5)
        self.ax1.plot(eps, self.moving_avg_rewards, label="10-Ep Moving Avg", color="#1E88E5", linewidth=2)
        self.ax1.set_title("Episode Reward Trend")
        self.ax1.set_xlabel("Episode")
        self.ax1.set_ylabel("Total Reward")
        self.ax1.legend(loc="upper left")
        self.ax1.grid(True, linestyle="--", alpha=0.6)

        self.ax2.plot(eps, self.episode_lengths, label="Survival Steps", color="#FF9800", linewidth=1.5)
        self.ax2.set_title("Survival Timesteps per Episode")
        self.ax2.set_xlabel("Episode")
        self.ax2.set_ylabel("Steps")
        self.ax2.legend(loc="upper left")
        self.ax2.grid(True, linestyle="--", alpha=0.6)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _on_training_end(self) -> None:
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.ioff()
            plt.show(block=False)


# ==========================================================
# 2. 주기적 3D 렌더링 시연 콜백
# ==========================================================
class PeriodicVisualEvalCallback(BaseCallback):
    """학습 도중 일정 스텝마다 3D 렌더링 창을 띄워 현재 정책의 보행 성능을 시각적으로 보여줍니다."""

    def __init__(self, eval_freq: int = 10000, n_eval_episodes: int = 1, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"\n[시각화 평가] {self.n_calls} 스텝 도달! 현재 정책의 보행 모습을 3D 윈도우로 시연합니다...")

            eval_env = gym.make("Humanoid-v5", render_mode="human")
            for ep in range(self.n_eval_episodes):
                obs, _ = eval_env.reset()
                done = False
                total_reward = 0.0
                step_count = 0
                while not done:
                    # 현재 모델로 행동 예측 (결정론적 정책)
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    total_reward += reward
                    step_count += 1
                    done = terminated or truncated
                    time.sleep(0.01)  # 감상하기 좋은 속도로 조절

                if self.verbose > 0:
                    print(f"  > 평가 에피소드 {ep + 1}: 총 보상 = {total_reward:.2f}, 생존 스텝 = {step_count}")
            eval_env.close()

        return True


# ==========================================================
# 3. 환경 생성 헬퍼 함수
# ==========================================================
def make_humanoid_env(render_mode=None):
    """Humanoid-v5 환경을 생성합니다."""
    return gym.make("Humanoid-v5", render_mode=render_mode)


# ==========================================================
# 4. 각 모드별 실행 로직
# ==========================================================
def run_random_demo(n_steps=500):
    """학습 전 기본 무작위 행동 시각화 (초기 상태 관찰)"""
    print("\n" + "=" * 60)
    print(" [모드 4] 무작위(Random) 행동 3D 시각화 시연")
    print(" 로봇이 학습되지 않은 상태에서 어떻게 넘어지는지 관찰합니다.")
    print("=" * 60)

    env = gym.make("Humanoid-v5", render_mode="human")
    obs, info = env.reset(seed=42)

    total_reward = 0.0
    ep_count = 1

    for step in range(n_steps):
        action = env.action_space.sample()  # 무작위 액션
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        time.sleep(0.015)  # 3D 뷰어 감상을 위한 딜레이

        if terminated or truncated:
            print(f"  [에피소드 {ep_count}] {step + 1}번째 스텝에서 종료 (누적 보상: {total_reward:.2f})")
            obs, info = env.reset()
            total_reward = 0.0
            ep_count += 1

    env.close()
    print("무작위 행동 시연이 완료되었습니다.\n")


def run_live_training(total_timesteps=50000, model_save_path="humanoid_ppo_model.zip"):
    """실시간 3D 렌더링 윈도우와 그래프를 보면서 직접 학습 진행"""
    print("\n" + "=" * 60)
    print(" [모드 1] 실시간 3D 시각화 + 강화학습 (Live Visual Training)")
    print(" 3D 화면으로 로봇이 넘어지고 일어서며 학습하는 모습과 실시간 그래프를 관찰합니다.")
    print(f" 목표 학습 타임스텝: {total_timesteps:,} steps")
    print("=" * 60)

    # 3D 렌더링 환경 생성
    env = make_humanoid_env(render_mode="human")

    # PPO 에이전트 생성
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
    )

    plot_cb = RealtimePlotCallback(plot_freq_episodes=1)

    try:
        model.learn(total_timesteps=total_timesteps, callback=[plot_cb])
        print(f"\n학습 완료! 모델을 저장합니다 -> {model_save_path}")
        model.save(model_save_path)
    except KeyboardInterrupt:
        print("\n사용자에 의해 학습이 중단되었습니다. 현재까지의 모델을 저장합니다.")
        model.save(model_save_path)
    finally:
        env.close()


def run_fast_training(total_timesteps=100000, eval_freq=10000, model_save_path="humanoid_ppo_model.zip"):
    """고속 백그라운드 학습 + 주기적 3D 평가 시연 + 실시간 통계 그래프"""
    print("\n" + "=" * 60)
    print(" [모드 2] 고속 학습 + 주기적 3D 시연 (Fast Training + Periodic Eval)")
    print(f" 빠른 속도로 학습하면서 매 {eval_freq:,} 스텝마다 3D 창으로 학습 성과를 시연합니다.")
    print(f" 목표 학습 타임스텝: {total_timesteps:,} steps")
    print("=" * 60)

    # 고속 학습용 환경 (render_mode=None)
    env = make_humanoid_env(render_mode=None)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
    )

    plot_cb = RealtimePlotCallback(plot_freq_episodes=2)
    visual_cb = PeriodicVisualEvalCallback(eval_freq=eval_freq, n_eval_episodes=1)

    try:
        model.learn(total_timesteps=total_timesteps, callback=[plot_cb, visual_cb])
        print(f"\n학습 완료! 모델을 저장합니다 -> {model_save_path}")
        model.save(model_save_path)
    except KeyboardInterrupt:
        print("\n사용자에 의해 학습이 중단되었습니다. 현재까지의 모델을 저장합니다.")
        model.save(model_save_path)
    finally:
        env.close()


def run_watch_trained_model(model_save_path="humanoid_ppo_model.zip", n_episodes=5):
    """저장된 모델을 불러와 3D 화면으로 휴머노이드 보행 모션 감상"""
    print("\n" + "=" * 60)
    print(" [모드 3] 학습된 모델 3D 시연 (Watch Trained Model)")
    print(f" 모델 경로: {model_save_path}")
    print("=" * 60)

    if not os.path.exists(model_save_path):
        print(f"오류: '{model_save_path}' 파일이 존재하지 않습니다.")
        print("먼저 [모드 1] 또는 [모드 2]로 학습을 진행하여 모델을 생성해주세요.")
        return

    env = gym.make("Humanoid-v5", render_mode="human")
    model = PPO.load(model_save_path, env=env)

    print(f"총 {n_episodes}개 에피소드 동안 학습된 모델의 보행을 시연합니다.\n")

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            time.sleep(0.015)  # 60fps에 맞춘 부드러운 시각화

        print(f"  [에피소드 {ep + 1}/{n_episodes}] 완료 - 생존 스텝: {step_count}, 총 보상: {total_reward:.2f}")

    env.close()
    print("\n시연이 완료되었습니다.")


# ==========================================================
# 5. 메인 진입점 및 대화형 메뉴
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="MuJoCo Humanoid-v5 실시간 시각화 강화학습")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train_live", "train_fast", "play", "random"],
        default=None,
        help="실행 모드: train_live (실시간 시각화 학습), train_fast (고속 학습+주기적 시연), play (저장된 모델 시연), random (무작위 동작)",
    )
    parser.add_argument("--timesteps", type=int, default=50000, help="총 학습 스텝 수 (기본: 50,000)")
    parser.add_argument("--eval_freq", type=int, default=10000, help="고속 학습 시 3D 시각화 평가 주기 스텝 (기본: 10,000)")
    parser.add_argument("--model_path", type=str, default="humanoid_ppo_model.zip", help="모델 저장/로드 경로")
    args = parser.parse_args()

    # 인자로 모드가 지정된 경우 한 번만 실행하고 종료
    if args.mode is not None:
        if args.mode == "train_live":
            run_live_training(total_timesteps=args.timesteps, model_save_path=args.model_path)
        elif args.mode == "train_fast":
            run_fast_training(total_timesteps=args.timesteps, eval_freq=args.eval_freq, model_save_path=args.model_path)
        elif args.mode == "play":
            run_watch_trained_model(model_save_path=args.model_path)
        elif args.mode == "random":
            run_random_demo(n_steps=args.timesteps if args.timesteps != 50000 else 500)
        return

    # 인자 없이 실행했을 때는 메뉴가 계속 유지되어 번호를 연속해서 선택할 수 있도록 구성
    while True:
        print("\n" + "=" * 65)
        print("   🤖 MuJoCo Humanoid-v5 실시간 시각화 강화학습 시스템 🤖")
        print("=" * 65)
        print("  1. [실시간 3D 시각화 학습]   - 3D 화면 & 그래프를 보면서 실시간 학습")
        print("  2. [고속 학습 + 주기적 시연] - 빠른 학습 + N 스텝마다 3D 창 시연")
        print("  3. [학습된 모델 시연]       - 저장된 모델로 휴머노이드 보행 감상")
        print("  4. [무작위 동작 시연]       - 학습 전 기본 무작위 상태 관찰")
        print("  q. [프로그램 종료]")
        print("=" * 65)
        choice = input("실행할 번호를 입력하세요 (1-4, q=종료): ").strip().lower()

        if choice == "1":
            run_live_training(total_timesteps=args.timesteps, model_save_path=args.model_path)
        elif choice == "2":
            run_fast_training(total_timesteps=args.timesteps, eval_freq=args.eval_freq, model_save_path=args.model_path)
        elif choice == "3":
            run_watch_trained_model(model_save_path=args.model_path)
        elif choice == "4":
            run_random_demo(n_steps=500)
        elif choice in ["q", "quit", "exit"]:
            print("\n프로그램을 종료합니다. 감사합니다!")
            break
        else:
            print("\n올바른 번호(1~4 또는 q)를 입력해주세요.")


if __name__ == "__main__":
    main()

