import warnings
warnings.filterwarnings("ignore")

import numpy as np
import gymnasium as gym
import pygame


class SimulationEngine:
    """MuJoCo Humanoid 시뮬레이션을 관리하는 초경량 엔진"""

    JOINT_NAMES = [
        "Abdomen Z", "Abdomen Y", "Abdomen X",
        "R Hip X", "R Hip Z", "R Hip Y", "R Knee",
        "L Hip X", "L Hip Z", "L Hip Y", "L Knee",
        "R Shoulder 1", "R Shoulder 2", "R Elbow",
        "L Shoulder 1", "L Shoulder 2", "L Elbow"
    ]

    def __init__(self, render_width: int = 590, render_height: int = 420):
        self.render_width = render_width
        self.render_height = render_height
        
        # Humanoid-v5 환경 생성
        self.env = gym.make("Humanoid-v5", render_mode="rgb_array", width=render_width, height=render_height)
        self.observation, self.info = self.env.reset(seed=42)
        
        self.current_action = np.zeros(17, dtype=np.float32)
        self.current_torques = np.zeros(17, dtype=np.float32)
        self.alive_steps = 0
        self.episode_reward = 0.0
        self.is_done = False

    def reset(self, seed=None):
        """환경을 초기화합니다."""
        self.observation, self.info = self.env.reset(seed=seed)
        self.current_action = np.zeros(17, dtype=np.float32)
        self.current_torques = np.zeros(17, dtype=np.float32)
        self.alive_steps = 0
        self.episode_reward = 0.0
        self.is_done = False
        return self.observation

    def step(self, action: np.ndarray):
        """행동을 수행하고 다음 상태와 보상을 반환합니다."""
        self.current_action = np.clip(action, -1.0, 1.0)
        self.observation, reward, terminated, truncated, self.info = self.env.step(self.current_action)
        
        self.episode_reward += float(reward)
        self.alive_steps += 1
        self.is_done = terminated or truncated

        # 실제 관절 모터 출력 토크 반영
        try:
            raw_torques = self.env.unwrapped.data.qfrc_actuator[-17:]
            max_t = np.max(np.abs(raw_torques)) + 1e-4
            norm_torques = raw_torques / max_t
            self.current_torques = np.clip(norm_torques, -1.0, 1.0)
        except Exception:
            self.current_torques = self.current_action

        return self.observation, reward, self.is_done, self.info

    def render_surface(self) -> pygame.Surface:
        """현재 시뮬레이션 프레임을 Pygame Surface로 변환하여 반환합니다."""
        frame = self.env.render()
        if frame is None:
            surf = pygame.Surface((self.render_width, self.render_height))
            surf.fill((20, 25, 35))
            return surf

        # (H, W, 3) -> Pygame Surface
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        return surface

    def apply_push_disturbance(self, force_magnitude: float = 30.0):
        """로봇에게 수평 외력을 가해 균형 유지 능력을 테스트합니다."""
        try:
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, "data") and hasattr(unwrapped.data, "qvel"):
                unwrapped.data.qvel[0] += (np.random.choice([-1.0, 1.0]) * force_magnitude)
                unwrapped.data.qvel[1] += (np.random.uniform(-0.5, 0.5) * force_magnitude)
                return True
        except Exception:
            pass
        return False

    def close(self):
        """환경 종료"""
        self.env.close()
