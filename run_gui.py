import os
import warnings
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore")

import collections
import numpy as np
import pygame

from simulation_engine import SimulationEngine
from train_rl import RLTrainer


# ==========================================================
# 천재디자인에이전트 토큰 시스템 (Slate & Semantic Neon)
# ==========================================================
COLOR_BG = (10, 14, 23)              # Slate 950 (Deep Slate Navy)
COLOR_PANEL = (17, 24, 39)           # Slate 900
COLOR_PANEL_HOVER = (30, 41, 59)     # Slate 800
COLOR_PANEL_BORDER = (30, 41, 59)    # Slate 800 (1px Precision Border)
COLOR_PANEL_SUB = (13, 18, 30)       # Deep Sub-panel Surface
COLOR_DROPDOWN_BG = (15, 23, 42)     # Dropdown Menu Background

# Semantic Accent Tokens
COLOR_CYAN = (6, 182, 212)           # Cyan 500
COLOR_EMERALD = (16, 185, 129)       # Emerald 500
COLOR_AMBER = (245, 158, 11)         # Amber 500
COLOR_ROSE = (244, 63, 94)           # Rose 500
COLOR_VIOLET = (139, 92, 246)        # Violet 500
COLOR_MAGENTA = (217, 70, 239)       # Fuchsia 500

# Typography Tokens (WCAG 4.5:1 High Contrast)
COLOR_TEXT_PRIMARY = (248, 250, 252) # Slate 50
COLOR_TEXT_SECONDARY = (148, 163, 184) # Slate 400
COLOR_TEXT_MUTED = (100, 116, 139)   # Slate 500


class RobotAILabGUI:
    """NEUROMOTION 3.0 마스터 대시보드 메인 클래스"""

    def __init__(self, width=1280, height=820):
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("NEUROMOTION 3.0 // Humanoid-v5 Continuous Control Telemetry")

        self.clock = pygame.time.Clock()
        self.is_running = True
        self.is_paused = False
        self.render_3d = True

        # 배속 제어
        self.speed_multiplier = 1
        self.speed_options = [
            (1, "1X  Normal (1x)"),
            (2, "2X  Fast (2x)"),
            (4, "4X  Hyper (4x)"),
            (8, "8X  Ultra (8x)"),
        ]

        # 외란 세기 옵션
        self.push_options = [
            (15.0, "Light Push  (15 N)"),
            (30.0, "Medium Push (30 N)"),
            (50.0, "Heavy Force (50 N)"),
        ]

        # 드롭다운 상태 ('speed', 'push', None)
        self.active_dropdown = None

        # 시각 피드백 타이머
        self.push_alert_timer = 0
        self.push_alert_text = "[ DISTURBANCE INJECTED ]"
        self.save_alert_timer = 0

        # 모던 시스템 폰트 로드
        font_names = "segoeui,malgungothic,arial,helvetica"
        self.font_brand = pygame.font.SysFont(font_names, 18, bold=True)
        self.font_sub = pygame.font.SysFont(font_names, 12)
        self.font_card_val = pygame.font.SysFont(font_names, 24, bold=True)
        self.font_card_lbl = pygame.font.SysFont(font_names, 11, bold=True)
        self.font_mono = pygame.font.SysFont("consolas,monospace,courier", 12)
        self.font_small = pygame.font.SysFont(font_names, 11)
        self.font_alert = pygame.font.SysFont(font_names, 14, bold=True)

        # 엔진 및 트레이너 초기화
        print("[TELEMETRY] Initializing MuJoCo Simulation & PPO Engine...")
        self.sim = SimulationEngine(render_width=590, render_height=420)
        self.trainer = RLTrainer(model_path="humanoid_ppo_model.zip")

        # 텔레메트리 버퍼
        self.episode_count = 1
        self.best_reward = 0.0
        self.recent_rewards = collections.deque(maxlen=100)
        self.recent_20_rewards = collections.deque(maxlen=20)
        self.console_logs = collections.deque(maxlen=5)
        self.console_logs.append("[SYSTEM] NEUROMOTION 3.0 Telemetry Online.")

        # 마우스 인터랙션 버튼 배치
        self._init_buttons()

    def _init_buttons(self):
        """헤더 상단 제어 버튼 배치"""
        self.btn_pause = pygame.Rect(630, 12, 130, 32)
        self.btn_speed = pygame.Rect(770, 12, 125, 32)
        self.btn_push = pygame.Rect(905, 12, 135, 32)
        self.btn_save = pygame.Rect(1050, 12, 100, 32)
        self.btn_load = pygame.Rect(1160, 12, 100, 32)

        self.buttons = [
            ("pause", self.btn_pause),
            ("speed", self.btn_speed),
            ("push", self.btn_push),
            ("save", self.btn_save),
            ("load", self.btn_load),
        ]

    def run(self):
        """메인 텔레메트리 루프"""
        train_step_timer = 0

        while self.is_running:
            self._handle_events()

            if not self.is_paused:
                for _ in range(self.speed_multiplier):
                    obs = self.sim.observation
                    action = self.trainer.predict_action(obs, deterministic=False)
                    next_obs, reward, done, info = self.sim.step(action)

                    if done:
                        ep_rew = self.sim.episode_reward
                        ep_steps = self.sim.alive_steps
                        self.recent_rewards.append(ep_rew)
                        self.recent_20_rewards.append(ep_rew)
                        if ep_rew > self.best_reward:
                            self.best_reward = ep_rew

                        log_msg = f"[LOG] EP {self.episode_count:02d} // REW: {ep_rew:+06.1f} // {ep_steps:02d} STEPS"
                        self.console_logs.append(log_msg)
                        self.episode_count += 1
                        self.sim.reset()

                    train_step_timer += 1
                    if train_step_timer % 128 == 0:
                        self.trainer.step_learning(total_steps=128)

            if self.push_alert_timer > 0:
                self.push_alert_timer -= 1
            if self.save_alert_timer > 0:
                self.save_alert_timer -= 1

            self._draw_gui()
            pygame.display.flip()
            self.clock.tick(60)

        self.sim.close()
        pygame.quit()

    def _handle_events(self):
        """이벤트 처리 및 드롭다운 메뉴 인터랙션"""
        mouse_pos = pygame.mouse.get_pos()
        
        is_hovering_btn = any(rect.collidepoint(mouse_pos) for _, rect in self.buttons)
        is_hovering_dd = self._is_hovering_dropdown(mouse_pos)
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if (is_hovering_btn or is_hovering_dd) else pygame.SYSTEM_CURSOR_ARROW)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.active_dropdown:
                    handled = self._handle_dropdown_click(mouse_pos)
                    if handled:
                        continue

                if self.btn_pause.collidepoint(mouse_pos):
                    self.is_paused = not self.is_paused
                    self.trainer.is_training_active = not self.is_paused
                    self.active_dropdown = None

                elif self.btn_speed.collidepoint(mouse_pos):
                    self.active_dropdown = None if self.active_dropdown == "speed" else "speed"

                elif self.btn_push.collidepoint(mouse_pos):
                    self.active_dropdown = None if self.active_dropdown == "push" else "push"

                elif self.btn_save.collidepoint(mouse_pos):
                    self.active_dropdown = None
                    self._trigger_save()

                elif self.btn_load.collidepoint(mouse_pos):
                    self.active_dropdown = None
                    self._trigger_load()

                else:
                    self.active_dropdown = None

            elif event.type == pygame.KEYDOWN:
                self.active_dropdown = None
                if event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                    self.trainer.is_training_active = not self.is_paused
                elif event.key == pygame.K_t:
                    self.render_3d = not self.render_3d
                elif event.key == pygame.K_f:
                    cur_idx = [opt[0] for opt in self.speed_options].index(self.speed_multiplier) if self.speed_multiplier in [1, 2, 4, 8] else 0
                    next_idx = (cur_idx + 1) % len(self.speed_options)
                    self.speed_multiplier = self.speed_options[next_idx][0]
                elif event.key == pygame.K_p:
                    self._trigger_push(30.0, "Medium (30 N)")
                elif event.key == pygame.K_s:
                    self._trigger_save()
                elif event.key == pygame.K_l:
                    self._trigger_load()
                elif event.key == pygame.K_e or event.key == pygame.K_r:
                    self.sim.reset()
                    self.console_logs.append("[RESET] Episode environment reset.")

    def _is_hovering_dropdown(self, mouse_pos) -> bool:
        if self.active_dropdown == "speed":
            menu_rect = pygame.Rect(self.btn_speed.x, self.btn_speed.bottom + 4, 155, len(self.speed_options) * 30 + 8)
            return menu_rect.collidepoint(mouse_pos)
        elif self.active_dropdown == "push":
            menu_rect = pygame.Rect(self.btn_push.x, self.btn_push.bottom + 4, 165, len(self.push_options) * 30 + 8)
            return menu_rect.collidepoint(mouse_pos)
        return False

    def _handle_dropdown_click(self, mouse_pos) -> bool:
        if self.active_dropdown == "speed":
            for i, (val, label) in enumerate(self.speed_options):
                item_rect = pygame.Rect(self.btn_speed.x, self.btn_speed.bottom + 6 + i * 30, 155, 28)
                if item_rect.collidepoint(mouse_pos):
                    self.speed_multiplier = val
                    self.active_dropdown = None
                    self.console_logs.append(f"[SPEED] Rate set to {val}X.")
                    return True

        elif self.active_dropdown == "push":
            for i, (force, label) in enumerate(self.push_options):
                item_rect = pygame.Rect(self.btn_push.x, self.btn_push.bottom + 6 + i * 30, 165, 28)
                if item_rect.collidepoint(mouse_pos):
                    self._trigger_push(force, label)
                    self.active_dropdown = None
                    return True

        return False

    def _trigger_push(self, force: float = 30.0, label: str = "30 N"):
        self.sim.apply_push_disturbance(force)
        self.push_alert_timer = 35
        self.push_alert_text = f"[ DISTURBANCE: +{force:.1f} N ]"
        self.console_logs.append(f"[TEST] Force injected (+{force:.0f} N)")

    def _trigger_save(self):
        self.trainer.save_model("humanoid_ppo_model.zip")
        self.save_alert_timer = 40
        self.console_logs.append("[SAVE] Checkpoint weights saved")

    def _trigger_load(self):
        if self.trainer.load_model("humanoid_ppo_model.zip"):
            self.console_logs.append("[LOAD] Checkpoint weights loaded")
        else:
            self.console_logs.append("[ERROR] Model file not found")

    def _draw_gui(self):
        """마스터 대시보드 렌더링"""
        self.screen.fill(COLOR_BG)

        self._draw_header()
        self._draw_main_3d_panel(x=20, y=60, w=590, h=420)
        self._draw_stat_cards(x=630, y=60, w=630, h=110)
        self._draw_learning_diagnostics(x=630, y=185, w=630, h=120)
        self._draw_actuator_torques(x=630, y=320, w=630, h=440)
        self._draw_reward_curve(x=20, y=495, w=290, h=265)
        self._draw_console_logs(x=320, y=495, w=290, h=265)
        self._draw_footer()

        # 최상단 드롭다운 메뉴 (앰비언트 드롭 섀도우 포함)
        self._draw_dropdown_menus()

    def _draw_header(self):
        """헤더 상단 제어 버튼 렌더링"""
        mouse_pos = pygame.mouse.get_pos()
        mouse_down = pygame.mouse.get_pressed()[0]

        # 브랜드 블록
        pygame.draw.rect(self.screen, COLOR_CYAN, (20, 15, 4, 26), border_radius=2)
        title_surf = self.font_brand.render("NEUROMOTION // HUMANOID-V5", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title_surf, (32, 14))

        sub_surf = self.font_mono.render("PPO CONTINUOUS CONTROL", True, COLOR_TEXT_MUTED)
        self.screen.blit(sub_surf, (325, 20))

        # 1. 일시정지 버튼
        status_col = COLOR_AMBER if self.is_paused else COLOR_EMERALD
        status_txt = "PAUSED" if self.is_paused else "PPO TRAINING"
        self._draw_engineering_button(self.btn_pause, status_txt, status_col, mouse_pos, mouse_down, dot=True)

        # 2. 배속 드롭다운 버튼
        speed_colors = {1: COLOR_CYAN, 2: COLOR_VIOLET, 4: COLOR_MAGENTA, 8: COLOR_AMBER}
        cur_col = speed_colors.get(self.speed_multiplier, COLOR_CYAN)
        speed_btn_txt = f"SPEED {self.speed_multiplier}X  v"
        self._draw_engineering_button(self.btn_speed, speed_btn_txt, cur_col, mouse_pos, mouse_down, is_active=(self.active_dropdown == "speed"))

        # 3. 외란 드롭다운 버튼
        self._draw_engineering_button(self.btn_push, "PUSH FORCE  v", COLOR_ROSE, mouse_pos, mouse_down, is_active=(self.active_dropdown == "push"))

        # 4. 저장 버튼
        self._draw_engineering_button(self.btn_save, "SAVE MODEL", COLOR_CYAN, mouse_pos, mouse_down)

        # 5. 로드 버튼
        self._draw_engineering_button(self.btn_load, "LOAD MODEL", COLOR_TEXT_SECONDARY, mouse_pos, mouse_down)

    def _draw_engineering_button(self, rect: pygame.Rect, text: str, accent_color, mouse_pos, mouse_down: bool, dot: bool = False, is_active: bool = False):
        """인터랙티브 버튼 렌더러"""
        is_hover = rect.collidepoint(mouse_pos)
        is_pressed = (is_hover and mouse_down) or is_active

        draw_rect = rect.move(0, 1) if is_pressed else rect
        bg_col = COLOR_PANEL_HOVER if (is_hover or is_active) else COLOR_PANEL

        pygame.draw.rect(self.screen, bg_col, draw_rect, border_radius=6)
        border_col = accent_color if (is_hover or is_active) else COLOR_PANEL_BORDER
        pygame.draw.rect(self.screen, border_col, draw_rect, width=1, border_radius=6)

        if dot:
            pygame.draw.circle(self.screen, accent_color, (draw_rect.x + 15, draw_rect.centery), 4)
            txt_surf = self.font_sub.render(text, True, COLOR_TEXT_PRIMARY if is_hover else accent_color)
            self.screen.blit(txt_surf, (draw_rect.x + 26, draw_rect.centery - 7))
        else:
            txt_surf = self.font_sub.render(text, True, accent_color if is_hover else COLOR_TEXT_PRIMARY)
            txt_rect = txt_surf.get_rect(center=draw_rect.center)
            self.screen.blit(txt_surf, txt_rect)

    def _draw_dropdown_menus(self):
        """최상단 오버레이 드롭다운 메뉴 (앰비언트 드롭 섀도우 탑재)"""
        mouse_pos = pygame.mouse.get_pos()

        if self.active_dropdown == "speed":
            menu_w = 155
            menu_h = len(self.speed_options) * 30 + 8
            menu_rect = pygame.Rect(self.btn_speed.x, self.btn_speed.bottom + 4, menu_w, menu_h)

            # 은은한 4px 반투명 앰비언트 섀도우
            shadow_s = pygame.Surface((menu_w + 8, menu_h + 8), pygame.SRCALPHA)
            shadow_s.fill((0, 0, 0, 140))
            self.screen.blit(shadow_s, (menu_rect.x - 4, menu_rect.y - 2))

            pygame.draw.rect(self.screen, COLOR_DROPDOWN_BG, menu_rect, border_radius=6)
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, menu_rect, width=1, border_radius=6)

            for i, (val, label) in enumerate(self.speed_options):
                item_rect = pygame.Rect(self.btn_speed.x + 4, self.btn_speed.bottom + 6 + i * 30, menu_w - 8, 26)
                is_hover = item_rect.collidepoint(mouse_pos)
                is_selected = (self.speed_multiplier == val)

                if is_hover or is_selected:
                    pygame.draw.rect(self.screen, COLOR_PANEL_HOVER, item_rect, border_radius=4)

                col = COLOR_CYAN if is_selected else (COLOR_TEXT_PRIMARY if is_hover else COLOR_TEXT_SECONDARY)
                txt_surf = self.font_sub.render(label, True, col)
                self.screen.blit(txt_surf, (item_rect.x + 8, item_rect.y + 5))

        elif self.active_dropdown == "push":
            menu_w = 165
            menu_h = len(self.push_options) * 30 + 8
            menu_rect = pygame.Rect(self.btn_push.x, self.btn_push.bottom + 4, menu_w, menu_h)

            # 앰비언트 섀도우
            shadow_s = pygame.Surface((menu_w + 8, menu_h + 8), pygame.SRCALPHA)
            shadow_s.fill((0, 0, 0, 140))
            self.screen.blit(shadow_s, (menu_rect.x - 4, menu_rect.y - 2))

            pygame.draw.rect(self.screen, COLOR_DROPDOWN_BG, menu_rect, border_radius=6)
            pygame.draw.rect(self.screen, COLOR_ROSE, menu_rect, width=1, border_radius=6)

            for i, (force, label) in enumerate(self.push_options):
                item_rect = pygame.Rect(self.btn_push.x + 4, self.btn_push.bottom + 6 + i * 30, menu_w - 8, 26)
                is_hover = item_rect.collidepoint(mouse_pos)

                if is_hover:
                    pygame.draw.rect(self.screen, COLOR_PANEL_HOVER, item_rect, border_radius=4)

                col = COLOR_ROSE if is_hover else COLOR_TEXT_PRIMARY
                txt_surf = self.font_sub.render(label, True, col)
                self.screen.blit(txt_surf, (item_rect.x + 8, item_rect.y + 5))

    def _draw_main_3d_panel(self, x, y, w, h):
        """메인 3D 뷰어 화면 (4자리수 에피소드 여백 최적화)"""
        panel_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, COLOR_PANEL, panel_rect, border_radius=8)

        if self.render_3d:
            sim_surf = self.sim.render_surface()
            self.screen.blit(sim_surf, (x, y))
        else:
            off_msg = self.font_sub.render("3D RENDERING SUSPENDED (PRESS T TO ENABLE)", True, COLOR_TEXT_MUTED)
            self.screen.blit(off_msg, (x + w // 2 - 140, y + h // 2))

        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, panel_rect, width=1, border_radius=8)

        # 상단 슬림 인포바 (EPISODE w=105px로 확장하여 장기 학습 완벽 지원)
        self._draw_glass_pill(x + 12, y + 12, 105, 30, "EPISODE", str(self.episode_count), COLOR_CYAN)
        self._draw_glass_pill(x + 125, y + 12, 105, 30, "ALIVE STEPS", str(self.sim.alive_steps), COLOR_EMERALD)
        self._draw_glass_pill(x + 238, y + 12, 135, 30, "NOISE SCALE", f"sigma = {self.trainer.exploration_noise:.2f}", COLOR_AMBER)

        # 앰비언트 충격 팝업
        if self.push_alert_timer > 0:
            alert_s = pygame.Surface((w - 30, 36), pygame.SRCALPHA)
            alert_s.fill((244, 63, 94, 210))
            pygame.draw.rect(alert_s, (255, 255, 255), (0, 0, w - 30, 36), width=1, border_radius=6)
            self.screen.blit(alert_s, (x + 15, y + h - 50))

            alert_txt = self.font_alert.render(self.push_alert_text, True, COLOR_TEXT_PRIMARY)
            txt_rect = alert_txt.get_rect(center=(x + w // 2, y + h - 32))
            self.screen.blit(alert_txt, txt_rect)

        # 모델 저장 팝업
        if self.save_alert_timer > 0:
            alert_s = pygame.Surface((240, 32), pygame.SRCALPHA)
            alert_s.fill((16, 185, 129, 220))
            pygame.draw.rect(alert_s, (255, 255, 255), (0, 0, 240, 32), width=1, border_radius=6)
            self.screen.blit(alert_s, (x + w - 255, y + 12))

            save_txt = self.font_sub.render("[ MODEL CHECKPOINT SAVED ]", True, COLOR_TEXT_PRIMARY)
            txt_rect = save_txt.get_rect(center=(x + w - 135, y + 28))
            self.screen.blit(save_txt, txt_rect)

    def _draw_glass_pill(self, x, y, w, h, label, val, color):
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        s.fill((10, 14, 23, 220))
        pygame.draw.rect(s, COLOR_PANEL_BORDER, (0, 0, w, h), width=1, border_radius=4)
        self.screen.blit(s, (x, y))

        lbl_s = self.font_small.render(label, True, COLOR_TEXT_MUTED)
        val_s = self.font_sub.render(val, True, color)
        self.screen.blit(lbl_s, (x + 8, y + 2))
        self.screen.blit(val_s, (x + 8, y + 14))

    def _draw_stat_cards(self, x, y, w, h):
        """우측 상단 4대 벤토 핵심 지표"""
        card_w = (w - 30) // 4
        card_h = h

        avg_20 = float(np.mean(list(self.recent_20_rewards))) if len(self.recent_20_rewards) > 0 else 0.0

        cards_data = [
            ("EPISODE REWARD", f"{self.sim.episode_reward:.1f}", COLOR_CYAN),
            ("PEAK REWARD", f"{self.best_reward:.1f}", COLOR_AMBER),
            ("20-EP MOVING AVG", f"{avg_20:.1f}", COLOR_EMERALD),
            ("TOTAL TIMESTEPS", f"{self.trainer.total_timesteps:,}", COLOR_VIOLET),
        ]

        for i, (lbl, val, col) in enumerate(cards_data):
            cx = x + i * (card_w + 10)
            rect = pygame.Rect(cx, y, card_w, card_h)
            pygame.draw.rect(self.screen, COLOR_PANEL, rect, border_radius=8)
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, rect, width=1, border_radius=8)

            lbl_surf = self.font_card_lbl.render(lbl, True, COLOR_TEXT_MUTED)
            val_surf = self.font_card_val.render(val, True, col)
            self.screen.blit(lbl_surf, (cx + 14, y + 14))
            self.screen.blit(val_surf, (cx + 14, y + 42))

    def _draw_learning_diagnostics(self, x, y, w, h):
        """PPO 학습 진단 카드"""
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, COLOR_PANEL, rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, rect, width=1, border_radius=8)

        t_surf = self.font_card_lbl.render("LEARNING DIAGNOSTICS // PPO TELEMETRY", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(t_surf, (x + 14, y + 12))

        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (x + 14, y + 30), (x + w - 14, y + 30), 1)

        p_loss_str = f"Policy Gradient Loss : {self.trainer.policy_loss:+.4f}"
        v_loss_str = f"Value Function Loss  : {self.trainer.value_loss:.4f}"
        ent_str    = f"Entropy (Noise)      : {self.trainer.entropy:.3f}   |   Iterations: #{self.trainer.update_count}"

        self.screen.blit(self.font_mono.render(p_loss_str, True, COLOR_ROSE), (x + 18, y + 42))
        self.screen.blit(self.font_mono.render(v_loss_str, True, COLOR_AMBER), (x + 18, y + 66))
        self.screen.blit(self.font_mono.render(ent_str, True, COLOR_EMERALD), (x + 18, y + 90))

    def _draw_actuator_torques(self, x, y, w, h):
        """우측 하단 17자유도 관절 토크 바"""
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, COLOR_PANEL, rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, rect, width=1, border_radius=8)

        t_surf = self.font_card_lbl.render("ACTUATOR TORQUES // 17 DOF CONTINUOUS CONTROL", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(t_surf, (x + 14, y + 12))

        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (x + 14, y + 30), (x + w - 14, y + 30), 1)

        col_w = (w - 40) // 2
        col1_x = x + 15
        col2_x = x + col_w + 25

        row_h = 36
        start_y = y + 40

        for i, name in enumerate(SimulationEngine.JOINT_NAMES):
            val = self.sim.current_torques[i] if i < len(self.sim.current_torques) else 0.0
            
            if i < 9:
                bx = col1_x
                by = start_y + i * row_h
            else:
                bx = col2_x
                by = start_y + (i - 9) * row_h

            name_surf = self.font_mono.render(name, True, COLOR_TEXT_SECONDARY)
            self.screen.blit(name_surf, (bx, by))

            bar_x = bx + 100
            bar_w = col_w - 110
            bar_h = 10
            bar_rect = pygame.Rect(bar_x, by + 3, bar_w, bar_h)
            pygame.draw.rect(self.screen, COLOR_PANEL_SUB, bar_rect, border_radius=3)

            mid_x = bar_x + bar_w // 2
            fill_w = int((abs(val)) * (bar_w // 2))
            bar_col = COLOR_MAGENTA if val < 0 else COLOR_CYAN

            if val < 0:
                fill_rect = pygame.Rect(mid_x - fill_w, by + 3, fill_w, bar_h)
            else:
                fill_rect = pygame.Rect(mid_x, by + 3, fill_w, bar_h)

            pygame.draw.rect(self.screen, bar_col, fill_rect, border_radius=2)
            pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (mid_x, by + 1), (mid_x, by + bar_h + 4), 1)

    def _draw_reward_curve(self, x, y, w, h):
        """좌측 하단 보상 텔레메트리 곡선"""
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, COLOR_PANEL, rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, rect, width=1, border_radius=8)

        t_surf = self.font_card_lbl.render("REWARD TELEMETRY CURVE", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(t_surf, (x + 14, y + 12))

        plot_x = x + 14
        plot_y = y + 36
        plot_w = w - 28
        plot_h = h - 48

        pygame.draw.rect(self.screen, COLOR_PANEL_SUB, (plot_x, plot_y, plot_w, plot_h), border_radius=4)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (plot_x, plot_y + plot_h // 2), (plot_x + plot_w, plot_y + plot_h // 2), 1)

        if len(self.recent_rewards) >= 2:
            all_r = list(self.recent_rewards)
            min_r = min(min(all_r), 0.0)
            max_r = max(max(all_r), min_r + 50.0)

            pad_y = 10
            usable_h = plot_h - (pad_y * 2)

            points = []
            for i, r in enumerate(all_r):
                px = plot_x + int(i * (plot_w / max(len(all_r) - 1, 1)))
                norm_y = float(np.clip((r - min_r) / (max_r - min_r + 1e-5), 0.0, 1.0))
                py = plot_y + plot_h - pad_y - int(norm_y * usable_h)
                points.append((px, py))

            pygame.draw.lines(self.screen, COLOR_CYAN, False, points, width=2)
            pygame.draw.circle(self.screen, COLOR_AMBER, points[-1], 4)

            max_lbl = self.font_mono.render(f"{int(max_r)}", True, COLOR_TEXT_MUTED)
            min_lbl = self.font_mono.render(f"{int(min_r)}", True, COLOR_TEXT_MUTED)
            self.screen.blit(max_lbl, (plot_x + plot_w - 32, plot_y + 4))
            self.screen.blit(min_lbl, (plot_x + plot_w - 32, plot_y + plot_h - 16))

    def _draw_console_logs(self, x, y, w, h):
        """중앙 하단 시스템 콘솔 로그"""
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, COLOR_PANEL, rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, rect, width=1, border_radius=8)

        t_surf = self.font_card_lbl.render("SYSTEM CONSOLE LOG", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(t_surf, (x + 14, y + 12))

        log_box = pygame.Rect(x + 10, y + 34, w - 20, h - 44)
        pygame.draw.rect(self.screen, COLOR_PANEL_SUB, log_box, border_radius=4)

        for i, log in enumerate(self.console_logs):
            is_latest = (i == len(self.console_logs) - 1)
            col = COLOR_CYAN if is_latest else COLOR_TEXT_MUTED
            txt_surf = self.font_mono.render(log, True, col)
            self.screen.blit(txt_surf, (x + 16, y + 42 + i * 36))

    def _draw_footer(self):
        """하단 단축키 가이드 바"""
        footer_y = self.height - 36
        guide_text = "[Space: Pause/Resume]  [T: Toggle 3D]  [F: Speed]  [P: Push]  [S: Save]  [L: Load]  [R/E: Reset]"
        guide_surf = self.font_mono.render(guide_text, True, COLOR_TEXT_MUTED)
        self.screen.blit(guide_surf, (20, footer_y))


if __name__ == "__main__":
    app = RobotAILabGUI(width=1280, height=820)
    app.run()
