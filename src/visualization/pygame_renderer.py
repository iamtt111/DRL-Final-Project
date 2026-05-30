import pygame
import sys
import numpy as np
from src.envs.building import Building
from src.envs.elevator import Elevator, ElevatorState
from src.envs.passenger import PassengerState
from typing import Tuple

# 色彩調色盤 (Curated Sleek Dark Palette)
COLOR_BG = (18, 18, 24)
COLOR_PANEL_BG = (30, 30, 40)
COLOR_TEXT = (240, 240, 245)
COLOR_TEXT_MUTED = (160, 160, 175)
COLOR_LINE = (50, 50, 70)
COLOR_SHAFT = (28, 28, 36)

# 電梯載客率漸變色 (綠 -> 黃 -> 紅)
COLOR_ELEV_EMPTY = (74, 222, 128)
COLOR_ELEV_HALF = (250, 204, 21)
COLOR_ELEV_FULL = (239, 68, 68)

# 優先權角色色彩
COLOR_PRIORITY_WHEELCHAIR = (59, 130, 246)  # 藍色 (輪椅)
COLOR_PRIORITY_STAFF = (245, 158, 11)       # 黃橘色 (醫護)
COLOR_PRIORITY_EMERGENCY = (239, 68, 68)     # 紅色 (急診)

class PygameRenderer:
    """智慧醫院電梯群控系統 Pygame 即時渲染引擎"""

    def __init__(self, building: Building, max_time: float = 600.0):
        pygame.init()
        self.width = 1000
        self.height = 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Smart Hospital EGCS - Based on DRL")
        
        self.clock = pygame.time.Clock()
        self.max_time = max_time
        
        # 字型設定
        try:
            self.font_title = pygame.font.SysFont("Outfit", 20, bold=True)
            self.font_normal = pygame.font.SysFont("Inter", 14)
            self.font_small = pygame.font.SysFont("Inter", 12)
        except:
            self.font_title = pygame.font.Font(None, 24)
            self.font_normal = pygame.font.Font(None, 18)
            self.font_small = pygame.font.Font(None, 14)

        # 佈局定位
        self.building_left = 50
        self.building_width = 500
        self.building_bottom = 650
        self.building_height = 580
        
        self.panel_left = 600
        self.panel_width = 350

    def render(self, building: Building) -> None:
        """渲染模擬幀"""
        # 處理 Pygame 視窗事件，防止當機
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        self.screen.fill(COLOR_BG)

        # 1. 繪製左方大樓剖面與電梯井
        self._draw_building(building)
        
        # 2. 繪製電梯轎廂
        self._draw_elevators(building)

        # 3. 繪製右方數據監控面板
        self._draw_dashboard(building)

        pygame.display.flip()
        self.clock.tick(30)  # 限制在 30 FPS 播放

    def _draw_building(self, building: Building) -> None:
        num_floors = building.num_floors
        max_height = building.max_height
        
        # 計算每公尺對應的像素高度
        scale_y = self.building_height / max_height if max_height > 0 else 1.0

        # 繪製樓層地板線
        for f in range(num_floors):
            floor_h = building.floor_heights[f]
            y = self.building_bottom - (floor_h * scale_y)
            
            # 地板線
            pygame.draw.line(self.screen, COLOR_LINE, (self.building_left, y), (self.building_left + self.building_width, y), 1)
            
            # 樓層文字
            lbl = f"F{f+1}" if f > 0 else "Lobby"
            txt_surf = self.font_normal.render(lbl, True, COLOR_TEXT_MUTED)
            self.screen.blit(txt_surf, (self.building_left - 35, y - 8))

            # 繪製在該樓層排隊等待的乘客
            self._draw_waiting_passengers(building, f, y)

    def _draw_waiting_passengers(self, building: Building, floor_idx: int, floor_y: float) -> None:
        floor = building.floors[floor_idx]
        waiting_passengers = [p for p in floor.waiting_queue if p.state.value == "waiting"]
        
        # 計算乘客繪製起點
        start_x = self.building_left + 10
        offset_x = 12

        for i, p in enumerate(waiting_passengers[:25]):  # 最多在畫面上顯示 25 個方向指示
            color = COLOR_TEXT
            if p.priority_level == 1:
                color = COLOR_PRIORITY_WHEELCHAIR
            elif p.priority_level == 2:
                color = COLOR_PRIORITY_STAFF
            elif p.priority_level == 3:
                color = COLOR_PRIORITY_EMERGENCY

            px = int(start_x + i * offset_x)
            py = int(floor_y - 6)

            # 如果是急診患者，額外繪製一個發光的紅色脈動圈以示警告
            if p.priority_level == 3:
                glow_radius = int(8 + 4 * np.sin(pygame.time.get_ticks() / 100.0))
                pygame.draw.circle(self.screen, COLOR_PRIORITY_EMERGENCY, (px, py), glow_radius, 1)

            # 繪製乘客的目的地樓層數字 (例如 "L" 代表 Lobby，"2" 代表 2 樓)
            dest_lbl = f"{p.destination_floor + 1}" if p.destination_floor > 0 else "L"
            dest_surf = self.font_small.render(dest_lbl, True, color)
            self.screen.blit(dest_surf, (px - dest_surf.get_width() / 2, py - 14))

            if p.direction == 1:
                # 往上三角形 (代表上行乘客)
                pygame.draw.polygon(self.screen, color, [(px, py - 4), (px - 4, py + 3), (px + 4, py + 3)])
            else:
                # 往下三角形 (代表下行乘客)
                pygame.draw.polygon(self.screen, color, [(px, py + 4), (px - 4, py - 3), (px + 4, py - 3)])

    def _draw_elevators(self, building: Building) -> None:
        num_floors = building.num_floors
        max_height = building.max_height
        scale_y = self.building_height / max_height if max_height > 0 else 1.0

        n_e = len(building.elevators)
        if n_e == 0:
            return

        # 分配電梯井的 X 軸寬度
        shaft_area_width = 300
        shaft_start_x = self.building_left + 150
        shaft_width = 40
        spacing = (shaft_area_width - (n_e * shaft_width)) / (n_e - 1) if n_e > 1 else 0.0

        for i, elev in enumerate(building.elevators):
            x = shaft_start_x + i * (shaft_width + spacing)
            
            # 繪製電梯井背景
            pygame.draw.rect(self.screen, COLOR_SHAFT, (x, self.building_bottom - self.building_height, shaft_width, self.building_height))
            pygame.draw.rect(self.screen, COLOR_LINE, (x, self.building_bottom - self.building_height, shaft_width, self.building_height), 1)

            # 計算電梯轎廂 Y 座標 (以其 current_position 為準)
            elev_y = self.building_bottom - (elev.current_position * scale_y)
            elev_height = 24

            # 決定轎廂的顏色漸變 (根據載重量)
            ratio = elev.load_ratio
            if ratio < 0.5:
                color = self._interpolate_color(COLOR_ELEV_EMPTY, COLOR_ELEV_HALF, ratio * 2.0)
            else:
                color = self._interpolate_color(COLOR_ELEV_HALF, COLOR_ELEV_FULL, (ratio - 0.5) * 2.0)

            # 繪製轎廂矩形
            pygame.draw.rect(self.screen, color, (x + 2, elev_y - elev_height, shaft_width - 4, elev_height))
            
            # 若門打開，繪製白邊框線
            if elev.state == ElevatorState.DOOR_OPEN:
                pygame.draw.rect(self.screen, (255, 255, 255), (x + 2, elev_y - elev_height, shaft_width - 4, elev_height), 2)
            else:
                pygame.draw.rect(self.screen, (0, 0, 0), (x + 2, elev_y - elev_height, shaft_width - 4, elev_height), 1)

            # 繪製車廂內人數文字
            load_txt = f"{elev.current_load}"
            txt_surf = self.font_small.render(load_txt, True, (0, 0, 0) if ratio < 0.8 else COLOR_TEXT)
            self.screen.blit(txt_surf, (x + (shaft_width - txt_surf.get_width()) / 2, elev_y - elev_height + 5))

            # 繪製方向箭頭或緊急警示
            if elev.emergency_target is not None:
                # 1. 繪製閃爍的紅色急診任務警示
                if int(pygame.time.get_ticks() / 250) % 2 == 0:
                    pygame.draw.circle(self.screen, COLOR_PRIORITY_EMERGENCY, (int(x + shaft_width / 2), int(elev_y - elev_height - 6)), 5)
                
                # 2. 繪製脈動的紅色虛線直達緊急目標樓層
                target_floor_h = building.floor_heights[elev.emergency_target]
                target_y_pixel = self.building_bottom - (target_floor_h * scale_y)
                glow = int(100 + 155 * abs(np.sin(pygame.time.get_ticks() / 150.0)))
                glow_color = (glow, 20, 20)
                self._draw_vertical_dashed_line(int(x + shaft_width / 2), int(elev_y - elev_height), int(target_y_pixel), glow_color, width=3)
                
            elif elev.current_direction == 1:
                # 上行三角形
                pygame.draw.polygon(self.screen, COLOR_TEXT, [(x + 15, elev_y - elev_height - 8), (x + 25, elev_y - elev_height - 8), (x + 20, elev_y - elev_height - 13)])
            elif elev.current_direction == -1:
                # 下行三角形
                pygame.draw.polygon(self.screen, COLOR_TEXT, [(x + 15, elev_y - elev_height - 13), (x + 25, elev_y - elev_height - 13), (x + 20, elev_y - elev_height - 8)])

    def _draw_dashboard(self, building: Building) -> None:
        # 面板底框
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, (self.panel_left, 50, self.panel_width, self.building_height))
        pygame.draw.rect(self.screen, COLOR_LINE, (self.panel_left, 50, self.panel_width, self.building_height), 1)

        # 提前計算指標以決定是否顯示警告橫幅
        waiting_passengers = []
        for floor in building.floors:
            waiting_passengers.extend([p for p in floor.waiting_queue if p.state.value == "waiting"])

        tot_wait = 0
        priority_wait = 0
        emergency_count = 0
        
        for p in waiting_passengers:
            tot_wait += p.get_wait_duration(building.current_time)
            if p.priority_level > 0:
                priority_wait += p.get_wait_duration(building.current_time)
            if p.priority_level == 3:
                emergency_count += 1

        awt = tot_wait / len(waiting_passengers) if waiting_passengers else 0.0
        pwt = priority_wait / sum(1 for p in waiting_passengers if p.priority_level > 0) if any(p.priority_level > 0 for p in waiting_passengers) else 0.0

        # 決定佈局偏移與繪製緊急警示橫幅
        if emergency_count > 0:
            # 閃爍的紅色警告橫幅
            bg_glow = int(80 + 60 * np.sin(pygame.time.get_ticks() / 100.0))
            pygame.draw.rect(self.screen, (bg_glow, 0, 0), (self.panel_left + 15, 65, self.panel_width - 30, 45), border_radius=4)
            pygame.draw.rect(self.screen, (239, 68, 68), (self.panel_left + 15, 65, self.panel_width - 30, 45), 2, border_radius=4)
            
            try:
                warn_font = pygame.font.SysFont("Outfit", 14, bold=True)
            except:
                warn_font = pygame.font.Font(None, 16)
            warn_surf = warn_font.render("EMERGENCY PREEMPTION ACTIVE", True, (255, 255, 255))
            self.screen.blit(warn_surf, (self.panel_left + (self.panel_width - warn_surf.get_width()) / 2, 80))
            
            title_y = 125
            time_y = 155
            metrics_start_y = 185
        else:
            title_y = 70
            time_y = 110
            metrics_start_y = 150

        # 標題
        title_surf = self.font_title.render("EGCS MONITOR PANEL", True, COLOR_TEXT)
        self.screen.blit(title_surf, (self.panel_left + 20, title_y))

        # 時間顯示
        time_surf = self.font_normal.render(f"Sim Time: {building.current_time:.1f}s / {self.max_time}s", True, COLOR_TEXT)
        self.screen.blit(time_surf, (self.panel_left + 20, time_y))

        metrics = [
            f"Active Waiting: {len(waiting_passengers)} persons",
            f"Average Wait (AWT): {awt:.1f} s",
            f"Priority Wait (PWT): {pwt:.1f} s",
            f"Active Emergencies: {emergency_count} 🔴"
        ]

        for i, m in enumerate(metrics):
            m_surf = self.font_normal.render(m, True, COLOR_TEXT)
            self.screen.blit(m_surf, (self.panel_left + 20, metrics_start_y + i * 25))

        # 繪製負載均衡度 (Load Balancing Bar)
        pygame.draw.line(self.screen, COLOR_LINE, (self.panel_left + 20, 280), (self.panel_left + self.panel_width - 20, 280), 1)
        load_lbl = self.font_title.render("ELEVATOR LOAD STATUS", True, COLOR_TEXT)
        self.screen.blit(load_lbl, (self.panel_left + 20, 300))

        for i, elev in enumerate(building.elevators):
            y_offset = 340 + i * 40
            elev_label = self.font_normal.render(f"E{i+1}:", True, COLOR_TEXT_MUTED)
            self.screen.blit(elev_label, (self.panel_left + 20, y_offset))

            # 負載條
            bar_x = self.panel_left + 60
            bar_width = 240
            bar_height = 14
            
            # 背景條
            pygame.draw.rect(self.screen, COLOR_SHAFT, (bar_x, y_offset + 2, bar_width, bar_height))
            
            # 當前載客率填充
            ratio = elev.load_ratio
            fill_width = int(bar_width * ratio)
            
            if ratio < 0.5:
                color = COLOR_ELEV_EMPTY
            elif ratio < 0.8:
                color = COLOR_ELEV_HALF
            else:
                color = COLOR_ELEV_FULL

            if fill_width > 0:
                pygame.draw.rect(self.screen, color, (bar_x, y_offset + 2, fill_width, bar_height))
            
            pygame.draw.rect(self.screen, COLOR_LINE, (bar_x, y_offset + 2, bar_width, bar_height), 1)

            # 文字百分比
            pct_surf = self.font_small.render(f"{elev.current_load}/{elev.max_capacity}", True, COLOR_TEXT)
            self.screen.blit(pct_surf, (bar_x + bar_width + 10, y_offset))

        # 優先權角色註解說明
        legend_y = 520
        legend_lbl = self.font_title.render("ROLE LEGEND", True, COLOR_TEXT)
        self.screen.blit(legend_lbl, (self.panel_left + 20, legend_y))

        legends = [
            (COLOR_PRIORITY_EMERGENCY, "Level 3 - Emergency Bed (急診病床)"),
            (COLOR_PRIORITY_STAFF, "Level 2 - Medical Staff (醫護人員)"),
            (COLOR_PRIORITY_WHEELCHAIR, "Level 1 - Wheelchair Users (輪椅族)")
        ]

        for i, (col, text) in enumerate(legends):
            pygame.draw.circle(self.screen, col, (self.panel_left + 30, legend_y + 35 + i * 25), 5)
            leg_surf = self.font_normal.render(text, True, COLOR_TEXT_MUTED)
            self.screen.blit(leg_surf, (self.panel_left + 45, legend_y + 26 + i * 25))

    def _draw_vertical_dashed_line(self, x: int, y1: int, y2: int, color: Tuple[int, int, int], width: int = 2, dash_length: int = 8, space: int = 6):
        """繪製垂直虛線"""
        start_y = min(y1, y2)
        end_y = max(y1, y2)
        y = start_y
        while y < end_y:
            pygame.draw.line(self.screen, color, (x, y), (x, min(y + dash_length, end_y)), width)
            y += dash_length + space

    def _interpolate_color(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        """線性插值計算漸變色"""
        f = max(0.0, min(1.0, factor))
        return (
            int(color1[0] + (color2[0] - color1[0]) * f),
            int(color1[1] + (color2[1] - color1[1]) * f),
            int(color1[2] + (color2[2] - color1[2]) * f)
        )

    def close(self) -> None:
        pygame.quit()
