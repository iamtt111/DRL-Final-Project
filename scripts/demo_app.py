import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import tkinter as tk
import customtkinter as ctk
import numpy as np
import torch
import yaml

from src.envs.elevator_env import HospitalElevatorEnv
from src.envs.elevator_ma_env import HospitalElevatorMAEnv
from src.envs.elevator import ElevatorState
from src.envs.passenger import PassengerState

# 載入所有代理人
from src.agents.ppo_agent import PPOAgent
from src.agents.sarsa_agent import SarsaAgent
from src.agents.rule_based import NearestCarAgent
from src.agents.mappo_agent import MAPPOAgent

# -------------------------------------------------------------
# 輔助函數：高度轉換為 1-16 樓的實數樓層 (Float)
# -------------------------------------------------------------
def get_floor_float(height, heights):
    if height <= 0.0:
        return 1.0
    if height >= heights[-1]:
        return float(len(heights))
    for i in range(len(heights) - 1):
        h_low = heights[i]
        h_high = heights[i+1]
        if h_low <= height <= h_high:
            ratio = (height - h_low) / (h_high - h_low)
            return (i + ratio) + 1.0
    return 1.0

class ElevatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 視窗基本設定
        self.title("🏥 Smart Hospital Elevator Group Control Dashboard")
        self.geometry("1150x700")
        self.resizable(True, True)

        # 預設主題
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # 模擬狀態變數
        self.running = False
        self.env = None
        self.agent = None
        self.passengers_delivered = []
        self.elevator_starts = []
        self.logs_list = []
        self.loop_timer_id = None

        # 建立 UI 佈局 (Grid)
        self.grid_columnconfigure(0, weight=0, minsize=280) # 側邊欄
        self.grid_columnconfigure(1, weight=1)              # 主區域
        self.grid_rowconfigure(0, weight=1)

        # 1. 建立左側控制面板 (Sidebar)
        self.sidebar_frame = ctk.CTkFrame(self, corner_radius=0, width=280)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar_frame.grid_rowconfigure(9, weight=1)

        # 側邊欄標題
        self.title_label = ctk.CTkLabel(self.sidebar_frame, text="🏥 Hospital EGCS", font=ctk.CTkFont(size=22, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 5))
        self.subtitle_label = ctk.CTkLabel(self.sidebar_frame, text="智慧電梯群控展示系統", font=ctk.CTkFont(size=13))
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        # 演算法選擇
        self.agent_label = ctk.CTkLabel(self.sidebar_frame, text="🤖 選擇調度演算法 (Agent)", anchor="w")
        self.agent_label.grid(row=2, column=0, padx=20, pady=(10, 5), sticky="w")
        self.agent_dropdown = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["MAPPO (嵌入+冷啟動版)", "MaskablePPO", "SARSA(λ)", "Nearest Car (規則式)"]
        )
        self.agent_dropdown.grid(row=3, column=0, padx=20, pady=5, sticky="ew")

        # 場景選擇
        self.scenario_label = ctk.CTkLabel(self.sidebar_frame, text="📊 選擇交通流量情境", anchor="w")
        self.scenario_label.grid(row=4, column=0, padx=20, pady=(15, 5), sticky="w")
        self.scenario_dropdown = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["morning_peak", "evening_peak", "mixed_traffic", "disaster_crisis"]
        )
        self.scenario_dropdown.grid(row=5, column=0, padx=20, pady=5, sticky="ew")

        # 動畫延遲速度
        self.speed_label = ctk.CTkLabel(self.sidebar_frame, text="⏱️ 模擬動畫速度 (間隔時間)", anchor="w")
        self.speed_label.grid(row=6, column=0, padx=20, pady=(15, 5), sticky="w")
        self.speed_slider = ctk.CTkSlider(self.sidebar_frame, from_=0.01, to=0.50, number_of_steps=49)
        self.speed_slider.set(0.05)
        self.speed_slider.grid(row=7, column=0, padx=20, pady=5, sticky="ew")
        
        # 按鈕
        self.start_btn = ctk.CTkButton(self.sidebar_frame, text="▶️ 開始模擬", command=self.start_simulation, fg_color="#2ecc71", hover_color="#27ae60", text_color="white")
        self.start_btn.grid(row=8, column=0, padx=20, pady=(30, 10), sticky="ew")
        
        self.stop_btn = ctk.CTkButton(self.sidebar_frame, text="⏹️ 停止", command=self.stop_simulation, fg_color="#e74c3c", hover_color="#c0392b", text_color="white")
        self.stop_btn.grid(row=9, column=0, padx=20, pady=10, sticky="ew")
        self.stop_btn.configure(state="disabled")

        # 2. 建立右側主面板 (Main Panel)
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=0) # KPI
        self.main_frame.grid_rowconfigure(1, weight=1) # 視覺化與日誌

        # 2a. KPI 區塊
        self.kpi_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.kpi_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        for i in range(5):
            self.kpi_frame.grid_columnconfigure(i, weight=1, uniform="kpi")

        self.kpi_cards = []
        kpi_titles = ["⏳ 模擬時間", "👥 AWT (全體等候)", "🚑 ERT (急診等候)", "🎯 ECR (急診完成率)", "⚡ NSS (總起停次數)"]
        kpi_colors = ["#2c3e50", "#2980b9", "#c0392b", "#d35400", "#27ae60"]

        for i, (title, color) in enumerate(zip(kpi_titles, kpi_colors)):
            card = ctk.CTkFrame(self.kpi_frame, border_width=1.5, border_color=color, height=85)
            card.grid(row=0, column=i, padx=5, sticky="ew")
            card.grid_propagate(False)
            
            title_lbl = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color="#a0a0a0")
            title_lbl.pack(anchor="w", padx=10, pady=(5, 0))
            
            val_lbl = ctk.CTkLabel(card, text="--", font=ctk.CTkFont(size=20, weight="bold"), text_color=color)
            val_lbl.pack(anchor="w", padx=10, pady=(2, 5))
            
            self.kpi_cards.append(val_lbl)

        # 2b. 下半部：大樓視覺化與終端日誌
        self.display_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.display_frame.grid(row=1, column=0, sticky="nsew")
        self.display_frame.grid_columnconfigure(0, weight=3, minsize=520) # 畫布
        self.display_frame.grid_columnconfigure(1, weight=2, minsize=320) # 日誌
        self.display_frame.grid_rowconfigure(0, weight=1)

        # 左側 Canvas
        self.canvas_container = ctk.CTkFrame(self.display_frame, border_width=1.5, border_color="#3a3a3a")
        self.canvas_container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.canvas_container.grid_rowconfigure(0, weight=0)
        self.canvas_container.grid_rowconfigure(1, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)
        
        self.canvas_title = ctk.CTkLabel(self.canvas_container, text="🏢 大樓運行實時動態監控 (1-16F)", font=ctk.CTkFont(size=14, weight="bold"))
        self.canvas_title.grid(row=0, column=0, pady=8)

        self.canvas = tk.Canvas(self.canvas_container, bg="#1e1e1e", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # 右側 Log
        self.log_container = ctk.CTkFrame(self.display_frame, border_width=1.5, border_color="#3a3a3a")
        self.log_container.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.log_container.grid_rowconfigure(0, weight=0)
        self.log_container.grid_rowconfigure(1, weight=1)
        self.log_container.grid_columnconfigure(0, weight=1)

        self.log_title = ctk.CTkLabel(self.log_container, text="📋 實時調度日誌看板", font=ctk.CTkFont(size=14, weight="bold"))
        self.log_title.grid(row=0, column=0, pady=8)

        self.log_textbox = ctk.CTkTextbox(self.log_container, font=("Courier New", 12), fg_color="#121212")
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.log_textbox.configure(state="disabled")

        # 初始化日誌 Tag 樣式
        self.log_textbox.tag_config("normal", foreground="#3498db")          # 寶藍色 (一般調度)
        self.log_textbox.tag_config("success", foreground="#2ecc71")         # 綠色 (完成送達)
        self.log_textbox.tag_config("emergency", foreground="#e74c3c")        # 亮紅色 (急診搶佔)
        self.log_textbox.tag_config("info", foreground="#a0a0a0")            # 灰色 (系統訊息)

        # 第一次靜態繪製 Canvas 背景
        self.bind("<Configure>", lambda e: self.draw_static_building_background())

    def draw_static_building_background(self):
        """繪製靜態大樓背景，供未啟動時顯示"""
        if self.running:
            return
        
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            w, h = 500, 600

        margin_left = 60
        margin_right = 160
        margin_top = 40
        margin_bottom = 40
        floor_height = (h - margin_top - margin_bottom) / 16.0

        # 樓層線與標籤
        for f in range(1, 17):
            y = margin_top + (16 - f) * floor_height
            self.canvas.create_line(margin_left, y, w - margin_right, y, fill="#2b2b2b", width=1)
            self.canvas.create_text(margin_left - 25, y, text=f"{f}F", fill="#555555", font=("Arial", 10, "bold"))

        # 電梯井
        for i in range(4):
            x_center = margin_left + 40 + i * 70
            self.canvas.create_rectangle(x_center - 15, margin_top, x_center + 15, h - margin_bottom, fill="#181818", outline="#2b2b2b")
            self.canvas.create_line(x_center, margin_top, x_center, h - margin_bottom, fill="#2b2b2b", width=1)
            self.canvas.create_text(x_center, h - margin_bottom + 18, text=f"E{i}", fill="#555555", font=("Arial", 10, "bold"))

    def add_log(self, text):
        """新增一條日誌並自動滾動 (支援按重要程度著色)"""
        self.log_textbox.configure(state="normal")
        
        # 根據日誌內容決定標籤顏色
        if "🚨" in text or "急診" in text or "強佔" in text or "搶佔" in text:
            tag = "emergency"
        elif "✅" in text or "送達" in text:
            tag = "success"
        elif "📢" in text or "🚀" in text or "⏹️" in text:
            tag = "info"
        else:
            tag = "normal"
            
        self.log_textbox.insert("end", text + "\n", tag)
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def clear_logs(self):
        """清空日誌"""
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

    def start_simulation(self):
        """按下開始按鈕，初始化環境並啟動模擬時鐘"""
        if self.running:
            return

        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.agent_dropdown.configure(state="disabled")
        self.scenario_dropdown.configure(state="disabled")

        self.clear_logs()
        self.add_log("📢 正在初始化環境配置...")

        # 讀取對應情境 YAML
        selected_scenario = self.scenario_dropdown.get()
        base_env = HospitalElevatorMAEnv()
        config = base_env.config
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        scenario_path = os.path.join(base_dir, "configs", "scenarios", f"{selected_scenario}.yaml")
        if os.path.exists(scenario_path):
            with open(scenario_path, "r", encoding="utf-8") as f:
                scenario_config = yaml.safe_load(f)
                config["traffic"] = scenario_config
            self.add_log(f"📊 已加載交通流情境：{selected_scenario}")
        else:
            self.add_log("⚠️ 找不到情境設定檔，使用環境預設交通流。")

        # 建立環境
        self.env = HospitalElevatorEnv(config=config)
        self.env.reset(seed=42)

        # 初始化 Agent
        selected_agent = self.agent_dropdown.get()
        self.add_log(f"🤖 正在加載 Agent：{selected_agent}...")

        if selected_agent == "MAPPO (嵌入+冷啟動版)":
            model_path = "models/mappo/best_model.pt"
            if not os.path.exists(model_path):
                model_path = "models/mappo/final_model.pt"
            self.agent = MAPPOAgent(model_path=model_path if os.path.exists(model_path) else None, env=self.env)
        elif selected_agent == "MaskablePPO":
            model_path = "models/ppo/best_model.zip"
            if not os.path.exists(model_path):
                model_path = "models/ppo/final_model.zip"
            self.agent = PPOAgent(model_path=model_path if os.path.exists(model_path) else None, env=self.env)
        elif selected_agent == "SARSA(λ)":
            s_agent = SarsaAgent(env=self.env)
            model_path = "models/sarsa/sarsa_weights.npz"
            if os.path.exists(model_path):
                s_agent.load(model_path)
            self.agent = s_agent
        else: # Nearest Car
            self.agent = NearestCarAgent(env=self.env)

        self.passengers_delivered = []
        self.elevator_starts = [0] * self.env.num_elevators

        # 預熱模擬機制：預先在背景快速運行 120 秒建立初始客流與電梯狀態
        warmup_time = 120
        self.add_log(f"⏳ 正在進行 {warmup_time} 秒的模擬快速預熱，建立初始交通流量與排隊...")
        self.update()  # 強制更新 UI 以顯示預熱日誌
        
        dt = self.env.dt
        steps = int(warmup_time / dt)
        for _ in range(steps):
            current_sim_time = self.env.building.current_time + dt
            new_passengers = self.env.traffic_gen.generate_arrivals(current_sim_time, dt)
            for p in new_passengers:
                if p.priority_level == 3:
                    self.env.priority_system.check_and_apply_preemption(self.env.building, p.origin_floor)
                self.env.building.add_passenger(p)
                
            if self.env.pending_assignments:
                # 依據所選演算法做決策
                if selected_agent == "MAPPO (嵌入+冷啟動版)":
                    action, _ = self.agent.predict(None)
                elif selected_agent == "MaskablePPO":
                    obs = self.env.building.get_state_vector()
                    action, _ = self.agent.predict(obs)
                elif selected_agent == "SARSA(λ)":
                    obs = self.env.building.get_state_vector()
                    action, _ = self.agent.predict(obs)
                else: # Nearest Car
                    action, _ = self.agent.predict(None)
                    
                call_pop = self.env.pending_assignments.pop(0)
                self.env.building.elevators[action].assign_hall_call(call_pop.floor, call_pop.direction)
                
            events = self.env.building.update(dt)
            self.env._update_pending_assignments()
            
            for event in events:
                if event.type.value == "passenger_delivered":
                    self.passengers_delivered.append({
                        "wait_time": event.data["wait_time"],
                        "priority": event.data["priority"]
                    })
                elif event.type.value == "elevator_arrived":
                    self.elevator_starts[event.data["elevator_id"]] += 1

        self.add_log("🚀 預熱完成！已建立繁忙的大樓客流。開始實時調度！")

        # 開始執行時鐘
        self.simulation_step()

    def stop_simulation(self):
        """點擊停止，停止時鐘"""
        self.running = False
        if self.loop_timer_id is not None:
            self.after_cancel(self.loop_timer_id)
            self.loop_timer_id = None
        
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.agent_dropdown.configure(state="normal")
        self.scenario_dropdown.configure(state="normal")
        self.add_log("⏹️ 模擬已被使用者強制停止。")

    def simulation_step(self):
        """核心物理模擬與調度遞迴時鐘"""
        if not self.running:
            return

        env = self.env
        
        # (a) 檢查是否有待指派的外呼
        if env.pending_assignments:
            call = env.pending_assignments[0]
            selected_agent = self.agent_dropdown.get()

            # 呼叫 Agent 做決策 (Action Masks 已經封裝在各 Agent 內部)
            if selected_agent == "MAPPO (嵌入+冷啟動版)":
                action, _ = self.agent.predict(None)
            elif selected_agent == "MaskablePPO":
                obs = env.building.get_state_vector()
                action, _ = self.agent.predict(obs)
            elif selected_agent == "SARSA(λ)":
                obs = env.building.get_state_vector()
                action, _ = self.agent.predict(obs)
            else: # Nearest Car
                action, _ = self.agent.predict(None)

            # 執行指派
            call_pop = env.pending_assignments.pop(0)
            env.building.elevators[action].assign_hall_call(call_pop.floor, call_pop.direction)
            
            action_desc = f"⏱️ {int(env.building.current_time)}s: 指派 {call_pop.floor + 1}F {'▲' if call_pop.direction == 1 else '▼'} 呼叫 ➡️ 電梯 {action}"
            self.add_log(action_desc)

        # (b) 推進物理時間 1 秒
        current_sim_time = env.building.current_time + env.dt
        new_passengers = env.traffic_gen.generate_arrivals(current_sim_time, env.dt)
        
        for p in new_passengers:
            if p.priority_level == 3:
                env.priority_system.check_and_apply_preemption(env.building, p.origin_floor)
                self.add_log(f"🚨 偵測到 {p.origin_floor + 1}F 急診呼叫，啟動強佔！")
            env.building.add_passenger(p)
            
        events = env.building.update(env.dt)
        env._update_pending_assignments()

        # (c) 解析事件
        for event in events:
            if event.type.value == "passenger_delivered":
                p_id = event.data["passenger_id"]
                wait_t = event.data["wait_time"]
                p_level = event.data["priority"]
                elev_id = event.data["elevator_id"]
                
                self.passengers_delivered.append({
                    "wait_time": wait_t,
                    "priority": p_level
                })
                
                p_names = {0: "普通", 1: "輪椅", 2: "醫護", 3: "🚨急診"}
                log_p = f"✅ 送達 {p_names.get(p_level, '普通')} 乘客 (等候: {wait_t:.1f}s) ➡️ 電梯 {elev_id}"
                self.add_log(log_p)
                
            elif event.type.value == "elevator_arrived":
                elev_id = event.data["elevator_id"]
                self.elevator_starts[elev_id] += 1

        # 繪圖與更新 KPIs
        self.draw_building()
        self.update_kpis()

        # (d) 判斷 Episode 結束
        if env.building.current_time >= env.max_time:
            self.add_log("🎉 模擬已完成 10 分鐘 Episode 結束。")
            self.running = False
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.agent_dropdown.configure(state="normal")
            self.scenario_dropdown.configure(state="normal")
            return

        # 繼續下一個時間步
        delay_ms = int(self.speed_slider.get() * 1000)
        self.loop_timer_id = self.after(delay_ms, self.simulation_step)

    def draw_building(self):
        """實時在 Canvas 上繪製大樓動態"""
        self.canvas.delete("all")
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            w, h = 500, 600

        margin_left = 60
        margin_right = 160
        margin_top = 40
        margin_bottom = 40
        floor_height = (h - margin_top - margin_bottom) / 16.0
        heights = self.env.building.floor_heights

        # 1. 繪製樓層分隔線與標籤 (1F - 16F)
        for f in range(1, 17):
            y = margin_top + (16 - f) * floor_height
            self.canvas.create_line(margin_left, y, w - margin_right, y, fill="#2b2b2b", width=1)
            self.canvas.create_text(margin_left - 25, y, text=f"{f}F", fill="#a0a0a0", font=("Arial", 10, "bold"))

        # 2. 繪製大樓電梯軌道
        for i in range(4):
            x_center = margin_left + 40 + i * 70
            # 寬的井道背景軌道
            self.canvas.create_rectangle(x_center - 15, margin_top, x_center + 15, h - margin_bottom, fill="#1c1c1c", outline="#2b2b2b")
            # 細的中心導軌
            self.canvas.create_line(x_center, margin_top, x_center, h - margin_bottom, fill="#3a3a3a", width=1)
            # 電梯井底部編號
            self.canvas.create_text(x_center, h - margin_bottom + 18, text=f"E{i}", fill="#a0a0a0", font=("Arial", 10, "bold"))

        # 3. 繪製各樓層大廳等候人數與排隊乘客 (呈現四個優先級)
        for f_idx, floor in enumerate(self.env.building.floors):
            y = margin_top + (15 - f_idx) * floor_height
            waiting_passengers = [p for p in floor.waiting_queue if p.state == PassengerState.WAITING]
            
            # (a) 首先在最右側保留數字統計標籤，方便快速閱讀
            normal_cnt = sum(1 for p in waiting_passengers if p.priority_level < 3)
            emergency_cnt = sum(1 for p in waiting_passengers if p.priority_level == 3)
            x_text = w - margin_right + 15
            
            # (b) 在電梯右方到邊界之間，繪製與先前 pygame 一致的排隊乘客圖示 (60 FPS 平滑顯示)
            # 電梯3的中心為 margin_left + 40 + 3 * 70 = 270 (當 margin_left=60)
            start_x = margin_left + 40 + 3 * 70 + 40
            space_available = w - margin_right - start_x - 10
            max_passengers_to_show = 12
            offset_x = max(8, min(14, space_available / max_passengers_to_show if space_available > 0 else 10))

            for idx, p in enumerate(waiting_passengers[:max_passengers_to_show]):
                px = start_x + idx * offset_x
                py = y
                
                # 四種優先權角色顏色
                if p.priority_level == 1:
                    color = "#3867d6"  # Level 1 輪椅：亮藍色
                elif p.priority_level == 2:
                    color = "#fa8231"  # Level 2 醫護：橘黃色
                elif p.priority_level == 3:
                    color = "#eb3b5a"  # Level 3 急診：鮮紅色
                else:
                    color = "#a5b1c2"  # Level 0 普通：灰白色
                
                # 如果是急診乘客，繪製一個雙環脈動效果 (類似 pygame 的外發光圈)
                if p.priority_level == 3:
                    pulse_radius = 8 + int(3 * np.sin(time.time() * 8)) # 使用系統時間模擬脈動
                    self.canvas.create_oval(px - pulse_radius, py - pulse_radius, px + pulse_radius, py + pulse_radius, outline="#eb3b5a", width=1.5)
                
                # 繪製代表上行/下行方向的三角形
                if p.direction == 1:
                    # 往上三角形
                    self.canvas.create_polygon(px, py - 4, px - 4, py + 4, px + 4, py + 4, fill=color, outline="")
                else:
                    # 往下三角形
                    self.canvas.create_polygon(px, py + 4, px - 4, py - 4, px + 4, py - 4, fill=color, outline="")

                # 在三角形上方繪製其目的地樓層數字 (如 Lobby 或 2-16)
                dest_lbl = f"{p.destination_floor + 1}"
                self.canvas.create_text(px, py - 9, text=dest_lbl, fill=color, font=("Arial", 7, "bold"))

            # 繪製統計數字標籤
            if normal_cnt > 0 and emergency_cnt > 0:
                self.canvas.create_text(x_text, y, text=f"👥{normal_cnt} 🚨{emergency_cnt}", fill="#eb3b5a", font=("Arial", 10, "bold"), anchor="w")
            elif normal_cnt > 0:
                self.canvas.create_text(x_text, y, text=f"👥{normal_cnt}", fill="#3867d6", font=("Arial", 10), anchor="w")
            elif emergency_cnt > 0:
                self.canvas.create_text(x_text, y, text=f"🚨{emergency_cnt}", fill="#eb3b5a", font=("Arial", 10, "bold"), anchor="w")

        # 4. 繪製電梯實體
        for elev in self.env.building.elevators:
            e_id = elev.elevator_id
            x_center = margin_left + 40 + e_id * 70
            
            # 轉換為 1-16 樓的實數高度
            y_pos = get_floor_float(elev.current_position, heights)
            y_canvas = margin_top + (16.0 - y_pos) * floor_height

            # 狀態顏色
            if elev.emergency_target is not None:
                color = "#e74c3c" # 🚨急診搶佔：亮紅色
            elif elev.state == ElevatorState.DOOR_OPEN:
                color = "#2ecc71" # 開門：亮綠色
            elif elev.state == ElevatorState.IDLE:
                color = "#555555" # 閒置：暗灰色
            else:
                color = "#2980b9" # 運行中：寶藍色

            # 繪製電梯箱體
            self.canvas.create_rectangle(x_center - 18, y_canvas - 12, x_center + 18, y_canvas + 12, fill=color, outline="white", width=1.5)
            
            # 繪製載客數與方向箭頭
            dir_arrow = "▲" if elev.current_direction == 1 else "▼" if elev.current_direction == -1 else ""
            self.canvas.create_text(x_center, y_canvas, text=f"{elev.current_load}{dir_arrow}", fill="white", font=("Arial", 9, "bold"))

            # 如果正在執行急診搶佔任務，繪製目標樓層的警示紅線與終點標誌
            if elev.emergency_target is not None:
                target_y = margin_top + (15 - elev.emergency_target) * floor_height
                # 繪製虛線導軌
                self.canvas.create_line(x_center, y_canvas, x_center, target_y, fill="#e74c3c", width=1.5, dash=(4, 4))
                # 繪製目標警示燈
                self.canvas.create_text(x_center, target_y - 18, text="🚨", font=("Arial", 9))

    def update_kpis(self):
        """實時計算 KPI 並填入頂部卡片"""
        delivered = self.passengers_delivered
        
        # AWT
        awt = np.mean([p["wait_time"] for p in delivered]) if delivered else 0.0
        # ERT
        ert_list = [p["wait_time"] for p in delivered if p["priority"] == 3]
        ert = np.mean(ert_list) if ert_list else 0.0
        # ECR
        ecr = sum(1 for p in delivered if p["priority"] == 3 and p["wait_time"] <= 30.0) / len(ert_list) * 100.0 if ert_list else 100.0
        # NSS
        nss = sum(self.elevator_starts)

        # 更新 UI
        self.kpi_cards[0].configure(text=f"{int(self.env.building.current_time)} 秒")
        self.kpi_cards[1].configure(text=f"{awt:.2f} 秒")
        self.kpi_cards[2].configure(text=f"{ert:.2f} 秒")
        self.kpi_cards[3].configure(text=f"{ecr:.2f} %")
        self.kpi_cards[4].configure(text=f"{nss} 次")

if __name__ == "__main__":
    app = ElevatorApp()
    app.mainloop()
