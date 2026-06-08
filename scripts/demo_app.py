import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import tkinter as tk
import customtkinter as ctk
import numpy as np
import torch
import yaml
import json

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
        self.paused = False
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
        self.sidebar_frame.grid_rowconfigure(11, weight=1)

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
            values=["MAPPO (嵌入+冷啟動版)", "MaskablePPO", "SARSA(λ)", "Nearest Car (規則式)"],
            command=self.update_benchmark_kpis
        )
        self.agent_dropdown.grid(row=3, column=0, padx=20, pady=5, sticky="ew")

        # 場景選擇
        self.scenario_label = ctk.CTkLabel(self.sidebar_frame, text="📊 選擇交通流量情境", anchor="w")
        self.scenario_label.grid(row=4, column=0, padx=20, pady=(15, 5), sticky="w")
        self.scenario_dropdown = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["morning_peak", "evening_peak", "mixed_traffic", "disaster_crisis"],
            command=self.update_benchmark_kpis
        )
        self.scenario_dropdown.grid(row=5, column=0, padx=20, pady=5, sticky="ew")

        # 動畫延遲速度
        self.speed_label = ctk.CTkLabel(self.sidebar_frame, text="⏱️ 模擬動畫速度 (間隔時間)", anchor="w")
        self.speed_label.grid(row=6, column=0, padx=20, pady=(15, 5), sticky="w")
        self.speed_slider = ctk.CTkSlider(self.sidebar_frame, from_=0.01, to=0.50, number_of_steps=49)
        self.speed_slider.set(0.05)
        self.speed_slider.grid(row=7, column=0, padx=20, pady=5, sticky="ew")
        
        # 隨機客流控制區塊
        self.seed_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.seed_frame.grid(row=8, column=0, padx=20, pady=(15, 5), sticky="ew")
        self.seed_frame.grid_columnconfigure(0, weight=1)
        self.seed_frame.grid_columnconfigure(1, weight=0)
        self.seed_frame.grid_columnconfigure(2, weight=0)
        
        self.seed_entry_label = ctk.CTkLabel(self.seed_frame, text="🎲 Seed:", anchor="w")
        self.seed_entry_label.grid(row=0, column=0, sticky="w")
        
        self.seed_entry = ctk.CTkEntry(self.seed_frame, width=55)
        self.seed_entry.insert(0, "42")
        self.seed_entry.grid(row=0, column=1, padx=2, sticky="e")
        
        self.random_seed_btn = ctk.CTkButton(self.seed_frame, text="🎲 隨機", width=45, command=self.generate_random_seed_in_entry)
        self.random_seed_btn.grid(row=0, column=2, padx=2, sticky="e")
        
        # 按鈕
        self.start_btn = ctk.CTkButton(self.sidebar_frame, text="▶️ 開始模擬", command=self.start_simulation, fg_color="#2ecc71", hover_color="#27ae60", text_color="white")
        self.start_btn.grid(row=9, column=0, padx=20, pady=(20, 10), sticky="ew")
        
        self.pause_btn = ctk.CTkButton(self.sidebar_frame, text="⏸️ 暫停", command=self.pause_simulation, fg_color="#f39c12", hover_color="#d35400", text_color="white")
        self.pause_btn.grid(row=10, column=0, padx=20, pady=10, sticky="ew")
        self.pause_btn.configure(state="disabled")
        
        self.stop_btn = ctk.CTkButton(self.sidebar_frame, text="⏹️ 停止", command=self.stop_simulation, fg_color="#e74c3c", hover_color="#c0392b", text_color="white")
        self.stop_btn.grid(row=11, column=0, padx=20, pady=10, sticky="ew")
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
        self.bench_labels = []
        kpi_titles = ["⏳ 模擬時間", "👥 AWT (全體等候)", "🚑 ERT (急診等候)", "🎯 ECR (急診完成率)", "⚡ NSS (總起停次數)"]
        kpi_colors = ["#2c3e50", "#2980b9", "#c0392b", "#d35400", "#27ae60"]

        for i, (title, color) in enumerate(zip(kpi_titles, kpi_colors)):
            card = ctk.CTkFrame(self.kpi_frame, border_width=1.5, border_color=color, height=95)
            card.grid(row=0, column=i, padx=5, sticky="ew")
            card.grid_propagate(False)
            
            title_lbl = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color="#a0a0a0")
            title_lbl.pack(anchor="w", padx=10, pady=(5, 0))
            
            if i == 0:
                val_lbl = ctk.CTkLabel(card, text="0 秒", font=ctk.CTkFont(size=18, weight="bold"), text_color=color)
                val_lbl.pack(anchor="w", padx=10, pady=(2, 0))
                bench_lbl = ctk.CTkLabel(card, text="單次運行", font=ctk.CTkFont(size=11), text_color="#888888")
                bench_lbl.pack(anchor="w", padx=10, pady=(1, 5))
            else:
                val_lbl = ctk.CTkLabel(card, text="--", font=ctk.CTkFont(size=18, weight="bold"), text_color=color)
                val_lbl.pack(anchor="w", padx=10, pady=(2, 0))
                bench_lbl = ctk.CTkLabel(card, text="此輪運行: --", font=ctk.CTkFont(size=11), text_color="#888888")
                bench_lbl.pack(anchor="w", padx=10, pady=(1, 5))
            
            self.kpi_cards.append(val_lbl)
            self.bench_labels.append(bench_lbl)

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

        # 初始化日誌 Tag 樣式 (依乘客優先級別著色)
        self.log_textbox.tag_config("l3_emergency", foreground="#eb3b5a")     # 鮮紅色 (急診)
        self.log_textbox.tag_config("l2_staff", foreground="#fa8231")         # 橘黃色 (醫護)
        self.log_textbox.tag_config("l1_wheelchair", foreground="#3498db")    # 寶藍色 (輪椅)
        self.log_textbox.tag_config("l0_normal", foreground="#a0a0a0")        # 灰色 (普通乘客 / 指派)
        self.log_textbox.tag_config("info", foreground="#666666")             # 暗灰色 (系統訊息)

        # 載入基準測試數據與初始化基準標籤
        self.benchmark_data = self.load_benchmark_data()
        self.update_benchmark_kpis()

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
        margin_right = 110
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
        """新增一條日誌並自動滾動 (支援按重要程度與乘客優先權著色)"""
        self.log_textbox.configure(state="normal")
        
        # 根據日誌內容決定標籤顏色
        if "🚨" in text or "急診" in text or "強佔" in text:
            tag = "l3_emergency"
        elif "👨‍⚕️" in text or "醫護" in text:
            tag = "l2_staff"
        elif "♿" in text or "輪椅" in text:
            tag = "l1_wheelchair"
        elif "普通" in text or "外呼指派" in text or "任務分配" in text or "✅" in text:
            tag = "l0_normal"
        else:
            tag = "info"
            
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
        self.paused = False
        self.start_btn.configure(state="disabled")
        self.pause_btn.configure(state="normal", text="⏸️ 暫停")
        self.stop_btn.configure(state="normal")
        self.agent_dropdown.configure(state="disabled")
        self.scenario_dropdown.configure(state="disabled")
        self.seed_entry.configure(state="disabled")
        self.random_seed_btn.configure(state="disabled")

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

        # 決定客流隨機編號
        try:
            seed = int(self.seed_entry.get())
        except ValueError:
            seed = 42
            self.add_log("⚠️ 客流編號格式錯誤，使用預設編號 42")
        self.add_log(f"🎲 使用模擬客流編號 (Seed)：{seed}")

        # 建立環境
        self.env = HospitalElevatorEnv(config=config)
        self.env.reset(seed=seed)

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

        # 重設頂部 KPI 實時數值
        self.kpi_cards[0].configure(text="0 秒")
        for i in range(1, 5):
            self.bench_labels[i].configure(text="此輪運行: --")

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
        self.paused = False
        if self.loop_timer_id is not None:
            self.after_cancel(self.loop_timer_id)
            self.loop_timer_id = None
        
        self.start_btn.configure(state="normal")
        self.pause_btn.configure(state="disabled", text="⏸️ 暫停")
        self.stop_btn.configure(state="disabled")
        self.agent_dropdown.configure(state="normal")
        self.scenario_dropdown.configure(state="normal")
        self.seed_entry.configure(state="normal")
        self.random_seed_btn.configure(state="normal")
        self.add_log("⏹️ 模擬已被使用者強制停止。")

    def simulation_step(self):
        """核心物理模擬與調度遞迴時鐘"""
        if not self.running or self.paused:
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
            
            direction_str = "向上" if call_pop.direction == 1 else "向下"
            action_desc = f"⏱️ 模擬 {int(env.building.current_time)}秒：【外呼指派】將 {call_pop.floor + 1}樓 {direction_str} 的等候呼叫 ➡️ 分配給【電梯 {action}】"
            self.add_log(action_desc)

        # (b) 推進物理時間 1 秒
        current_sim_time = env.building.current_time + env.dt
        new_passengers = env.traffic_gen.generate_arrivals(current_sim_time, env.dt)
        
        for p in new_passengers:
            if p.priority_level == 3:
                env.priority_system.check_and_apply_preemption(env.building, p.origin_floor)
                self.add_log(f"🚨 【緊急強佔】偵測到 {p.origin_floor + 1}樓 有急診呼叫！系統立即啟動強佔，命令最近電梯火速前往！")
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
                
                p_names = {0: "普通", 1: "♿輪椅", 2: "👨‍⚕️醫護", 3: "🚨急診"}
                log_p = f"✅ 【載客送達】電梯 {elev_id} 成功送達一員【{p_names.get(p_level, '普通')}】乘客（該乘客等候時間：{wait_t:.1f} 秒）"
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
            self.paused = False
            self.start_btn.configure(state="normal")
            self.pause_btn.configure(state="disabled", text="⏸️ 暫停")
            self.stop_btn.configure(state="disabled")
            self.agent_dropdown.configure(state="normal")
            self.scenario_dropdown.configure(state="normal")
            self.seed_entry.configure(state="normal")
            self.random_seed_btn.configure(state="normal")
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
        margin_right = 110
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
            l0_cnt = sum(1 for p in waiting_passengers if p.priority_level == 0)
            l1_cnt = sum(1 for p in waiting_passengers if p.priority_level == 1)
            l2_cnt = sum(1 for p in waiting_passengers if p.priority_level == 2)
            l3_cnt = sum(1 for p in waiting_passengers if p.priority_level == 3)
            x_text = w - margin_right + 15
            
            # (b) 在電梯右方到邊界之間，繪製與先前 pygame 一致的排隊乘客圖示 (60 FPS 平滑顯示)
            # 電梯3的中心為 margin_left + 40 + 3 * 70 = 270 (當 margin_left=60)
            start_x = margin_left + 40 + 3 * 70 + 30
            space_available = w - margin_right - start_x - 10
            max_passengers_to_show = 10
            # 加大間距，使得圖示與文字更大時不會重疊
            offset_x = max(18, min(26, space_available / max_passengers_to_show if space_available > 0 else 20))

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
                
                # 如果是急診乘客，繪製一個雙環脈動效果 (大一點且外圈閃爍更清晰)
                if p.priority_level == 3:
                    pulse_radius = 12 + int(4 * np.sin(time.time() * 8)) # 使用系統時間模擬脈動
                    self.canvas.create_oval(px - pulse_radius, py - pulse_radius, px + pulse_radius, py + pulse_radius, outline="#eb3b5a", width=2.0)
                
                # 繪製更醒目、更大的代表上行/下行三角形 (大小由 14x14 加大為 16x16)
                if p.direction == 1:
                    # 往上三角形
                    self.canvas.create_polygon(px, py - 8, px - 8, py + 8, px + 8, py + 8, fill=color, outline="")
                else:
                    # 往下三角形
                    self.canvas.create_polygon(px, py + 8, px - 8, py - 8, px + 8, py - 8, fill=color, outline="")

                # 在三角形上方繪製其目的地樓層數字 (字體大小由 9 加大為 10，更加清晰可讀)
                dest_lbl = f"{p.destination_floor + 1}"
                self.canvas.create_text(px, py - 14, text=dest_lbl, fill=color, font=("Arial", 10, "bold"))

            # 繪製統計數字標籤（包含四個優先級：👥普通, ♿輪椅, 👨‍⚕️醫護, 🚨急診）
            text_parts = []
            if l3_cnt > 0:
                text_parts.append(f"🚨{l3_cnt}")
            if l2_cnt > 0:
                text_parts.append(f"👨‍⚕️{l2_cnt}")
            if l1_cnt > 0:
                text_parts.append(f"♿{l1_cnt}")
            if l0_cnt > 0:
                text_parts.append(f"👥{l0_cnt}")
                
            label_text = " ".join(text_parts)
            
            # 依據等候隊列中最高優先級決定標籤顏色與字型
            if l3_cnt > 0:
                text_color = "#eb3b5a"  # 鮮紅色 (急診)
                text_font = ("Arial", 11, "bold")
            elif l2_cnt > 0:
                text_color = "#fa8231"  # 橘黃色 (醫護)
                text_font = ("Arial", 11, "bold")
            elif l1_cnt > 0:
                text_color = "#3867d6"  # 亮藍色 (輪椅)
                text_font = ("Arial", 11, "bold")
            else:
                text_color = "#a0a0a0"  # 灰色 (普通)
                text_font = ("Arial", 11)
                
            if label_text:
                self.canvas.create_text(x_text, y, text=label_text, fill=text_color, font=text_font, anchor="w")

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
        # 1. 模擬時間大字更新
        self.kpi_cards[0].configure(text=f"{int(self.env.building.current_time)} 秒")
        # 2. 實時指標小字更新 (大字保持基準值不變)
        self.bench_labels[1].configure(text=f"此輪運行: {awt:.2f} 秒")
        self.bench_labels[2].configure(text=f"此輪運行: {ert:.2f} 秒")
        self.bench_labels[3].configure(text=f"此輪運行: {ecr:.2f} %")
        self.bench_labels[4].configure(text=f"此輪運行: {nss} 次")

    def load_benchmark_data(self):
        """讀取基準測試 JSON 資料"""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        json_path = os.path.join(base_dir, "docs", "benchmark_results.json")
        try:
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return None
        except Exception as e:
            print(f"Error loading benchmark results: {e}")
            return None

    def update_benchmark_kpis(self, *args):
        """依據下拉選單更新頂部 KPI 卡片的基準測試平均值"""
        if not self.benchmark_data:
            for i in range(1, 5):
                self.kpi_cards[i].configure(text="--")
            return

        selected_scenario = self.scenario_dropdown.get()
        selected_agent = self.agent_dropdown.get()

        # 對應的 JSON 鍵值
        AGENT_MAPPING = {
            "MAPPO (嵌入+冷啟動版)": "MAPPO",
            "MaskablePPO": "MaskablePPO",
            "SARSA(λ)": "SARSA(λ)",
            "Nearest Car (規則式)": "Nearest Car"
        }

        agent_key = AGENT_MAPPING.get(selected_agent)
        scenarios = self.benchmark_data.get("scenarios", {})
        scenario_data = scenarios.get(selected_scenario, {})
        agent_data = scenario_data.get(agent_key, {}) if scenario_data else {}

        if agent_data:
            awt = agent_data.get("awt", 0.0)
            ert = agent_data.get("ert", 0.0)
            ecr = agent_data.get("ecr", 0.0)
            nss = agent_data.get("nss", 0.0)

            # 更新基準大字 (kpi_cards 包含 5 個，第 0 個是時間)
            self.kpi_cards[1].configure(text=f"{awt:.2f} 秒")
            self.kpi_cards[2].configure(text=f"{ert:.2f} 秒")
            self.kpi_cards[3].configure(text=f"{ecr:.2f} %")
            self.kpi_cards[4].configure(text=f"{nss:.2f} 次")
        else:
            for i in range(1, 5):
                self.kpi_cards[i].configure(text="--")

        # 同步重設「此輪運行」的小字為未開始狀態
        for i in range(1, 5):
            self.bench_labels[i].configure(text="此輪運行: --")

    def generate_random_seed_in_entry(self):
        """隨機產生一個模擬客流編號並寫入輸入框"""
        random_seed = int(np.random.randint(0, 100000))
        self.seed_entry.delete(0, "end")
        self.seed_entry.insert(0, str(random_seed))

    def pause_simulation(self):
        """切換暫停/繼續狀態"""
        if not self.running:
            return
        
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.configure(text="▶️ 繼續")
            self.add_log("⏸️ 模擬已暫停。")
            if self.loop_timer_id is not None:
                self.after_cancel(self.loop_timer_id)
                self.loop_timer_id = None
        else:
            self.pause_btn.configure(text="⏸️ 暫停")
            self.add_log("▶️ 模擬繼續運行。")
            # 重新啟動時鐘
            self.simulation_step()

if __name__ == "__main__":
    app = ElevatorApp()
    app.mainloop()
