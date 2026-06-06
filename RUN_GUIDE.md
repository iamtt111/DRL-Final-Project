# 🏥 智慧醫院電梯群控與優先調度系統 (Hospital EGCS) - 運行指引文件

本指引文件旨在幫助團隊成員與評審快速設定環境、運行本專案的互動式模擬器（GUI Demo）、訓練強化學習模型，以及重現學術評估報告與圖表。

---

## 📌 專案主要特性
1. **多部電梯群控**：模擬 4 部電梯、16 層樓的醫院大樓環境。
2. **四類乘客優先權級（Priority L0-L3）**：包含普通乘客 (L0)、輪椅患者 (L1)、醫護人員 (L2)、急診病床 (L3)。
3. **四種調度演算法支援**：
   * 傳統工業規則：**Nearest Car (最近車輛優先)**
   * 經典表格式 RL：**SARSA(λ)**
   * 單智能體深度 RL：**MaskablePPO**
   * 多智能體深度 RL（本專案終極方案）：**MAPPO**
4. **CustomTkinter 實時互動視覺化模擬器**：支援即時切換演算法、調整乘客生成速度、自訂極端情境（如早高峰、晚高峰、災難危機等）、以及區分緊急程度的實時日誌系統。

---

## ⚙️ 環境配置與安裝

建議使用 **Python 3.10** 的環境（專案已在此版本進行完整測試）。

### 1. 建立並啟動虛擬環境 (以 Windows PowerShell 為例)
```powershell
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境
.\venv\Scripts\Activate.ps1
```

### 2. 安裝相依套件與本機開發模式
```powershell
# 升級 pip
python -m pip install --upgrade pip

# 安裝基本依賴套件
pip install -r requirements.txt

# 以開發者編輯模式安裝本專案套件 (這步會註冊 elevator_egcs 模組)
pip install -e .
```

---

## 🖥️ 互動式 GUI 模擬 Demo 運行指引

我們移除了舊版不穩定且會頻繁閃爍更新的 Streamlit 介面，改用 **CustomTkinter** 打造了高質感的桌面應用程式，以便於進行 Demo 展示。

### 1. 啟動指令
```powershell
python scripts/demo_app.py
```

### 2. GUI 介面功能與操作說明
* **左側電梯動畫展示區**：即時動態呈現 4 部電梯的樓層位置、上下行狀態、開關門狀態，以及電梯內部的載客優先權人數。
* **左下方候梯隊伍**：動態展示各樓層大廳在電梯外排隊等待的乘客。
  * 乘客外觀顏色與其優先級對應：**藍色 (普通人 L0)**、**黃色 (輪椅 L1)**、**綠色 (醫生 L2)**、**紅色 (急診 L3)**。
* **右側演算法切換與控制面板**：
  * **演算法下拉選單**：支援即時在 `Nearest Car`、`SARSA(λ)`、`MaskablePPO`、`MAPPO` 之間切換。
  * **情境設定下拉選單**：可即時切換為 `Morning Peak (早高峰)`、`Evening Peak (晚高峰)`、`Mixed Traffic (混合流量)`、`Disaster Crisis (災難危機)`，觀察電梯在不同高負載交通流下的調度策略。
  * **模擬控制按鈕**：提供「開始」、「暫停」、「重置」功能。
  * **乘客生成率滑桿 (Traffic Rate)**：可即時調整新乘客的出現頻率（每分鐘生成人數）。
  * **時間膨脹滑桿 (Time Scale)**：可調整模擬的加速倍率，方便快速觀測長時間的系統表現。
* **右下方實時調度日誌 (Live Log)**：
  * 即時顯示系統的調度日誌（包含新乘客到達、電梯指派、乘客上梯/到達等事件）。
  * **自動區分緊急程度**：針對急診 (L3 Emergency) 的事件，會以 **[EMERGENCY] 加上醒目的紅色粗體** 顯示，幫助教授一眼看出 AI 正在對急診任務實施優先資源傾斜。
* **視窗大小自我調整**：視窗已針對筆記型電腦與實驗室螢幕進行適配最佳化，所有內容均能清晰呈現，不會被遮擋。

---

## 🏋️ 強化學習模型訓練指引

若您需要重新訓練模型或調整獎勵函數，請使用以下腳本。

### 1. 訓練多智能體 MAPPO 模型 (本專案核心)
```powershell
python -m scripts.train_mappo --timesteps 1500000
```
* 訓練完成後的模型與最佳檢查點 (Best Model) 將會自動保存在 `models/mappo/` 目錄中。
* 設定參數如網絡結構、學習率等可參考 `configs/mappo.yaml`。

### 2. 訓練單智能體 PPO (MaskablePPO) 模型
```powershell
python scripts/train.py --config configs/train_ppo.yaml
```
* 模型將保存在 `models/ppo/` 目錄下。

### 3. 訓練經典 SARSA(λ) 模型
```powershell
python scripts/train_sarsa.py
```
* 權重將保存在 `models/sarsa/sarsa_weights.npz`。

---

## 📊 基準演算法比較與評估報告生成

為保障學術評估的嚴謹性，本專案提供了一鍵運行基準測試與學術圖表重現的方案。

### 1. 運行 100 Episodes 基準測試模擬 (耗時較長)
```powershell
python -m scripts.compare_baselines --episodes 100
```
* 該腳本會在四個交通流情境下，對四種演算法各進行 100 個 Episode 的靜默模擬測試。
* 測試結束後，會將詳細的所有乘客乘梯歷程與時間數據寫入至 `docs/benchmark_results.json`，並在終端機輸出 AWT 與 ERT 的獨立樣本雙尾 t 檢定 (Welch's t-test) p-value 顯著性結果。

### 2. 重現學術評估報告與 4 色分析圖表 (推薦，使用已有數據一鍵生成)
如果您不需要重新運行漫長的模擬，可以直接使用專案中已保存的 `docs/benchmark_results.json` (包含 100 episodes 的完整記錄)，來重新渲染所有論文圖表並更新評估報告：
```powershell
python scripts/regenerate_report.py
```
* **產出物 1 (圖表)**：自動於 `docs/images/` 目錄中生成以下 4 色演算法對比圖：
  * `comparison_tradeoff_*.png`：AWT vs ERT 的多目標決策 **Pareto 邊界分析散點圖**。
  * `comparison_cdf_*.png`：急診等待時間的**累積機率分佈 (CDF) 圖**，包含 95% 安全響應對齊線與 10s/15s 關鍵閾值。
  * `comparison_disaster_*.png`：災難情境下，按 Priority (L0-L3) 分級的 **95% 最壞等候時間與平均等候時間對比箱線圖**。
  * `comparison_awt_*.png`：AWT 分組條形圖（附帶獨立 t 檢定 p-value 顯著性對比括號與星號標註）。
  * `comparison_priority_boxplot.png`：乘客優先權等候時間分佈箱線圖。
  * `comparison_radar.png`：多目標效能綜合雷達圖。
  * `training_convergence.png`：訓練收斂曲線圖。
* **產出物 2 (報告)**：更新 **`docs/evaluation_report.md`** 評估報告，包含最新的 4 欄情境指標對比表格及深入的學術深度分析。

---

## 🧹 專案結構說明
* `src/envs/`：HospitalEGCS 醫院電梯 Gymnasium 模擬環境。
* `src/agents/`：四種調度策略 Agent 的實作代碼。
* `src/visualization/`：Pygame 渲染器與 Matplotlib 圖表繪製模組。
* `configs/`：YAML 模型設定檔。
* `models/`：存放訓練完成的模型。
* `docs/`：包含 PDF 參考論文、對照結果 json、與繁體中文學術評估報告。
