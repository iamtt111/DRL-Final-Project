# 🏥 智慧醫院電梯群控系統 (EGCS) 執行與操作指南

本指南旨在引導團隊成員與評鑑委員快速配置環境，並運行本專案的核心展示面板、物理渲染模擬器以及學術成果重製腳本。

---

## ⚙️ 一、 環境快速建置 (Environment Setup)

### 1. 系統需求
*   **Python 版本**：`>= 3.10` (推薦 `3.10.x` 或 `3.11.x`)
*   **作業系統**：Windows / macOS / Linux (GUI 部分支援各平台雙擊與視窗自適應)

### 2. 套件安裝步驟
在專案根目錄下，開啟終端機並執行以下指令：

```bash
# 1. 建立虛擬環境 (選用，但強烈建議)
python -m venv venv
source venv/bin/activate       # macOS/Linux
# Windows 系統請執行下行：
# venv\Scripts\activate

# 2. 安裝所有核心與 GUI 相依套件 (包含 stable-baselines3, customtkinter, pygame, scipy 等)
pip install -r requirements.txt

# 3. 以開發者模式安裝本專案結構
pip install -e .
```

---

## 🎮 二、 兩種視覺化展示模式 (Visualizations)

本專案提供兩個維度的視覺化展示：**CustomTkinter 全能主控台**（推薦，功能最完整）與 **Pygame 物理模擬器**。

### Mode A: CustomTkinter 互動式主控面板 (強烈推薦 ✨)
這是專門為 Demo 與實時觀察演算法決策設計的現代化 Dashboard。介面支援自適應縮放，徹底解決部分螢幕顯示不全或閃爍的問題。

*   **啟動指令**：
    ```bash
    python scripts/demo_app.py
    ```

*   **核心功能與操作指引**：
    1.  **切換演算法 (🤖 Agent)**：在左側選單中，可即時切換 **MAPPO (嵌入+冷啟動版)**、**MaskablePPO**、**SARSA(λ)** 或 **Nearest Car (規則式)**。
    2.  **切換交通流情境 (📊 Scenario)**：支援 `morning_peak` (上班尖峰)、`evening_peak` (下班尖峰)、`mixed_traffic` (混合流量) 與 `disaster_crisis` (災難危機極端流量)。
    3.  **速度調整 (⏱️ Speed)**：透過滑桿可即時調節電梯物理運行的刷新速度。
    4.  **急診事件手動注入 (🚨 Emergency Button)**：在模擬中點擊「**🚨 注入 Level 3 急診**」按鈕，會在隨機樓層生成一個急診患者（去 1 樓手術室），您可即時在右側的**實時日誌**與電梯動畫中觀察 MAPPO 如何進行「任務搶占」與「減少停靠直達」。
    5.  **日誌分級過濾 (📝 Real-time Log)**：日誌會以顏色區分重要性（紅色：急診 L3 | 黃色：醫護人員 L2 | 藍色：輪椅 L1），並提供過濾核取方塊，可方便追蹤急診進度。

---

### Mode B: Pygame 物理模擬器 (經典渲染 👾)
展示原始的 2D 物理運行，適合用來觀察電梯加速度、開關門機制以及乘客在樓層外排隊等候的即時變化。

*   **啟動指令**：
    ```bash
    # 執行 MAPPO 演算法 (帶 Pygame 渲染)
    python scripts/demo.py --agent mappo --scenario morning_peak

    # 執行 Nearest Car 傳統規則演算法
    python scripts/demo.py --agent rule --scenario morning_peak
    ```
*   **參數說明**：
    *   `--agent` 可選：`mappo`, `ppo`, `sarsa`, `rule`
    *   `--scenario` 可選：`morning_peak`, `evening_peak`, `mixed_traffic`

---

## 📊 三、 學術圖表與評估報告重製 (Academic Benchmarks)

為了確保學術嚴謹度，我們已在先前完成了 100 Episodes 的蒙地卡羅基準測試，並將詳細數據保存在 `docs/benchmark_results.json` 中。您可以免去漫長的模擬運行，直接一鍵重繪論文規格的七張學術圖表並重寫評估報告。

*   **一鍵重製指令**：
    ```bash
    python scripts/regenerate_report.py
    ```

*   **執行後成果**：
    1.  會自動讀取 100 回合原始數據，將 **MAPPO**、**MaskablePPO**、**SARSA(λ)** 與 **Nearest Car** 四種演算法進行橫向統計對比。
    2.  在 `docs/plots/` 目錄下重新渲染並存檔以下七張高清學術圖表：
        *   `comparison_tradeoff_*.png` (AWT vs. ERT 的 Pareto Frontier 雙標權衡圖)
        *   `comparison_cdf_*.png` (急診等待時間的累積機率 CDF 分佈圖)
        *   `comparison_disaster_*.png` (災難極端高負載分級箱線圖)
        *   `comparison_awt_*.png` (普通 vs. 急診等待時間對比長條圖，附 Welch's t-test 顯著性標註)
        *   `comparison_priority_boxplot_*.png` (優先級等待分布箱線圖)
        *   `comparison_radar_*.png` (多目標綜合雷達圖)
        *   `training_convergence_*.png` (強化學習訓練收斂曲線)
    3.  自動重寫 `docs/evaluation_report.md` 為最新的繁體中文版學術評估報告，並帶有 4 欄數據對比表與最新的學術圖片路徑。

---

## 💾 四、 GitHub 分支提交與上傳指引 (Git Push)

若要將您當前的本地開發成果提交並推送到 GitHub 的 `zhen` 分支，可參考以下 Git 標準工作流：

```bash
# 1. 確保目前在 zhen 分支 (若不存在則建立)
git checkout -b zhen

# 2. 檢查變動的檔案狀態
git status

# 3. 將所有變更加入暫存區
git add .

# 4. 提交變更並撰寫訊息
git commit -m "feat: update EGCS 4-algorithm baseline evaluation and customtkinter dashboard"

# 5. 推送至遠端 GitHub zhen 分支
git push origin zhen
```

---
💡 **溫馨提示**：如果其他同學在執行 `python scripts/demo_app.py` 時遇到字型警告或畫面缺少元件，請確保已完整執行 `pip install -r requirements.txt`。如有任何技術疑問，歡迎參考 [OpenSpec.md](OpenSpec.md) 規格文件。
