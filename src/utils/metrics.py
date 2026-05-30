import numpy as np
import scipy.stats as stats
from typing import Tuple, Dict

def calculate_t_test(group1: list, group2: list) -> Tuple[float, float]:
    """計算兩組獨立樣本的 t 檢定值與 p-value (Welch's t-test)"""
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
    return float(t_stat), float(p_val)

def calculate_confidence_interval(group1: list, group2: list, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    計算兩組獨立樣本均值差的信賴區間 (Confidence Interval)
    回傳: (均值差, 信賴區間下界, 信賴區間上界)
    """
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    mean_diff = m1 - m2
    
    # 標準誤差 (Standard Error of Difference)
    se = np.sqrt(v1 / n1 + v2 / n2)
    
    if se == 0:
        return float(mean_diff), float(mean_diff), float(mean_diff)
        
    # 自由度 (Welch-Satterthwaite equation)
    numerator = (v1 / n1 + v2 / n2) ** 2
    denominator = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    df = numerator / denominator
    
    # 雙尾 t 臨界值
    alpha = 1.0 - confidence
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    
    margin_of_error = t_crit * se
    lower_bound = mean_diff - margin_of_error
    upper_bound = mean_diff + margin_of_error
    
    return float(mean_diff), float(lower_bound), float(upper_bound)

def calculate_cohens_d(group1: list, group2: list) -> float:
    """計算兩組獨立樣本的 Cohen's d 效應量"""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # 處理零方差情況
    if v1 == 0.0 and v2 == 0.0:
        return 0.0
        
    # 合併標準差 (Pooled Standard Deviation)
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    pooled_sd = np.sqrt(pooled_var)
    
    if pooled_sd == 0:
        return 0.0
        
    d = (m1 - m2) / pooled_sd
    return float(d)

def generate_comparative_stats(group_ppo: list, group_baseline: list) -> Dict[str, float]:
    """生成完整的對比統計數據報告"""
    t_stat, p_val = calculate_t_test(group_ppo, group_baseline)
    diff, ci_lower, ci_upper = calculate_confidence_interval(group_ppo, group_baseline)
    cohens_d = calculate_cohens_d(group_ppo, group_baseline)
    
    return {
        "mean_ppo": float(np.mean(group_ppo)),
        "mean_baseline": float(np.mean(group_baseline)),
        "mean_diff": diff,
        "t_statistic": t_stat,
        "p_value": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cohens_d": cohens_d
    }
