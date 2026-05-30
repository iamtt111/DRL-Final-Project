import pytest
import numpy as np
from src.utils.metrics import calculate_t_test, calculate_confidence_interval, calculate_cohens_d, generate_comparative_stats

def test_t_test_and_cohens_d():
    group1 = [10.0, 12.0, 11.0, 13.0, 12.0]
    group2 = [15.0, 17.0, 16.0, 18.0, 17.0]
    
    t_stat, p_val = calculate_t_test(group1, group2)
    assert p_val < 0.05
    
    diff, lower, upper = calculate_confidence_interval(group1, group2)
    assert lower < diff < upper
    assert diff == pytest.approx(-5.0)
    
    d = calculate_cohens_d(group1, group2)
    assert d < -2.0

def test_generate_comparative_stats():
    group1 = [1, 2, 3, 4, 5]
    group2 = [2, 3, 4, 5, 6]
    
    report = generate_comparative_stats(group1, group2)
    assert "mean_ppo" in report
    assert "mean_baseline" in report
    assert "mean_diff" in report
    assert report["mean_diff"] == -1.0
