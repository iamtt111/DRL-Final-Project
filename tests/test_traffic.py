import pytest
import numpy as np
from src.envs.traffic_generator import PoissonTrafficGenerator, HospitalTrafficGenerator

def test_poisson_traffic_generator():
    config = {
        "num_floors": 5,
        "duration_minutes": 10,
        "total_passengers": 600,  # 平均每秒 1 人
        "flow_distribution": {
            "incoming": 1.0,      # 全部為進入流量
            "outgoing": 0.0,
            "interfloor": 0.0
        }
    }
    
    gen = PoissonTrafficGenerator(config)
    rng = np.random.default_rng(42)
    gen.reset(rng)
    
    # 生成 10 秒的乘客
    passengers = gen.generate(current_time=0.0, dt=10.0)
    assert isinstance(passengers, list)
    for p in passengers:
        assert p.origin_floor == 0
        assert p.destination_floor > 0
        assert p.priority_level == 0

def test_hospital_traffic_generator():
    config = {
        "building": {
            "num_floors": 10
        },
        "traffic": {
            "duration_minutes": 10,
            "total_passengers": 0  # 無基礎流量，便於測試優先事件
        },
        "priority_events": {
            "emergency_rate": 0.5,  # 高發生率
            "staff_rate": 0.0,
            "wheelchair_rate": 0.0
        }
    }
    
    gen = HospitalTrafficGenerator(config)
    rng = np.random.default_rng(42)
    gen.reset(rng)
    
    passengers = gen.generate_arrivals(current_time=0.0, dt=10.0)
    emergencies = [p for p in passengers if p.priority_level == 3]
    assert len(emergencies) > 0
    for p in emergencies:
        assert p.priority_level == 3
        assert p.origin_floor != p.destination_floor
