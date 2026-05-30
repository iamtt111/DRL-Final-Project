import pytest
from src.envs.passenger import Passenger, PassengerState

def test_passenger_init():
    p = Passenger(id=1, arrival_time=10.0, origin_floor=0, destination_floor=5, priority_level=0)
    assert p.id == 1
    assert p.arrival_time == 10.0
    assert p.origin_floor == 0
    assert p.destination_floor == 5
    assert p.priority_level == 0
    assert p.state == PassengerState.WAITING
    assert p.wait_start_time == 10.0
    assert p.board_time is None
    assert p.arrive_time is None

def test_passenger_direction():
    p1 = Passenger(id=1, arrival_time=10.0, origin_floor=0, destination_floor=5, priority_level=0)
    assert p1.direction == 1
    
    p2 = Passenger(id=2, arrival_time=10.0, origin_floor=5, destination_floor=0, priority_level=0)
    assert p2.direction == -1

def test_passenger_wait_duration():
    p = Passenger(id=1, arrival_time=10.0, origin_floor=0, destination_floor=5, priority_level=0)
    # 在等待時，等待時間為當前時間減去到達時間
    assert p.get_wait_duration(15.0) == 5.0
    
    # 上車後
    p.board_time = 18.0
    p.state = PassengerState.IN_TRANSIT
    assert p.get_wait_duration(25.0) == 8.0
