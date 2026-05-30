from enum import Enum
from dataclasses import dataclass

class EventType(Enum):
    HALL_CALL_NEW = "hall_call_new"
    HALL_CALL_SERVED = "hall_call_served"
    PASSENGER_BOARDED = "passenger_boarded"
    PASSENGER_DELIVERED = "passenger_delivered"
    ELEVATOR_ARRIVED = "elevator_arrived"
    ELEVATOR_IDLE = "elevator_idle"
    PRIORITY_TRIGGERED = "priority_triggered"
    PRIORITY_PREEMPTION = "priority_preemption"

@dataclass
class Event:
    type: EventType
    timestamp: float
    data: dict
