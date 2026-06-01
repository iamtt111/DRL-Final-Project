## REMOVED Requirements
### Requirement: Action Transition Penalty
**Reason**: Reverted MAPPO architecture back to stable centralized bidding/dispatch system, where motor control is handled by the SCAN heuristic and transition penalties are no longer necessary.
**Migration**: Remove transition penalty calculation from environment step logic.
