## Why
To reduce the high number of starts and stops (NSS) and improve overall average waiting time (AWT), we need to discourage the MAPPO agents from changing directions or stopping too frequently (second-by-second jitter).

## What Changes
- **ADDED**: An action transition penalty of 0.5 points in `HospitalElevatorMAEnv` when an elevator transitions from moving (MOVE_UP/MOVE_DOWN) to stopping/reversing.
- **ADDED**: Tracking of previous actions to detect transitions.
- **ADDED**: Recording of the `action_transition_penalty` component in step information.

## Impact
- Affected specs: `elevator-ma-env`
- Affected code: `src/envs/elevator_ma_env.py`
