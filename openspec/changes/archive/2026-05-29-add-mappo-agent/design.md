## Context
The single-agent MaskablePPO agent handles the global dispatch task by selecting one elevator out of $N_e$ for each hall call. This leads to a complex observation space and poor scaling, causing long wait times (AWT) for ordinary passengers because the agent prioritizes avoiding emergency penalties and conserving energy. A decentralized multi-agent approach (MAPPO) with cooperative bidding distributes the state representation and decision-making, allowing each elevator agent to focus on its own physical state and proximity to the task.

## Goals / Non-Goals
### Goals
- Introduce cooperative multi-agent PPO (MAPPO) with parameter-sharing decentralized actors.
- Maintain compatibility with the existing simulator and code bases.
- Avoid introducing heavy external multi-agent RL libraries.
- Allow running training, evaluation, comparison, and the interactive demo using the new MAPPO agent.

### Non-Goals
- Modify or break the existing single-agent PPO, SARSA, or Nearest Car baselines.
- Change the underlying elevator simulator physical dynamics or event queue logic.

## Decisions

### 1. Bidding-Based Multi-Agent Modeling
- **Concept**: Instead of controlling raw elevator movements directly (which requires continuous action spaces at every second), we model each elevator as an agent that "bids" for the pending hall call.
- **Action**: Continuous value $a_i \in [0, 1]$ output by agent $i$'s actor network representing the bidding score (suitability) for the hall call.
- **Resolution**: The environment assigns the hall call to the elevator with the highest bid score: $i^* = \operatorname{argmax}_i(a_i)$. If an elevator is out of service or full, the environment overrides its action/bid to a large negative number (e.g. $-10^9$) to prevent assignment.

### 2. Observation Spaces
- **Local Observation ($o_i$)**: A compact 15-dimensional vector:
  - Call characteristics (normalized source floor, direction, priority level, source floor type, call wait time).
  - Self-elevator state (normalized current position, direction, load, door state, distance to call, compatibility flag, pending stops count, door timer remaining, out-of-service flag).
- **Global Centralized State ($S$)**: Concatenation of all agents' local observations, supplemented with building global queue summaries, used by the Centralized Critic network during training.

### 3. Policy Architecture (Centralized Training, Decentralized Execution)
- **Parameter Sharing**: All elevator agents share the weights of the Actor Network to speed up convergence and ensure homogeneous cooperative behavior.
- **Actor Network**: MLP mapping local observation $o_i$ to a Gaussian distribution over the bidding action $a_i \in [0, 1]$.
- **Critic Network**: Centralized MLP mapping global state $S$ to a single value estimate $V(S)$ for GAE calculations during training.
- **Training Algorithm**: Standard PPO update on collected multi-agent trajectories (actor policy loss with clipping, critic MSE loss, and entropy regularization).

## Risks / Trade-offs

### Non-convergence or lazy learning
- **Risk**: Agents might learn to output flat bids (e.g., all 0.5) resulting in random or static assignments.
- **Mitigation**: Introduce local energy penalties in individual agent rewards and shape the team cooperative reward to heavily penalize long wait times. Add entropy regularization to ensure exploration.

## Migration Plan
None required, as MAPPO is added as new files and does not modify the baseline data models or schemas.

## Open Questions
None.
