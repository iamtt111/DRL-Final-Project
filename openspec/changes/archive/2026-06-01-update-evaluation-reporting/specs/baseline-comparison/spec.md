## ADDED Requirements
### Requirement: Console Markdown Table Output
The comparison engine SHALL print a Markdown table summarizing the core metrics (AWT, ERT, ECR, NSS) for all traffic scenarios side-by-side for MAPPO, SARSA(λ), and Nearest Car to the console.

#### Scenario: Print table after benchmarking
- **WHEN** benchmark runs successfully
- **THEN** it outputs side-by-side Markdown tables in the console

### Requirement: Evaluation Report Dashboard
The comparison engine SHALL automatically generate or update a Markdown dashboard at `docs/evaluation_report.md` containing the Markdown tables of core metrics and embedding the newly generated timestamped evaluation charts.

#### Scenario: Generate report after benchmarking
- **WHEN** benchmarking completes and charts are generated
- **THEN** it saves `docs/evaluation_report.md` with the embedded images and tables
