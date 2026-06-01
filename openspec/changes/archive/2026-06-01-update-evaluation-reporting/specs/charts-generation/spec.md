## ADDED Requirements
### Requirement: Timestamped Chart File Names
The chart generator SHALL save all generated charts with a timestamp appended to the filename to prevent caching issues.

#### Scenario: Appending timestamp to file names
- **WHEN** charts are saved
- **THEN** their filenames contain a timestamp matching `_%Y%m%d_%H%M%S` format
