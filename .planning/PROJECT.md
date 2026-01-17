# Belief-Sensitive Embodied Assistance Research

## What This Is

A complete, research-grade GitHub repository implementing a VirtualHome-based benchmark and methods for belief-sensitive embodied assistance under object-centered false belief (Theory of Mind). The system features a "human" agent performing long-horizon household tasks while holding false beliefs about object locations due to occlusion/partial observability. A helper agent observes, infers both the human's goal AND belief state, detects false beliefs, and assists via actions or communication. The repository is fully reproducible, automation-first, and includes scripts to generate episodes, run models, compute metrics, and produce plots plus a technical report.

## Core Value

**Reproducible, research-grade experiments that demonstrate belief-sensitive assistance outperforms reactive and goal-only baselines on false-belief detection (AUROC), detection latency, task completion improvements, and wasted action reduction.** The system must work end-to-end with both VirtualHome and a fallback GridHouse simulator, with independent verification at each phase.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Complete repository scaffold with clean Python packaging, configs, tests, CI
- [ ] VirtualHome task runner + episode generator supporting false-belief interventions
- [ ] GridHouse fallback simulator (self-contained, no external dependencies)
- [ ] Baseline helper models (reactive, goal-only)
- [ ] Belief-sensitive helper with online belief tracking (particle filter/Bayesian filter)
- [ ] Automated experiment runner for multiple conditions with result saving
- [ ] Automated analysis: metrics, plots, tables, technical report generation
- [ ] One-command reproduction: `make reproduce` produces figures and report on small dataset
- [ ] Documentation: README with quickstart, methodology, extension guide
- [ ] VirtualHome integration with installation verification and separate test suite
- [ ] In-browser testing after each change with validation checks
- [ ] Maximum flexibility and customization for research, automation, and data collection at scale
- [ ] Detailed technical report with extensive tests, tasks, and data collection
- [ ] Independent verification of VirtualHome simulations at each phase

### Out of Scope

- SOTA RL training — using scripted human policies and lightweight inference policies
- GPU requirements — must run on laptop
- Proprietary APIs — optional LLM scorer can be stubbed behind interface
- Chasing state-of-the-art performance — focus on research methodology and reproducibility

## Context

**Research Background:**
- False-belief tasks are classic Theory of Mind paradigms from cognitive science
- Goal inference alone is insufficient for effective embodied assistance
- Belief inference enables detection of false beliefs and proactive correction
- VirtualHome provides realistic household simulation for embodied AI research

**Technical Environment:**
- Python-based research codebase with clean packaging
- VirtualHome simulator (may have installation challenges, must be robustly handled)
- GridHouse fallback for guaranteed reproducibility
- Config-driven experiments (Hydra/Pydantic)
- Deterministic seeding for reproducibility
- Parquet/JSONL episode storage

**Implementation Approach:**
- Phased build: Core interfaces → Baselines → Belief-PF → VirtualHome → Experiments → Analysis
- Each phase must compile/run before moving on
- Independent verification of VirtualHome at each phase
- Maximum flexibility for research extensions and data collection at scale

## Constraints

- **Compatibility**: Must run on laptop (no GPU requirement)
- **Dependencies**: No proprietary APIs (LLM scorer optional/stubbed)
- **Reproducibility**: Deterministic seeding, artifact saving, manifest tracking
- **Robustness**: VirtualHome must install and work, with separate tests and in-browser validation
- **Code Quality**: Type hints, docstrings, clean OOP boundaries
- **Self-contained**: No fetching datasets from internet during runtime
- **Verification**: Independent testing of VirtualHome simulations at each phase

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| GridHouse as fallback simulator | Ensures reproducibility even if VirtualHome fails | — Pending |
| Particle filter for belief tracking | Enables online inference over (goal, object locations) | — Pending |
| Config-driven experiments | Enables large-scale automation and data collection | — Pending |
| Parquet for episode storage | Efficient, schema-enforced storage for research data | — Pending |
| Separate VirtualHome test suite | Early error detection and validation | — Pending |
| In-browser testing after changes | Ensures VirtualHome integration works at each phase | — Pending |

---
*Last updated: 2024-12-19 after initialization*
