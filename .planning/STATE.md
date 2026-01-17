# Project State

## Current Position

**Phase**: 3 (VirtualHome Backend)
**Status**: In Progress - Plan 03-01 complete

## Accumulated Decisions

- GridHouse as fallback simulator (ensures reproducibility)
- Particle filter for belief tracking (enables online inference)
- Config-driven experiments (enables large-scale automation)
- Parquet for episode storage (efficient, schema-enforced)
- Python package structure with type hints
- Deterministic seeding throughout
- Virtual environment setup for VirtualHome compatibility (Python 3.9-3.11, NumPy <2.0)
- VirtualHome as optional dependency (GridHouse fallback always available)

## Deferred Issues

None yet

## Blockers/Concerns

None - Phase 1 complete

## Next Phase Readiness

Phase 2 complete:
1. ✓ Base helper interface
2. ✓ Reactive helper baseline
3. ✓ Goal inference + Goal-only helper
4. ✓ Particle filter + Likelihood models
5. ✓ Belief inference + Belief-sensitive helper + Intervention policy

Phase 3 Plan 03-01 complete:
1. ✓ Virtual environment setup script
2. ✓ VirtualHome installation script
3. ✓ Basic VirtualHomeEnvironment adapter

Ready for Plan 03-02 (Task programs library and episode generator).

## Alignment Status

On track. Repository structure matches specification. GridHouse simulator functional. Core logic implementation is next step.
