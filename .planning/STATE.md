# Project State

## Current Position

**Phase**: 2 (Helper Models)
**Status**: Complete - All plans executed

## Accumulated Decisions

- GridHouse as fallback simulator (ensures reproducibility)
- Particle filter for belief tracking (enables online inference)
- Config-driven experiments (enables large-scale automation)
- Parquet for episode storage (efficient, schema-enforced)
- Python package structure with type hints
- Deterministic seeding throughout

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

Ready for Phase 3 (VirtualHome Backend) or Phase 4 (Experiment Harness).

## Alignment Status

On track. Repository structure matches specification. GridHouse simulator functional. Core logic implementation is next step.
