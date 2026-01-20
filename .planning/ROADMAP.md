# Research Project Roadmap

## Phase 9: Data & Methodology Fixes (COMPLETED)
**Status**: ✅ Completed
**Summary**: Fixed data leakage in particle filter, differentiated helper policies, improved efficiency metrics, enabled task completion tracking.

**Reference**: `.planning/phases/09-data-methodology-fixes/ISSUES-AND-FIXES.md`

---

## Phase 10: Statistical Strengthening and Publication Readiness
**Status**: ✅ COMPLETED
**Completed**: 2025-01-20
**Priority**: Critical
**Builds On**: Phase 9 fixes

### Objectives
- Fix AUROC computation and variance issues
- Add bootstrap confidence intervals
- Implement temporal evaluation metrics
- Fix efficiency metric calculation
- Add partial_false_belief condition
- Add statistical reporting (CIs, effect sizes, p-values)
- Create visual diagnostics
- Document all changes

### Deliverables
- Refactored `run_phase9_experiments.py`
- Enhanced `metrics.py` with bootstrap CIs
- New `analyze_results.py` with statistical tests
- Updated `episode_generator.py` with partial condition
- New visualization scripts
- `METHODOLOGY_CHANGES.md` documentation

---

## Future Phases (TBD)
- Phase 2: Neural Prior Learning (Optional)
- Phase 3: Hybrid Particle Filter + Neural Encoder (Optional)
- Phase 4: Adaptive Resampling Strategy (Optional)
