# Phase 6 User Acceptance Testing Checklist

**Scope**: Plans 06-01 (Core Component Tests) and 06-02 (Helper Agent Tests)

**Date**: 2025-01-17

## Test Environment Setup

- [ ] Virtual environment is activated
- [ ] All dependencies installed (`pip install -e ".[dev]"`)
- [ ] pytest is available and working

## Plan 06-01: Core Component Tests

### Episode Generator Tests (`tests/test_episode_generators.py`)

- [ ] **Test 1.1**: Run episode generator tests
  - Command: `pytest tests/test_episode_generators.py -v`
  - Expected: 11/13 tests pass (2 skipped if VirtualHome not installed)
  - Result: [ ] PASS [ ] FAIL [ ] PARTIAL

- [ ] **Test 1.2**: Verify GridHouse episode generation
  - Command: `pytest tests/test_episode_generators.py::TestGridHouseEpisodeGenerator::test_generate_episode -v`
  - Expected: Episode is generated with correct structure
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 1.3**: Verify intervention logic
  - Command: `pytest tests/test_episode_generators.py::TestGridHouseEpisodeGenerator::test_intervention_applied -v`
  - Expected: Intervention is applied at tau timestep
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 1.4**: Verify false belief creation
  - Command: `pytest tests/test_episode_generators.py::TestGridHouseEpisodeGenerator::test_false_belief_created -v`
  - Expected: False belief is created after intervention
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 1.5**: Verify deterministic seeding
  - Command: `pytest tests/test_episode_generators.py::TestGridHouseEpisodeGenerator::test_deterministic_seeding -v`
  - Expected: Same seed produces same episode
  - Result: [ ] PASS [ ] FAIL

### Particle Filter Tests (`tests/test_particle_filter.py`)

- [ ] **Test 2.1**: Run particle filter tests
  - Command: `pytest tests/test_particle_filter.py -v`
  - Expected: 14/14 tests pass
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 2.2**: Verify initialization
  - Command: `pytest tests/test_particle_filter.py::TestParticleFilter::test_initialization -v`
  - Expected: Particle filter initializes with correct number of particles
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 2.3**: Verify update and resampling
  - Command: `pytest tests/test_particle_filter.py::TestParticleFilter::test_update_with_action tests/test_particle_filter.py::TestParticleFilter::test_resampling -v`
  - Expected: Particles update correctly and resampling maintains count
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 2.4**: Verify belief extraction
  - Command: `pytest tests/test_particle_filter.py::TestParticleFilter::test_get_belief_distribution tests/test_particle_filter.py::TestParticleFilter::test_get_most_likely_goal -v`
  - Expected: Belief distributions are extracted correctly
  - Result: [ ] PASS [ ] FAIL

### Inference Module Tests (`tests/test_inference.py`)

- [ ] **Test 3.1**: Run inference module tests
  - Command: `pytest tests/test_inference.py -v`
  - Expected: 16/16 tests pass
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 3.2**: Verify goal inference
  - Command: `pytest tests/test_inference.py::TestGoalInference -v`
  - Expected: Goal inference updates correctly and converges
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 3.3**: Verify belief inference
  - Command: `pytest tests/test_inference.py::TestBeliefInference -v`
  - Expected: Belief inference tracks goals and object locations
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 3.4**: Verify likelihood models
  - Command: `pytest tests/test_inference.py::TestLikelihoodModel -v`
  - Expected: Likelihood models compute probabilities correctly
  - Result: [ ] PASS [ ] FAIL

## Plan 06-02: Helper Agent Tests

### Reactive Helper Tests

- [ ] **Test 4.1**: Run reactive helper tests
  - Command: `pytest tests/test_helper_agents.py::TestReactiveHelper -v`
  - Expected: 6/6 tests pass
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 4.2**: Verify interface compliance
  - Command: `pytest tests/test_helper_agents.py::TestReactiveHelper::test_implements_interface -v`
  - Expected: ReactiveHelper implements HelperAgent interface
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 4.3**: Verify reactive behavior
  - Command: `pytest tests/test_helper_agents.py::TestReactiveHelper::test_plan_action -v`
  - Expected: Helper reacts to visible objects correctly
  - Result: [ ] PASS [ ] FAIL

### Goal-Only Helper Tests

- [ ] **Test 5.1**: Run goal-only helper tests
  - Command: `pytest tests/test_helper_agents.py::TestGoalOnlyHelper -v`
  - Expected: 7/7 tests pass
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 5.2**: Verify goal inference integration
  - Command: `pytest tests/test_helper_agents.py::TestGoalOnlyHelper::test_goal_inference -v`
  - Expected: Goal inference updates correctly
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 5.3**: Verify action planning
  - Command: `pytest tests/test_helper_agents.py::TestGoalOnlyHelper::test_plan_action -v`
  - Expected: Actions are planned based on inferred goal
  - Result: [ ] PASS [ ] FAIL

### Belief-Sensitive Helper Tests

- [ ] **Test 6.1**: Run belief-sensitive helper tests
  - Command: `pytest tests/test_helper_agents.py::TestBeliefSensitiveHelper -v`
  - Expected: 7/7 tests pass
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 6.2**: Verify belief tracking
  - Command: `pytest tests/test_helper_agents.py::TestBeliefSensitiveHelper::test_belief_tracking -v`
  - Expected: Both goal and object location beliefs are tracked
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 6.3**: Verify false-belief detection
  - Command: `pytest tests/test_helper_agents.py::TestBeliefSensitiveHelper::test_detect_false_belief -v`
  - Expected: False beliefs are detected correctly
  - Result: [ ] PASS [ ] FAIL

### Intervention Policy Tests

- [ ] **Test 7.1**: Run intervention policy tests
  - Command: `pytest tests/test_helper_agents.py::TestInterventionPolicy -v`
  - Expected: 3/3 tests pass
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 7.2**: Verify intervention decision logic
  - Command: `pytest tests/test_helper_agents.py::TestInterventionPolicy::test_should_intervene -v`
  - Expected: Intervention decisions are made correctly
  - Result: [ ] PASS [ ] FAIL

### Integration Tests

- [ ] **Test 8.1**: Run integration tests
  - Command: `pytest tests/test_helper_agents.py::TestHelperAgentIntegration -v`
  - Expected: 3/3 tests pass
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 8.2**: Verify agents work with GridHouse
  - Command: `pytest tests/test_helper_agents.py::TestHelperAgentIntegration::test_reactive_helper_with_gridhouse tests/test_helper_agents.py::TestHelperAgentIntegration::test_belief_sensitive_helper_with_gridhouse -v`
  - Expected: All helper agents work with GridHouseEnvironment
  - Result: [ ] PASS [ ] FAIL

## Overall Test Suite Verification

- [ ] **Test 9.1**: Run all Phase 6 tests together
  - Command: `pytest tests/test_episode_generators.py tests/test_particle_filter.py tests/test_inference.py tests/test_helper_agents.py -v`
  - Expected: 68 tests pass (67 passed, 3 skipped)
  - Result: [ ] PASS [ ] FAIL [ ] PARTIAL

- [ ] **Test 9.2**: Verify test coverage
  - Command: `pytest tests/test_episode_generators.py tests/test_particle_filter.py tests/test_inference.py tests/test_helper_agents.py --cov=src/bsa/inference --cov=src/bsa/agents/helper --cov-report=term-missing`
  - Expected: Good coverage for tested modules (>70%)
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 9.3**: Verify tests are deterministic
  - Run same test twice with same seed
  - Expected: Results are identical
  - Result: [ ] PASS [ ] FAIL

## Code Quality Checks

- [ ] **Test 10.1**: Verify test code follows project standards
  - Check: Tests use pytest fixtures appropriately
  - Check: Tests have clear docstrings
  - Check: Tests are well-organized
  - Result: [ ] PASS [ ] FAIL

- [ ] **Test 10.2**: Verify edge cases are covered
  - Check: Empty inputs, single particles, degenerate cases
  - Result: [ ] PASS [ ] FAIL

## Summary

**Total Tests**: 68 (67 passed, 3 skipped expected)

**Pass Rate**: [ ] 100% [ ] >90% [ ] <90%

**Issues Found**: [ ] None [ ] Minor [ ] Major

**Overall Verdict**: [ ] APPROVED [ ] NEEDS FIXES [ ] REJECTED

## Notes

[Any additional observations or issues]
