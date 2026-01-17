# User Acceptance Test: Phase 3 - VirtualHome Backend

**Scope:** Complete Phase 3 implementation (Plans 03-01 through 03-04)  
**Testing:** Manual user validation  
**Date:** 2025-01-16

## Pre-flight Checks

- [ ] All Phase 3 files exist
- [ ] No syntax errors in Python files
- [ ] Virtual environment is set up

## Plan 03-01: Virtual Environment & Basic Adapter

### Test 1: Virtual Environment Setup
**What to test:** Virtual environment setup script works correctly

**Steps:**
1. Check if venv exists: `Test-Path venv`
2. Verify venv has correct Python version
3. Verify dependencies installed (NumPy <2.0)

**Expected:** Venv exists, Python 3.9-3.11 (or 3.12 with note), NumPy <2.0

**Result:** [ ] Pass  [ ] Fail  [ ] Partial  [ ] Skip

---

### Test 2: VirtualHome Installation Script
**What to test:** Installation script handles VirtualHome gracefully

**Steps:**
1. Run: `python scripts/install_virtualhome.py`
2. Check: Script runs without crashes
3. Verify: Provides clear status messages

**Expected:** Script runs, provides clear status (installed or not)

**Result:** [ ] Pass  [ ] Fail  [ ] Partial  [ ] Skip

---

### Test 3: VirtualHomeEnvironment Import
**What to test:** VirtualHomeEnvironment can be imported

**Steps:**
1. Activate venv
2. Run: `python -c "from src.bsa.envs.virtualhome import VirtualHomeEnvironment; print('OK')"`
3. Check: Import succeeds OR clear error if VirtualHome not installed

**Expected:** Import works or clear error message

**Result:** [ ] Pass  [ ] Fail  [ ] Partial  [ ] Skip

---

### Test 4: VirtualHomeEnvironment Basic Methods
**What to test:** All Environment interface methods exist

**Steps:**
1. Check: `VirtualHomeEnvironment` has methods: reset, step, get_true_state, get_visible_state, get_object_locations
2. Verify: Methods match Environment interface

**Expected:** All required methods exist

**Result:** [ ] Pass  [ ] Fail  [ ] Partial  [ ] Skip

---

## Plan 03-02: Task Programs & Episode Generator

### Test 5: Task Programs Library
**What to test:** VirtualHome tasks can be imported and queried

**Steps:**
1. Run: `python -c "from src.bsa.envs.virtualhome import get_task, list_tasks; print(list_tasks()); t = get_task('prepare_meal'); print(t.critical_objects)"`
2. Verify: Tasks list correctly
3. Verify: Task details are correct

**Expected:** Tasks import, list, and query correctly

**Result:** [ ] Pass  [ ] Fail  [ ] Partial  [ ] Skip

---

### Test 6: VirtualHomeEpisodeGenerator Import
**What to test:** Episode generator can be imported

**Steps:**
1. Run: `python -c "from src.bsa.envs.virtualhome import VirtualHomeEpisodeGenerator; print('OK')"`
2. Check: Import succeeds

**Expected:** Import works

**Result:** [ ] Pass  [ ] Fail  [ ] Partial  [ ] Skip

---

### Test 7: Episode Generation (if VirtualHome available)
**What to test:** Episode generator creates episodes

**Steps:**
1. If VirtualHome installed, try: `python -c "from src.bsa.envs.virtualhome import VirtualHomeEnvironment, VirtualHomeEpisodeGenerator; env = VirtualHomeEnvironment(); gen = VirtualHomeEpisodeGenerator(env, seed=42); episode = gen.generate_episode(); print(f'Steps: {len(episode.steps)}')"`
2. Check: Episode generated without errors
3. Verify: Episode has correct structure

**Expected:** Episode generated (if VirtualHome installed) or clear error (if not)

**Result:** [ ] Pass  [ ] Fail  [ ] Partial  [ ] Skip

---

## Plan 03-03: Recorder & Observability

### Test 8: VirtualHomeEpisodeRecorder Import
**What to test:** Recorder can be imported

**Steps:**
1. Run: `python -c "from src.bsa.envs.virtualhome import VirtualHomeEpisodeRecorder; print('OK')"`
2. Check: Import succeeds

**Expected:** Import works

**Result:** [ ] Pass  [ ] Fail  [ ] Partial  [ ] Skip

---

### Test 9: Episode Serialization (if episode available)
**What to test:** Episodes can be saved to Parquet/JSONL

**Steps:**
1. If episode available, test saving to Parquet
2. If episode available, test saving to JSONL
3. Verify: Files created successfully

**Expected:** Serialization works (if episode available)

**Result:** [ ] Pass  [ ] Fail  [ ] Partial  [ ] Skip

---

### Test 10: Observability Module
**What to test:** Observability functions work

**Steps:**
1. Run: `python -c "from src.bsa.envs.virtualhome import get_scene_state, get_agent_view; print('OK')"`
2. Check: Functions import successfully
3. If VirtualHome available, test functions with environment

**Expected:** Observability functions import and work

**Result:** [ ] Pass  [ ] Fail  [ ] Partial  [ ] Skip

---

## Plan 03-04: Tests & Verification

### Test 11: Unit Tests Structure
**What to test:** Test files exist and are structured correctly

**Steps:**
1. Check: `tests/test_virtualhome.py` exists
2. Check: `tests/test_virtualhome_integration.py` exists
3. Verify: Tests handle missing VirtualHome gracefully

**Expected:** Test files exist, handle VirtualHome gracefully

**Result:** [ ] Pass  [ ] Fail  [ ] Partial  [ ] Skip

---

### Test 12: End-to-End Verification Script
**What to test:** Verification script runs

**Steps:**
1. Run: `python scripts/verify_virtualhome.py`
2. Check: Script runs without crashes
3. Verify: Provides clear pass/fail output

**Expected:** Script runs, provides clear results

**Result:** [ ] Pass  [ ] Fail  [ ] Partial  [ ] Skip

---

### Test 13: Integration with Helper Agents
**What to test:** Helper agents work with VirtualHome (if available)

**Steps:**
1. If VirtualHome available, test helper agents import and work
2. Verify: No import errors

**Expected:** Helper agents compatible with VirtualHome

**Result:** [ ] Pass  [ ] Fail  [ ] Partial  [ ] Skip

---

## Summary

**Tests Completed:** ___ / 13  
**Passed:** ___  
**Failed:** ___  
**Partial:** ___  
**Skipped:** ___

## Issues Found

[List any issues here]

## Verdict

[ ] All tests passed - Phase 3 validated  
[ ] Minor issues logged - Phase 3 works with issues  
[ ] Major issues found - Review before proceeding  
[ ] Blocking issues found - Must fix before continuing
