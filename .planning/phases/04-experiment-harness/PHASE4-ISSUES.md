# Phase 4: Experiment Harness + Reproducibility - Issues Log

## Issue 1: Unicode Encoding Error in reproduce() Function

**Severity**: Medium  
**Status**: Fixed  
**Discovered**: During UAT testing  
**Fixed**: 2025-01-17

### Description
The `reproduce()` function used Unicode checkmark characters (✓ and ✗) in print statements, which cause `UnicodeEncodeError` on Windows systems using cp1252 encoding.

### Error Message
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 2: character maps to <undefined>
```

### Root Cause
Windows console uses cp1252 encoding by default, which doesn't support Unicode checkmark characters.

### Fix Applied
Replaced Unicode characters with ASCII equivalents:
- `✓` → `[OK]`
- `✗` → `[FAIL]`

### Files Modified
- `src/bsa/experiments/run_experiment.py`

### Prevention
- Use ASCII-compatible characters in all print statements
- Consider using a logging library that handles encoding automatically
- Test on Windows systems during development

---

## Issue 2: Full Experiment Runs Not Tested

**Severity**: Low  
**Status**: Deferred  
**Reason**: Time constraints - full experiment runs take significant time

### Description
Full experiment execution with multiple models and conditions was not tested during UAT due to time constraints. Small dataset mode was tested instead.

### Recommendation
- Test full experiment runs in CI/CD pipeline
- Add integration tests that run with minimal configurations
- Document expected runtime for different experiment sizes

---

## Issue 3: Full Sweep Execution Not Tested

**Severity**: Low  
**Status**: Deferred  
**Reason**: Time constraints - parameter sweeps take significant time

### Description
Full parameter sweep execution was not tested during UAT due to time constraints. Configuration files were validated instead.

### Recommendation
- Test sweep execution in CI/CD pipeline with minimal parameter sets
- Add unit tests for sweep configuration parsing
- Document expected runtime for different sweep sizes
