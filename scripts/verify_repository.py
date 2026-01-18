#!/usr/bin/env python3
"""Verify repository structure and essential files."""

from pathlib import Path
import sys

def verify():
    """Verify repository structure."""
    print("=" * 70)
    print("Repository Verification")
    print("=" * 70)
    
    issues = []
    
    # Check essential directories
    print("\n[1] Checking essential directories...")
    essential_dirs = [
        "src",
        "configs",
        "scripts",
        "tests",
        "data",
        "results",
        "paper",
    ]
    for d in essential_dirs:
        p = Path(d)
        if p.exists():
            print(f"  [OK] {d}/")
        else:
            print(f"  [FAIL] {d}/ MISSING")
            issues.append(f"Missing directory: {d}/")
    
    # Check essential files
    print("\n[2] Checking essential files...")
    essential_files = [
        "README.md",
        "pyproject.toml",
        "Makefile",
        "paper/research_paper.md",
    ]
    for f in essential_files:
        p = Path(f)
        if p.exists():
            print(f"  [OK] {f}")
        else:
            print(f"  [FAIL] {f} MISSING")
            issues.append(f"Missing file: {f}")
    
    # Check data
    print("\n[3] Checking data files...")
    episodes = list(Path("data/episodes/large_scale").glob("*.parquet"))
    print(f"  Episodes: {len(episodes):,} files")
    if len(episodes) == 0:
        issues.append("No episode files found")
    
    results = Path("results/metrics/large_scale_research/results.parquet")
    if results.exists():
        print(f"  Results: EXISTS")
    else:
        print(f"  Results: MISSING")
        issues.append("Results file missing")
    
    # Check figures
    print("\n[4] Checking figures...")
    figures = list(Path("results/figures").glob("*.png"))
    print(f"  Figures: {len(figures)} PNG files")
    if len(figures) == 0:
        issues.append("No figure files found")
    else:
        # Check for non-PNG files
        non_pngs = [f for f in Path("results/figures").iterdir() if f.is_file() and not f.name.endswith('.png')]
        if non_pngs:
            print(f"  Warning: {len(non_pngs)} non-PNG files in figures directory")
    
    # Check tables
    print("\n[5] Checking tables...")
    tables_md = list(Path("results/tables").glob("*.md"))
    tables_tex = list(Path("results/tables").glob("*.tex"))
    print(f"  Tables: {len(tables_md)} Markdown + {len(tables_tex)} LaTeX")
    if len(tables_md) == 0:
        issues.append("No table files found")
    
    # Check source code
    print("\n[6] Checking source code...")
    python_files = list(Path("src").rglob("*.py"))
    print(f"  Python files: {len(python_files)}")
    if len(python_files) == 0:
        issues.append("No Python source files found")
    
    # Check scripts
    print("\n[7] Checking scripts...")
    scripts = list(Path("scripts").glob("*.py"))
    print(f"  Scripts: {len(scripts)}")
    essential_scripts = [
        "run_large_experiments.py",
        "generate_all_figures.py",
        "regenerate_comprehensive_analysis.py",
    ]
    for s in essential_scripts:
        if Path(f"scripts/{s}").exists():
            print(f"    [OK] {s}")
        else:
            print(f"    [FAIL] {s} MISSING")
            issues.append(f"Missing script: {s}")
    
    # Summary
    print("\n" + "=" * 70)
    if issues:
        print("VERIFICATION FAILED")
        print("=" * 70)
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    else:
        print("VERIFICATION PASSED")
        print("=" * 70)
        print("All essential files and directories present.")
        return 0

if __name__ == "__main__":
    sys.exit(verify())
