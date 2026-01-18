#!/usr/bin/env python3
"""Final comprehensive verification of repository."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 70)
print("Final Comprehensive Verification")
print("=" * 70)

all_ok = True

# 1. Data loading
print("\n[1] Testing data loading...")
try:
    from bsa.analysis.aggregate import AnalysisAggregator
    agg = AnalysisAggregator()
    df = agg.load_results(input_path=Path("results/metrics/large_scale_research/results.parquet"))
    print(f"  [OK] Data loads: {len(df)} rows, {len(df.columns)} columns")
except Exception as e:
    print(f"  [FAIL] Data loading error: {e}")
    all_ok = False

# 2. Figure generation
print("\n[2] Testing figure generation...")
try:
    from bsa.viz.plots import PlotGenerator
    from bsa.analysis.aggregate import AnalysisAggregator
    agg = AnalysisAggregator()
    df = agg.load_results(input_path=Path("results/metrics/large_scale_research/results.parquet"))
    agg_df = agg.aggregate_metrics(df)
    plotter = PlotGenerator(agg_df, Path("results/figures"), raw_df=df)
    print(f"  [OK] Plot generator initialized")
except Exception as e:
    print(f"  [FAIL] Figure generation error: {e}")
    all_ok = False

# 3. File structure
print("\n[3] Verifying file structure...")
checks = [
    ("Episodes", len(list(Path("data/episodes/large_scale").glob("*.parquet"))), 9000, 10000),
    ("Figures", len(list(Path("results/figures").glob("*.png"))), 15, 20),
    ("Tables MD", len(list(Path("results/tables").glob("*.md"))), 4, 8),
    ("Tables TEX", len(list(Path("results/tables").glob("*.tex"))), 4, 8),
    ("Source files", len(list(Path("src").rglob("*.py"))), 40, 50),
    ("Scripts", len(list(Path("scripts").glob("*.py"))), 8, 15),
]

for name, count, min_val, max_val in checks:
    if min_val <= count <= max_val:
        print(f"  [OK] {name}: {count}")
    else:
        print(f"  [WARN] {name}: {count} (expected {min_val}-{max_val})")

# 4. Paper
print("\n[4] Verifying paper...")
paper = Path("paper/research_paper.md")
if paper.exists():
    word_count = len(paper.read_text().split())
    print(f"  [OK] Paper exists: {word_count:,} words")
    if word_count < 5000:
        print(f"  [WARN] Paper seems short ({word_count} words)")
else:
    print(f"  [FAIL] Paper missing")
    all_ok = False

# Summary
print("\n" + "=" * 70)
if all_ok:
    print("VERIFICATION PASSED")
    print("=" * 70)
    print("Repository is clean and ready for research/publication.")
else:
    print("VERIFICATION FAILED")
    print("=" * 70)
    print("Some issues found. Please review above.")
