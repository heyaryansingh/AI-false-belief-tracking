#!/usr/bin/env python3
"""Generate METHODOLOGY_CHANGES.md from code comments.

# Fix: Auto-generate change log from code comments (Phase 10)

This script parses Python files for "# Fix:" comments and generates
a structured Markdown document summarizing all methodology changes.
"""

import sys
from pathlib import Path
import re
from datetime import datetime
from typing import List, Dict, Any
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def find_fix_comments(directory: Path) -> List[Dict[str, Any]]:
    """Find all "# Fix:" comments in Python files.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of dictionaries with file, line, and comment info
    """
    fixes = []
    
    for py_file in directory.rglob("*.py"):
        # Skip venv and __pycache__
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue
        
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")
            continue
        
        for i, line in enumerate(lines):
            if "# Fix:" in line:
                # Extract the fix description
                match = re.search(r"#\s*Fix:\s*(.+?)(?:\s*\(Phase\s*(\d+)\))?$", line.strip())
                if match:
                    description = match.group(1).strip()
                    phase = match.group(2) if match.group(2) else "Unknown"
                    
                    fixes.append({
                        "file": str(py_file.relative_to(directory)),
                        "line": i + 1,
                        "description": description,
                        "phase": phase,
                        "full_line": line.strip(),
                    })
    
    return fixes


def categorize_fixes(fixes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize fixes by topic.
    
    Args:
        fixes: List of fix dictionaries
        
    Returns:
        Dictionary mapping category to list of fixes
    """
    categories = {
        "AUROC Stabilization": [],
        "Efficiency Fix": [],
        "Temporal Metrics": [],
        "Statistical Reporting": [],
        "Experimental Design": [],
        "Visualization": [],
        "Other": [],
    }
    
    keyword_map = {
        "AUROC Stabilization": ["auroc", "bootstrap", "ci", "confidence"],
        "Efficiency Fix": ["efficiency", "wasted", "per-model", "per-episode"],
        "Temporal Metrics": ["temporal", "detection", "latency", "ttd", "time-to-detection"],
        "Statistical Reporting": ["statistics", "effect size", "p-value", "significance", "pairwise"],
        "Experimental Design": ["condition", "partial", "drift", "seed"],
        "Visualization": ["plot", "figure", "roc", "violin", "heatmap"],
    }
    
    for fix in fixes:
        desc_lower = fix["description"].lower()
        categorized = False
        
        for category, keywords in keyword_map.items():
            if any(kw in desc_lower for kw in keywords):
                categories[category].append(fix)
                categorized = True
                break
        
        if not categorized:
            categories["Other"].append(fix)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def generate_markdown(
    fixes: List[Dict[str, Any]],
    categorized: Dict[str, List[Dict[str, Any]]],
) -> str:
    """Generate METHODOLOGY_CHANGES.md content.
    
    Args:
        fixes: All fixes
        categorized: Fixes by category
        
    Returns:
        Markdown content
    """
    lines = [
        "# Methodology Changes Documentation",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Phase**: 10 - Statistical Strengthening",
        f"**Total Fixes**: {len(fixes)}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "This document summarizes all methodology changes made during Phase 10 to strengthen",
        "the statistical validity and scientific rigor of the experimental pipeline.",
        "",
        "### Key Improvements",
        "",
        "1. **Bootstrap Confidence Intervals**: Replaced mean ± SD with mean [95% CI] for AUROC",
        "2. **Effect Size Calculations**: Added Cohen's d for pairwise model comparisons",
        "3. **Temporal Metrics**: Added time-to-detection and false alarm rate tracking",
        "4. **Three Conditions**: Added partial_false_belief (drift_probability=0.5)",
        "5. **Visualization**: Added ROC curves, violin plots, and diagnostic figures",
        "6. **Per-Episode Independence**: Ensured efficiency is computed independently per model/episode",
        "",
        "---",
        "",
    ]
    
    # Add categorized sections
    for category, cat_fixes in categorized.items():
        lines.append(f"## {category}")
        lines.append("")
        
        for fix in cat_fixes:
            lines.append(f"### {fix['description']}")
            lines.append("")
            lines.append(f"**Location**: `{fix['file']}` (line {fix['line']})")
            lines.append("")
            lines.append(f"**Phase**: {fix['phase']}")
            lines.append("")
            
            # Add rationale based on category
            rationale = get_rationale(category, fix['description'])
            if rationale:
                lines.append(f"**Rationale**: {rationale}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
    
    # Add file listing
    lines.append("## Files Modified")
    lines.append("")
    
    files_modified = set(fix["file"] for fix in fixes)
    for file in sorted(files_modified):
        file_fixes = [f for f in fixes if f["file"] == file]
        lines.append(f"- `{file}` ({len(file_fixes)} changes)")
    
    lines.append("")
    
    # Add appendix with all comments
    lines.append("## Appendix: All Fix Comments")
    lines.append("")
    lines.append("| File | Line | Description |")
    lines.append("|------|------|-------------|")
    
    for fix in sorted(fixes, key=lambda x: (x["file"], x["line"])):
        desc = fix["description"][:60] + "..." if len(fix["description"]) > 60 else fix["description"]
        lines.append(f"| `{fix['file']}` | {fix['line']} | {desc} |")
    
    lines.append("")
    
    return "\n".join(lines)


def get_rationale(category: str, description: str) -> str:
    """Get rationale explanation for a fix category.
    
    Args:
        category: Fix category
        description: Fix description
        
    Returns:
        Rationale explanation
    """
    rationales = {
        "AUROC Stabilization": (
            "High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. "
            "Bootstrap CI provides robust confidence bounds that are more reliable than "
            "simple standard deviation for classification metrics."
        ),
        "Efficiency Fix": (
            "Identical efficiency across models (0.815) suggested shared calculation or "
            "caching. Ensuring per-model/episode independence provides valid comparative metrics."
        ),
        "Temporal Metrics": (
            "AUROC treats detection as static classification, ignoring temporal dynamics. "
            "Adding detection latency and time-to-detection captures realistic detection timing."
        ),
        "Statistical Reporting": (
            "Raw means without inferential statistics limit scientific validity. "
            "Effect sizes and significance tests enable proper hypothesis testing."
        ),
        "Experimental Design": (
            "Two-condition design (control, false_belief) lacks intermediate states. "
            "Adding partial_false_belief enables more nuanced analysis of model behavior."
        ),
        "Visualization": (
            "No visual diagnostics for belief inference or particle stability. "
            "Diagnostic plots improve interpretability and support publication."
        ),
    }
    
    return rationales.get(category, "")


def main():
    parser = argparse.ArgumentParser(description="Generate METHODOLOGY_CHANGES.md")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("METHODOLOGY_CHANGES.md"),
        help="Output file path"
    )
    parser.add_argument(
        "--source-dir", "-d",
        type=Path,
        default=Path("."),
        help="Source directory to search"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generating Methodology Changes Documentation")
    print("=" * 60)
    
    # Find all fix comments
    print(f"\nSearching for '# Fix:' comments in {args.source_dir}...")
    fixes = find_fix_comments(args.source_dir)
    print(f"Found {len(fixes)} fix comments")
    
    # Categorize fixes
    print("\nCategorizing fixes...")
    categorized = categorize_fixes(fixes)
    for cat, cat_fixes in categorized.items():
        print(f"  {cat}: {len(cat_fixes)} fixes")
    
    # Generate markdown
    print("\nGenerating Markdown...")
    content = generate_markdown(fixes, categorized)
    
    # Write output
    args.output.write_text(content, encoding="utf-8")
    print(f"\nSaved to: {args.output}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
