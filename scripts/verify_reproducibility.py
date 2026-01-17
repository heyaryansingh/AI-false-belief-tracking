"""Reproducibility verification script."""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bsa.experiments.manifest import verify_reproducibility


def main() -> int:
    """Main function for reproducibility verification."""
    parser = argparse.ArgumentParser(
        description="Verify reproducibility of an experiment"
    )
    parser.add_argument(
        "manifest_path",
        type=Path,
        help="Path to manifest file to verify",
    )
    
    args = parser.parse_args()
    
    if not args.manifest_path.exists():
        print(f"Error: Manifest file not found: {args.manifest_path}")
        return 1
    
    print("=" * 70)
    print("Reproducibility Verification".center(70))
    print("=" * 70)
    print()
    
    # Verify reproducibility
    report = verify_reproducibility(args.manifest_path)
    
    # Print report
    print(f"Manifest: {report['manifest_path']}")
    print()
    
    print("Comparisons:")
    for key, comparison in report["comparisons"].items():
        match_symbol = "✓" if comparison["match"] else "✗"
        print(f"  {match_symbol} {key}:")
        print(f"    Manifest: {comparison['manifest']}")
        print(f"    Current:  {comparison['current']}")
        print()
    
    print("Package Differences:")
    pkg_diff = report["package_differences"]
    if pkg_diff["added"]:
        print(f"  Added packages: {pkg_diff['added']}")
    if pkg_diff["removed"]:
        print(f"  Removed packages: {pkg_diff['removed']}")
    if pkg_diff["version_changed"]:
        print(f"  Version changes:")
        for change in pkg_diff["version_changed"]:
            print(
                f"    {change['package']}: "
                f"{change['manifest_version']} -> {change['current_version']}"
            )
    if not any([pkg_diff["added"], pkg_diff["removed"], pkg_diff["version_changed"]]):
        print("  No differences")
    print()
    
    # Overall status
    if report["reproducible"]:
        print("✓ Environment matches manifest - experiment is reproducible")
        return 0
    else:
        print("✗ Environment differs from manifest - experiment may not be reproducible")
        print("\nRecommendations:")
        print("  - Check git commit hash matches")
        print("  - Verify Python version matches")
        print("  - Ensure package versions match (check requirements.txt)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
