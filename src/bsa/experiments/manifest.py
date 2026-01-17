"""Manifest generation for reproducibility tracking."""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import subprocess
import sys
import platform
import hashlib

from ..common.config import load_config


def generate_manifest(
    experiment_config: Dict[str, Any],
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate experiment manifest with reproducibility information.

    Args:
        experiment_config: Experiment configuration dictionary
        config_path: Path to experiment config file (for computing hash)

    Returns:
        Dictionary with manifest information
    """
    manifest = {
        "experiment_name": experiment_config.get("name", "experiment"),
        "timestamp": _get_timestamp(),
        "git_info": _get_git_info(),
        "config_hash": _compute_config_hash(config_path) if config_path else None,
        "python_version": sys.version,
        "system_info": _get_system_info(),
        "package_versions": _get_package_versions(),
        "experiment_config": experiment_config,
    }
    
    return manifest


def save_manifest(manifest: Dict[str, Any], output_path: Path) -> None:
    """Save manifest to JSON file.

    Args:
        manifest: Manifest dictionary
        output_path: Path to save manifest file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def verify_reproducibility(manifest_path: Path) -> Dict[str, Any]:
    """Verify current environment matches manifest.

    Args:
        manifest_path: Path to manifest file

    Returns:
        Dictionary with verification report
    """
    # Load manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    # Compare current environment to manifest
    current_git = _get_git_info()
    current_python = sys.version
    current_system = _get_system_info()
    current_packages = _get_package_versions()
    
    report = {
        "manifest_path": str(manifest_path),
        "comparisons": {
            "git_hash": {
                "manifest": manifest.get("git_info", {}).get("commit_hash"),
                "current": current_git.get("commit_hash"),
                "match": manifest.get("git_info", {}).get("commit_hash") == current_git.get("commit_hash"),
            },
            "git_status": {
                "manifest": manifest.get("git_info", {}).get("status"),
                "current": current_git.get("status"),
                "match": manifest.get("git_info", {}).get("status") == current_git.get("status"),
            },
            "python_version": {
                "manifest": manifest.get("python_version"),
                "current": current_python,
                "match": manifest.get("python_version") == current_python,
            },
            "system": {
                "manifest": manifest.get("system_info"),
                "current": current_system,
                "match": manifest.get("system_info") == current_system,
            },
        },
        "package_differences": _compare_packages(
            manifest.get("package_versions", {}),
            current_packages,
        ),
    }
    
    # Overall match status
    all_match = (
        report["comparisons"]["git_hash"]["match"]
        and report["comparisons"]["python_version"]["match"]
        and len(report["package_differences"]["added"]) == 0
        and len(report["package_differences"]["removed"]) == 0
        and len(report["package_differences"]["version_changed"]) == 0
    )
    report["reproducible"] = all_match
    
    return report


def _get_timestamp() -> str:
    """Get current timestamp."""
    from datetime import datetime
    return datetime.now().isoformat()


def _get_git_info() -> Dict[str, Any]:
    """Get git information (commit hash, branch, status).

    Returns:
        Dictionary with git information
    """
    git_info = {
        "commit_hash": None,
        "branch": None,
        "status": None,
    }
    
    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            git_info["commit_hash"] = result.stdout.strip()
        
        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()
        
        # Get git status (clean/dirty)
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            git_info["status"] = "clean" if not result.stdout.strip() else "dirty"
    except (FileNotFoundError, subprocess.SubprocessError):
        # Git not available
        pass
    
    return git_info


def _compute_config_hash(config_path: Path) -> Optional[str]:
    """Compute SHA256 hash of config file.

    Args:
        config_path: Path to config file

    Returns:
        SHA256 hash string or None if file not found
    """
    if not config_path or not config_path.exists():
        return None
    
    with open(config_path, "rb") as f:
        content = f.read()
    
    return hashlib.sha256(content).hexdigest()


def _get_system_info() -> Dict[str, Any]:
    """Get system information.

    Returns:
        Dictionary with system information
    """
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _get_package_versions() -> Dict[str, str]:
    """Get installed package versions.

    Returns:
        Dictionary mapping package name to version
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        
        packages = {}
        for line in result.stdout.strip().split("\n"):
            if "==" in line:
                name, version = line.split("==", 1)
                packages[name] = version
        
        return packages
    except (subprocess.SubprocessError, FileNotFoundError):
        # pip not available or freeze failed
        return {}


def _compare_packages(
    manifest_packages: Dict[str, str],
    current_packages: Dict[str, str],
) -> Dict[str, Any]:
    """Compare package versions between manifest and current.

    Args:
        manifest_packages: Package versions from manifest
        current_packages: Current package versions

    Returns:
        Dictionary with differences
    """
    manifest_set = set(manifest_packages.keys())
    current_set = set(current_packages.keys())
    
    added = list(current_set - manifest_set)
    removed = list(manifest_set - current_set)
    
    version_changed = []
    for pkg in manifest_set & current_set:
        if manifest_packages[pkg] != current_packages[pkg]:
            version_changed.append({
                "package": pkg,
                "manifest_version": manifest_packages[pkg],
                "current_version": current_packages[pkg],
            })
    
    return {
        "added": added,
        "removed": removed,
        "version_changed": version_changed,
    }
