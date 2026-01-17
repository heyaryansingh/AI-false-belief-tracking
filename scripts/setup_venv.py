#!/usr/bin/env python3
"""Setup virtual environment for the project.

Creates a virtual environment with compatible Python version and installs dependencies.
Handles VirtualHome compatibility issues by using Python 3.9-3.10 if available.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def find_python_version(required_versions: list[str]) -> Optional[str]:
    """Find available Python version from required list.

    Args:
        required_versions: List of Python versions to check (e.g., ['3.9', '3.10', '3.11'])

    Returns:
        Path to Python executable if found, None otherwise
    """
    for version in required_versions:
        # Try python3.9, python3.10, etc.
        for cmd in [f"python{version}", f"py -{version}"]:
            try:
                result = subprocess.run(
                    [cmd, "--version"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if version in result.stdout:
                    return cmd
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
    return None


def create_venv(venv_path: Path, python_cmd: Optional[str] = None) -> bool:
    """Create virtual environment.

    Args:
        venv_path: Path where venv should be created
        python_cmd: Python command to use (if None, uses current Python)

    Returns:
        True if successful, False otherwise
    """
    if venv_path.exists():
        print(f"Virtual environment already exists at {venv_path}")
        return True

    print(f"Creating virtual environment at {venv_path}...")
    cmd = [sys.executable, "-m", "venv", str(venv_path)]
    if python_cmd:
        cmd = [python_cmd, "-m", "venv", str(venv_path)]

    try:
        subprocess.run(cmd, check=True)
        print("[OK] Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Failed to create virtual environment: {e}")
        return False


def get_venv_python(venv_path: Path) -> Path:
    """Get Python executable path in venv.

    Args:
        venv_path: Path to venv directory

    Returns:
        Path to Python executable
    """
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def install_dependencies(venv_python: Path, include_virtualhome: bool = True) -> bool:
    """Install project dependencies in venv.

    Args:
        venv_python: Path to Python executable in venv
        include_virtualhome: Whether to install VirtualHome (optional dependency)

    Returns:
        True if successful, False otherwise
    """
    print("Installing project dependencies...")
    project_root = Path(__file__).parent.parent
    
    try:
        # Upgrade pip first
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
        )
        
        # Install base requirements (with NumPy <2.0 constraint)
        requirements_file = project_root / "requirements.txt"
        if requirements_file.exists():
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "-r", str(requirements_file)],
                check=True,
            )
        else:
            # Fallback: install project in editable mode
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "-e", str(project_root)],
                check=True,
            )
        
        # Optionally install VirtualHome
        if include_virtualhome:
            print("Installing VirtualHome (optional)...")
            try:
                # Install VirtualHome with compatible dependencies
                vh_requirements = project_root / "requirements-virtualhome.txt"
                if vh_requirements.exists():
                    subprocess.run(
                        [str(venv_python), "-m", "pip", "install", "-r", str(vh_requirements)],
                        check=True,
                    )
                else:
                    # Fallback: install VirtualHome without deps, then install compatible versions
                    subprocess.run(
                        [str(venv_python), "-m", "pip", "install", "virtualhome", "--no-deps"],
                        check=True,
                    )
                    subprocess.run(
                        [
                            str(venv_python),
                            "-m",
                            "pip",
                            "install",
                            "numpy>=1.19.3,<2.0.0",
                            "networkx>=2.3",
                        ],
                        check=True,
                    )
                
                # Verify VirtualHome import works
                result = subprocess.run(
                    [str(venv_python), "-c", "import virtualhome; print('OK')"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print("[OK] VirtualHome installed and verified")
                else:
                    print("[WARN] VirtualHome installed but import failed")
                    print(f"  Error: {result.stderr}")
            except subprocess.CalledProcessError as e:
                print("[WARN] VirtualHome installation failed (will use GridHouse fallback)")
                print(f"  Error: {e}")
        
        print("[OK] Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Failed to install dependencies: {e}")
        return False


def main():
    """Main setup routine."""
    print("=" * 70)
    print("Virtual Environment Setup")
    print("=" * 70)
    print()

    # Determine venv location
    project_root = Path(__file__).parent.parent
    venv_path = project_root / "venv"
    
    # Check if venv already exists
    if venv_path.exists():
        print(f"Virtual environment exists at {venv_path}")
        # Check if running in non-interactive mode
        non_interactive = not sys.stdin.isatty() or os.getenv("NON_INTERACTIVE", "").lower() == "true"
        
        if non_interactive:
            print("Using existing virtual environment (non-interactive mode)")
        else:
            try:
                response = input("Recreate? (y/n): ").strip().lower()
                if response == "y":
                    import shutil
                    shutil.rmtree(venv_path)
                else:
                    print("Using existing virtual environment")
            except (EOFError, KeyboardInterrupt):
                print("Using existing virtual environment (non-interactive mode)")
        
        venv_python = get_venv_python(venv_path)
        if venv_python.exists():
            print(f"Python: {venv_python}")
            print("\nTo activate:")
            if sys.platform == "win32":
                print(f"  {venv_path}\\Scripts\\activate")
            else:
                print(f"  source {venv_path}/bin/activate")
            return 0

    # Check Python version compatibility
    # VirtualHome works best with Python 3.9-3.10, but we'll try current Python first
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Current Python version: {current_version}")

    # Try to find Python 3.9 or 3.10 for better VirtualHome compatibility
    preferred_python = None
    if current_version not in ["3.9", "3.10"]:
        print("Checking for Python 3.9 or 3.10 (better VirtualHome compatibility)...")
        preferred_python = find_python_version(["3.9", "3.10"])
        if preferred_python:
            print(f"Found: {preferred_python}")
        else:
            print("Using current Python (VirtualHome may have dependency issues)")

    # Create venv
    python_cmd = preferred_python if preferred_python else None
    if not create_venv(venv_path, python_cmd):
        return 1

    # Get venv Python
    venv_python = get_venv_python(venv_path)
    if not venv_python.exists():
        print(f"[FAIL] Python executable not found at {venv_python}")
        return 1

    # Install dependencies
    include_virtualhome = True
    if "--no-virtualhome" in sys.argv:
        include_virtualhome = False

    if not install_dependencies(venv_python, include_virtualhome):
        return 1

    # Success
    print()
    print("=" * 70)
    print("Setup Complete!")
    print("=" * 70)
    print()
    print("To activate the virtual environment:")
    if sys.platform == "win32":
        print(f"  {venv_path}\\Scripts\\activate")
    else:
        print(f"  source {venv_path}/bin/activate")
    print()
    print("To verify installation:")
    print(f"  {venv_python} -c \"from src.bsa.envs.gridhouse import GridHouseEnvironment; print('OK')\"")
    if include_virtualhome:
        print(f"  {venv_python} -c \"from src.bsa.envs.virtualhome import VirtualHomeEnvironment; print('OK')\"")

    return 0


if __name__ == "__main__":
    sys.exit(main())
