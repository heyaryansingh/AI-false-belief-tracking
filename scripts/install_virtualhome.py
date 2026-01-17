#!/usr/bin/env python3
"""Installation script for VirtualHome simulator.

This script checks for VirtualHome installation and attempts to install it if missing.
VirtualHome is an optional dependency - if installation fails, GridHouse fallback is used.
"""

import os
import subprocess
import sys
from typing import Optional


def check_virtualhome_installed() -> bool:
    """Check if VirtualHome is already installed.

    Returns:
        True if VirtualHome can be imported, False otherwise
    """
    try:
        import virtualhome
        return True
    except ImportError:
        return False


def get_virtualhome_version() -> Optional[str]:
    """Get VirtualHome version if installed.

    Returns:
        Version string if installed, None otherwise
    """
    try:
        import virtualhome
        if hasattr(virtualhome, "__version__"):
            return virtualhome.__version__
        return "installed (version unknown)"
    except ImportError:
        return None


def install_virtualhome() -> bool:
    """Attempt to install VirtualHome via pip.

    Returns:
        True if installation succeeded, False otherwise
    """
    print("Installing VirtualHome...")
    try:
        # Try installing from PyPI
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "virtualhome>=2.3.0"],
            capture_output=True,
            text=True,
            check=True,
        )
        print("VirtualHome installed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install VirtualHome from PyPI:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error during installation: {e}")
        return False


def verify_installation() -> bool:
    """Verify VirtualHome installation works.

    Returns:
        True if verification succeeds, False otherwise
    """
    try:
        import virtualhome
        # Try basic import and check for key modules
        # VirtualHome has different modes - check for evolving graph simulator
        try:
            from virtualhome.simulation import evolvinggraph
            print("✓ VirtualHome Evolving Graph simulator available")
        except ImportError:
            print("⚠ VirtualHome installed but Evolving Graph module not found")
            print("  This may be okay if using Unity mode instead")
        
        return True
    except ImportError as e:
        print(f"✗ VirtualHome import failed: {e}")
        return False


def main():
    """Main installation routine."""
    print("=" * 70)
    print("VirtualHome Installation Check")
    print("=" * 70)
    print()

    # Check if already installed
    if check_virtualhome_installed():
        version = get_virtualhome_version()
        print(f"✓ VirtualHome is already installed ({version})")
        
        if verify_installation():
            print("\nVirtualHome is ready to use!")
            return 0
        else:
            print("\n⚠ Installation detected but verification failed.")
            print("  You may need to reinstall VirtualHome.")
            return 1

    # Not installed - attempt installation
    print("VirtualHome is not installed.")
    print()
    
    # Check if running in non-interactive mode
    non_interactive = not sys.stdin.isatty() or os.getenv("NON_INTERACTIVE", "").lower() == "true"
    
    if non_interactive:
        print("Running in non-interactive mode - skipping installation.")
        print("Note: GridHouse fallback simulator will be used instead.")
        print("To install VirtualHome, run: pip install virtualhome>=2.3.0")
        return 0
    
    # Try to get user input, handle EOFError for non-interactive environments
    try:
        response = input("Would you like to install VirtualHome now? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nRunning in non-interactive mode - skipping installation.")
        print("Note: GridHouse fallback simulator will be used instead.")
        return 0
    
    if response not in ("y", "yes"):
        print("\nInstallation skipped.")
        print("Note: GridHouse fallback simulator will be used instead.")
        return 0

    # Attempt installation
    if install_virtualhome():
        if verify_installation():
            print("\n✓ VirtualHome installation verified!")
            print("\nNote: For visual rendering, you may need to download the Unity simulator separately.")
            print("      See: https://github.com/xavierpuigf/virtualhome")
            return 0
        else:
            print("\n⚠ Installation completed but verification failed.")
            print("  VirtualHome may not work correctly.")
            return 1
    else:
        print("\n✗ Installation failed.")
        print("\nNote: GridHouse fallback simulator will be used instead.")
        print("      This is fine for research purposes - VirtualHome is optional.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
