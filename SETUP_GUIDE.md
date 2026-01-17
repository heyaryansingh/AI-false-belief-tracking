# Setup Guide: Virtual Environment and VirtualHome Installation

## Overview

This project uses a virtual environment to manage dependencies and ensure VirtualHome compatibility. VirtualHome requires Python 3.9-3.11 and NumPy <2.0, which conflicts with newer Python versions and NumPy 2.x.

## Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Create virtual environment and install dependencies
python scripts/setup_venv.py

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Verify installation
python -c "from src.bsa.envs.gridhouse import GridHouseEnvironment; print('GridHouse OK')"
python -c "from src.bsa.envs.virtualhome import VirtualHomeEnvironment; print('VirtualHome OK')"  # Optional
```

### Option 2: Manual Setup

```bash
# Create virtual environment (Python 3.9-3.11 recommended)
python3.9 -m venv venv  # or python3.10, python3.11

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Optionally install VirtualHome
pip install -r requirements-virtualhome.txt
```

## Python Version Requirements

- **Recommended**: Python 3.9 or 3.10 (best VirtualHome compatibility)
- **Minimum**: Python 3.9
- **Maximum**: Python 3.11 (Python 3.12 has NumPy conflicts)

The setup script will automatically detect and use Python 3.9 or 3.10 if available.

## Dependency Constraints

### NumPy Version

- **Constraint**: NumPy <2.0.0
- **Reason**: VirtualHome 2.3.0 was compiled with NumPy 1.x and is incompatible with NumPy 2.x
- **Impact**: All dependencies must be compatible with NumPy 1.x

### VirtualHome Dependencies

VirtualHome requires specific versions:
- `virtualhome>=2.3.0`
- `numpy>=1.19.3,<2.0.0`
- `networkx>=2.3`

These are installed separately via `requirements-virtualhome.txt` to avoid conflicts.

## Virtual Environment Structure

```
venv/
├── bin/          # Linux/Mac executables
│   └── python
├── Scripts/      # Windows executables
│   └── python.exe
└── lib/          # Installed packages
```

## Troubleshooting

### VirtualHome Import Errors

If you get NumPy compatibility errors:

```bash
# Check NumPy version
python -c "import numpy; print(numpy.__version__)"

# Should be <2.0.0. If not:
pip install "numpy>=1.19.3,<2.0.0"
```

### Python Version Issues

If setup script can't find Python 3.9-3.11:

1. Install Python 3.9 or 3.10
2. Use it explicitly: `python3.9 scripts/setup_venv.py`
3. Or create venv manually: `python3.9 -m venv venv`

### VirtualHome Installation Fails

VirtualHome is optional - GridHouse fallback will be used automatically:

```bash
# Skip VirtualHome installation
python scripts/setup_venv.py --no-virtualhome

# Or install manually later
pip install virtualhome --no-deps
pip install "numpy>=1.19.3,<2.0.0" networkx>=2.3
```

## Using the Virtual Environment

### Activation

**Windows:**
```powershell
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Deactivation

```bash
deactivate
```

### Running Commands

Always activate the venv before running project commands:

```bash
# Activate venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run commands
python scripts/install_virtualhome.py
make reproduce
pytest tests/
```

## Integration with Makefile

The Makefile includes venv-aware targets:

```bash
make venv          # Create virtual environment
make setup         # Setup venv and install dependencies
make install-venv  # Install package in venv
```

## CI/CD Considerations

For automated environments, use non-interactive mode:

```bash
NON_INTERACTIVE=true python scripts/setup_venv.py
```

Or skip VirtualHome if not needed:

```bash
python scripts/setup_venv.py --no-virtualhome
```

## Next Steps

After setup:

1. **Verify GridHouse works**: `python -c "from src.bsa.envs.gridhouse import GridHouseEnvironment; env = GridHouseEnvironment(); print('OK')"`
2. **Verify VirtualHome** (if installed): `python -c "from src.bsa.envs.virtualhome import VirtualHomeEnvironment; print('OK')"`
3. **Run tests**: `pytest tests/`
4. **Generate episodes**: `make generate`

## Additional Resources

- VirtualHome GitHub: https://github.com/xavierpuigf/virtualhome
- VirtualHome Installation Notes: `src/bsa/envs/virtualhome/install_notes.md`
- Project README: `README.md`
