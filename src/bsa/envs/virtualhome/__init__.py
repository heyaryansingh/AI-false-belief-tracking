"""VirtualHome adapter."""

try:
    from .env import VirtualHomeEnvironment

    __all__ = ["VirtualHomeEnvironment"]
except ImportError:
    # VirtualHome not installed - export empty
    __all__ = []
