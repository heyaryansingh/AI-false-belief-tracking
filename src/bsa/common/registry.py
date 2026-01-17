"""Registry pattern for extensible components."""

from typing import Dict, Type, TypeVar, Callable

T = TypeVar("T")


class Registry:
    """Simple registry for components."""

    def __init__(self) -> None:
        self._registry: Dict[str, Type[T]] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Register a component."""

        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                raise ValueError(f"Component '{name}' already registered")
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type[T]:
        """Get a registered component."""
        if name not in self._registry:
            raise ValueError(f"Component '{name}' not found in registry")
        return self._registry[name]

    def list(self) -> list[str]:
        """List all registered components."""
        return list(self._registry.keys())


# Global registries
env_registry = Registry()
agent_registry = Registry()
model_registry = Registry()
metric_registry = Registry()
