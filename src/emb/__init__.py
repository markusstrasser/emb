try:
    from importlib.metadata import version
    __version__ = version("emb")
except Exception:
    __version__ = "0.0.0"

from emb.generation import (  # noqa: E402
    Generation,
    GenerationError,
    activate_generation,
    prune_generations,
    read_active_manifest,
    read_generation,
    validate_generation,
    write_generation,
)

__all__ = [
    "Generation",
    "GenerationError",
    "activate_generation",
    "prune_generations",
    "read_active_manifest",
    "read_generation",
    "validate_generation",
    "write_generation",
]
