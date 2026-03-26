"""DeepScript: Transcript Intelligence Engine."""

__version__ = "0.1.0"

# Auto-load .env for API keys (NVIDIA_API_KEY, ANTHROPIC_API_KEY, etc.)
from pathlib import Path as _Path

def _load_env() -> None:
    """Load .env file from current dir or project root."""
    for env_path in [_Path(".env"), _Path(__file__).parent.parent / ".env"]:
        if env_path.exists():
            import os
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        if key and key not in os.environ:  # Don't override existing env vars
                            os.environ[key] = value
            break

_load_env()

from deepscript.utils.logging import setup_logging
setup_logging()
