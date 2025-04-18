"""DotLLM CLI entrypoint."""

import importlib.metadata

# Get version from pyproject.toml
try:
    __version__ = importlib.metadata.version("dotllm")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"


def main():
    """Main entry point for the DotLLM CLI."""
    # For now, just pass through all args to the API server
    from dotllm.api_server import cli_main

    cli_main()


if __name__ == "__main__":
    main()
