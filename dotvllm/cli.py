"""DotLLM CLI entrypoint."""


def main():
    """Main entry point for the DotLLM CLI."""
    from dotllm.api_server import cli_main

    cli_main()


if __name__ == "__main__":
    main()
