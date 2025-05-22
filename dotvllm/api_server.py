"""DotLLM API server implementation."""

import logging
import sys
import os
import uvloop

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
    set_ulimit,
)
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.utils import FlexibleArgumentParser, is_valid_ipv6_address


from dotllm.engine import DotEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("dotllm.api_server")

# Force V0 mode by setting environment variable directly
# Without this vLLM will try to instantiate a V1 `LLMEngine`
os.environ["VLLM_USE_V1"] = "0"


async def run_dot_server(args) -> None:
    """Run the DotLLM API server with a custom engine.

    This function is based on vLLM's `run_server` function but uses
    our custom DotEngine implementation.

    This is the only function that we had to copy almost verbatim to
    be able to write a minimal wrapper. I can't see a way around it.

    Code copied from vLLM:
    https://github.com/vllm-project/vllm/blob/9b70e2b4c147ea650f9b943e6aecd977377fbbfd/vllm/entrypoints/openai/api_server.py#L1041

    We only changed the definition of `engine_client`

    """
    logger.info("DotLLM API server starting...")
    logger.info("args: %s", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valid_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} "
            f"(chose from {{ {','.join(valid_tool_parses)} }})"
        )

    valid_reasoning_parses = ReasoningParserManager.reasoning_parsers.keys()
    if args.reasoning_parser and args.reasoning_parser not in valid_reasoning_parses:
        raise KeyError(
            f"invalid reasoning parser: {args.reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parses)} }})"
        )

    # Set up socket binding like vLLM
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # Increase ulimit if needed to handle more requests
    set_ulimit()

    # Create the FastAPI application
    app = build_app(args)

    # Create engine args and then modify to use our custom engine
    # !! This is the only line from the original code that was modified.
    engine_args = AsyncEngineArgs.from_cli_args(args)

    from vllm.usage.usage_lib import UsageContext

    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )
    engine_client: EngineClient = DotEngine.from_vllm_config(
        vllm_config=vllm_config,
        disable_log_requests=args.disable_log_requests,
        disable_log_stats=args.disable_log_stats,
    )

    try:
        # Initialize the app state with our engine
        model_config = await engine_client.get_model_config()
        await init_app_state(engine_client, model_config, app.state, args)

        # Log server info
        def _listen_addr(a: str) -> str:
            if is_valid_ipv6_address(a):
                return "[" + a + "]"
            return a or "0.0.0.0"

        is_ssl = args.ssl_keyfile and args.ssl_certfile
        logger.info(
            "Starting DotLLM API server on http%s://%s:%d",
            "s" if is_ssl else "",
            _listen_addr(sock_addr[0]),
            sock_addr[1],
        )

        # Serve HTTP using vLLM's serve_http helper
        from vllm.entrypoints.launcher import serve_http

        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=5,  # seconds
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
        )

        # Wait for server shutdown
        await shutdown_task
    finally:
        # Ensure cleanup on exit
        if hasattr(engine_client, "shutdown"):
            engine_client.shutdown()
        sock.close()


def cli_main():
    """CLI entrypoint for the DotLLM API server.

    Code copied from vLLM:
    https://github.com/vllm-project/vllm/blob/9b70e2b4c147ea650f9b943e6aecd977377fbbfd/vllm/entrypoints/openai/api_server.py#L1118

    """
    from vllm.entrypoints.openai.cli_args import validate_parsed_serve_args
    from vllm.entrypoints.utils import cli_env_setup
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    # Set up CLI environment
    cli_env_setup()

    # Parse and validate arguments
    parser = FlexibleArgumentParser(description="DotLLM OpenAI-Compatible API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    # Run the server
    uvloop.run(run_dot_server(args))


if __name__ == "__main__":
    cli_main()
