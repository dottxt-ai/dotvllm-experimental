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
# Without this vLLM will try to instantiate a `LLMEngine`
os.environ["VLLM_USE_V1"] = "0"


async def run_dot_server(args) -> None:
    """Run the DotLLM API server with a custom engine.
    
    This function is based on vLLM's run_server function but uses
    our custom DotEngine implementation.

    """
    logger.info("DotLLM API server starting...")
    logger.info("args: %s", args)

    # Set up socket binding like vLLM
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)
    
    # Increase ulimit if needed to handle more requests
    set_ulimit()
    
    # Create the FastAPI application
    app = build_app(args)
    
    # Create engine args and then modify to use our custom engine
    engine_args = AsyncEngineArgs.from_cli_args(args)
    
    # We need to use the proper enum value for usage_context
    from vllm.usage.usage_lib import UsageContext
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )
    
    # Use our custom DotEngine instead of AsyncLLMEngine
    engine_client: EngineClient = DotEngine.from_vllm_config(
        vllm_config=vllm_config,
        disable_log_requests=args.disable_log_requests,
        disable_log_stats=args.disable_log_stats
    )
    
    try:
        # Initialize the app state with our engine
        model_config = await engine_client.get_model_config()
        await init_app_state(engine_client, model_config, app.state, args)
        
        # Log server info
        def _listen_addr(a: str) -> str:
            if is_valid_ipv6_address(a):
                return '[' + a + ']'
            return a or "0.0.0.0"
        
        is_ssl = args.ssl_keyfile and args.ssl_certfile
        logger.info(
            "Starting DotLLM API server on http%s://%s:%d",
            "s" if is_ssl else "", 
            _listen_addr(sock_addr[0]),
            sock_addr[1]
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


def make_parser() -> FlexibleArgumentParser:
    """Create argument parser for DotLLM CLI.
    
    This uses vLLM's argument parser with any additions we need.
    """
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    
    parser = FlexibleArgumentParser(
        description="DotLLM OpenAI-Compatible API server."
    )
    parser = make_arg_parser(parser)

    return parser


def cli_main():
    """CLI entrypoint for the DotLLM API server."""
    from vllm.entrypoints.openai.cli_args import validate_parsed_serve_args
    from vllm.entrypoints.utils import cli_env_setup
    
    # Set up CLI environment
    cli_env_setup()
    
    # Parse and validate arguments
    parser = make_parser()
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    
    # Run the server
    uvloop.run(run_dot_server(args))


if __name__ == "__main__":
    cli_main()
