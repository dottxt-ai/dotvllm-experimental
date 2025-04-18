"""DotLLM Engine implementation."""

import os
import logging
import vllm.envs as envs
from typing import Dict, Optional, Type, Any
from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine
from vllm.config import VllmConfig
from vllm.engine.metrics_types import StatLoggerBase
from vllm.usage.usage_lib import UsageContext
from vllm.executor.executor_base import ExecutorBase


logger = logging.getLogger("dotllm.engine")


class _DotAsyncLLMEngine(_AsyncLLMEngine):
    """Extended AsyncLLMEngine with custom behavior for DotLLM."""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Add custom initialization here if needed


class DotEngine(AsyncLLMEngine):
    """Custom AsyncLLMEngine for DotLLM.
    
    This class extends vLLM's AsyncLLMEngine to support custom behavior
    for the DotLLM CLI and API server.
    """
    
    _engine_class: Type[_AsyncLLMEngine] = _DotAsyncLLMEngine
    
    @classmethod
    def _get_executor_cls(cls, engine_config: VllmConfig) -> Type[ExecutorBase]:
        """Get the executor class based on the engine configuration."""
        return AsyncLLMEngine._get_executor_cls(engine_config)
    
    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        disable_log_requests: bool = False,
        disable_log_stats: bool = False,
    ) -> "DotEngine":
        """Create a DotEngine from VllmConfig."""
        
        # Force V0 mode by setting environment variable directly
        # This affects all imported modules that check this env var
        os.environ["VLLM_USE_V1"] = "0"
        envs.VLLM_USE_V1 = False
        
        logger.info("DotEngine: Using vLLM V0 engine for better compatibility")
        
        try:
            engine = cls(
                vllm_config=vllm_config,
                executor_class=cls._get_executor_cls(vllm_config),
                start_engine_loop=start_engine_loop,
                log_requests=not disable_log_requests,
                log_stats=not disable_log_stats,
                usage_context=usage_context,
                stat_loggers=stat_loggers,
            )
            return engine
        except Exception as e:
            logger.error(f"Error creating DotEngine: {e}")
            raise
