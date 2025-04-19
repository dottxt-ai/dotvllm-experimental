"""DotLLM Engine implementation."""

import copy
import os
import logging
import vllm.envs as envs
from typing import Dict, Optional, Type, Any, Union, Mapping
from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine
from vllm.config import VllmConfig
from vllm.engine.metrics_types import StatLoggerBase
from vllm.usage.usage_lib import UsageContext
from vllm.executor.executor_base import ExecutorBase
from vllm.sampling_params import SamplingParams
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.lora.request import LoRARequest
from vllm.inputs import PromptType


from dotllm.logits_processor import get_logits_processor
from dotllm.compilation_manager import CompilationManager


logger = logging.getLogger("dotllm.engine")


class _DotAsyncLLMEngine(_AsyncLLMEngine):
    """Extended AsyncLLMEngine with custom behavior for DotLLM."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.compilation_manager = CompilationManager()

    async def add_request_async(
        self,
        request_id: str,
        prompt: Optional[PromptType] = None,
        params: Optional[Union[SamplingParams, PoolingParams]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
        *,
        inputs: Optional[PromptType] = None,  # DEPRECATED
    ) -> None:
        """Add a request to the engine with support for guided_decoding.

        This method extends the parent class method to check for the guided_decoding
        attribute in params and apply custom logits processing if present.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string.
            params: The sampling parameters.
            prompt_token_ids: The token IDs of the prompt. If None, the prompt will
                be tokenized.
            arrival_time: The arrival time of the request.
            **kwargs: Additional arguments.
        """
        if self.tokenizer is not None:
            tokenizer = await self.get_tokenizer_async(lora_request)
            self._validate_token_prompt(prompt, tokenizer=tokenizer)

        preprocessed_inputs = await self.input_preprocessor.preprocess_async(
            prompt,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )
        processed_inputs = self.input_processor(preprocessed_inputs)

        # Check if guided_decoding is present in params
        if isinstance(params, SamplingParams) and params.guided_decoding is not None:
            logger.info(f"Using guided decoding for request {request_id}")

            # Defensively copy sampling params since guided decoding logits
            # processors can have different state for each request
            params = copy.copy(params)
            guided_decoding = params.guided_decoding
            processor = get_logits_processor(
                guided_decoding, tokenizer, self.compilation_manager
            )

            if processor:
                if params.logits_processors is None:
                    params.logits_processors = []
                params.logits_processors.append(processor)

            # Unset guided decoding params after constructing the lp from them
            params.guided_decoding = None

        self._add_processed_request(
            request_id=request_id,
            processed_inputs=processed_inputs,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            trace_headers=trace_headers,
            priority=priority,
        )


class DotEngine(AsyncLLMEngine):
    """Custom AsyncLLMEngine for DotLLM.

    This class extends vLLM's AsyncLLMEngine to support custom behavior
    for the DotLLM CLI and API server, including guided logits processing.
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
