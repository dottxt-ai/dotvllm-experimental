"""DotLLM Engine implementation."""

import logging
from typing import Optional, Type, Any, Union, Mapping
from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine
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
        """Add a request to the engine with support for structured generation.

        This method extends the parent class method to check for the `guided_decoding`
        attribute in params and apply custom logits processing if present.

        !! Warning

           We should validate the schema before submitting the request
           since an invalid schema will make the compilation fail.

           Raising an exception in a task will take down the engine
           loop, raising an `AsyncEngineDeadError`.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string.
            params: The sampling parameters.
            prompt_token_ids: The token IDs of the prompt. If None, the prompt will
                be tokenized.
            arrival_time: The arrival time of the request.
            **kwargs: Additional arguments.
        """

        if isinstance(params, SamplingParams) and params.guided_decoding is not None:
            logger.info(f"Using guided decoding for request {request_id}")
            tokenizer = await self.get_tokenizer_async(lora_request)
            guided_decoding = params.guided_decoding

            # Validate the schema here
            processor = get_logits_processor(
                guided_decoding, tokenizer, self.compilation_manager
            )

            if processor:
                if params.logits_processors is None:
                    params.logits_processors = []
                params.logits_processors.append(processor)

            # So that super().add_request_async does not try to
            # set the logits processor
            params.guided_decoding = None

        return await super().add_request_async(
            request_id=request_id,
            prompt=prompt,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
            inputs=inputs,
        )


class DotEngine(AsyncLLMEngine):
    """Custom AsyncLLMEngine for DotLLM.

    This class extends vLLM's `AsyncLLMEngine` to support custom behavior for
    the DotLLM CLI and API server, including using our proprietary libraries for
    structured generation.

    """

    _engine_class: Type[_AsyncLLMEngine] = _DotAsyncLLMEngine
