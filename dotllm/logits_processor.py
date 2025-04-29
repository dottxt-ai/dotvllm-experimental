"""DotLLM custom logits processors."""

import logging
from typing import Optional

from transformers import PreTrainedTokenizerBase
import numpy as np
import torch

from vllm.sampling_params import GuidedDecodingParams

from dotllm.processors.dotregex import compile_regex, build_regex_guide
from dotllm.processors.dotgrammar import compile_grammar, build_grammar_guide
from dotllm.processors.dotjson import compile_json, build_json_guide
from dotllm.compilation_manager import CompilationManager


logger = logging.getLogger("dotllm.logits_processor")


def get_logits_processor(
    guided_decoding_params: GuidedDecodingParams,
    tokenizer: PreTrainedTokenizerBase,
    compilation_manager: Optional[CompilationManager] = None,
):
    """Get a logits processor for the given guided decoding parameters.

    Args:
        guided_decoding_params: The guided decoding parameters.
        tokenizer: The tokenizer to use.
        compilation_manager: Optional compilation manager for asynchronous compilation.
            If provided, the index building will be performed in a background thread.

    Returns:
        A logits processor for the given parameters.

    Raises:
        ValueError: If the guided decoding mode is unknown.
    """
    model_name = tokenizer.name_or_path

    if guided_decoding_params.json:
        compilation_key = compilation_manager.submit(
            compile_json, model_name, guided_decoding_params.json
        )
        build_guide = build_json_guide
    elif guided_decoding_params.regex:
        compilation_key = compilation_manager.submit(
            compile_regex, model_name, guided_decoding_params.regex
        )
        build_guide = build_regex_guide
    elif guided_decoding_params.grammar:
        compilation_key = compilation_manager.submit(
            compile_grammar, model_name, guided_decoding_params.grammar
        )
        build_guide = build_grammar_guide
    else:
        raise ValueError(f"Unknown guided decoding mode {guided_decoding_params}")

    return LogitsProcessor(compilation_key, compilation_manager, build_guide)


class LogitsProcessor:
    def __init__(self, compilation_key: str, compilation_manager, build_guide):
        """Initialize the base logits processor.

        Args:
            guide: Pre-compiled guide instance. If None, compilation will be needed.
        """
        self.compilation_key = compilation_key
        self.compilation_manager = compilation_manager
        self.build_guide = build_guide
        self.guide = None

    def __call__(self, input_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        """Mask the allowed tokens.

        This implementation waits for guide compilation to complete if necessary,
        then uses the guide to determine allowed tokens.

        Args:
            input_ids: The input token IDs.
            logits: The logits to process.

        Returns:
            The processed logits.
        """

        # During the first run we retrieve the compiled index from the compilation
        # manager (and wait for it to be available) and build the guide.
        #
        # Ideally we would use a custom scheduler that does not schedule sequences
        # for generation until the index is compiled to avvoid blocking.
        if self.guide is None:
            serialized_index = self.compilation_manager.get_index(self.compilation_key)
            if serialized_index is None:
                return logits
            self.guide = self.build_guide(serialized_index)

        # Now use the guide to process logits
        #
        # This is a very low-performance implementation. This should be replaced, and
        # then ideally move this computation to happen behind the forward pass, which
        # may require modifying `AsyncEngine` or subclassing the sampler.
        if len(input_ids) == 0:
            allowed_tokens = self.guide.get_start_tokens()
        else:
            last_token = input_ids[-1]
            allowed_tokens = self.guide.read_next_token(last_token)

        mask = torch.full((logits.shape[-1],), -torch.inf, device=logits.device)
        allowed_tokens = np.array(allowed_tokens, dtype=np.int64)
        allowed_tokens = torch.tensor(allowed_tokens, device=logits.device)
        mask.index_fill_(0, allowed_tokens, 0)
        return logits.add_(mask)

    def __clone__(self):
        return LogitsProcessor(
            self.compilation_key, self.compilation_manager, self.build_guide
        )
