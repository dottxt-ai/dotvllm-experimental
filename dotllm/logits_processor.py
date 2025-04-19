"""DotLLM custom logits processors."""

import logging
from typing import Optional

from transformers import PreTrainedTokenizerBase
import numpy as np
import torch

from vllm.sampling_params import GuidedDecodingParams

from dotllm.processors.dotregex import compile_regex
from dotllm.processors.dotgrammar import compile_grammar
from dotllm.processors.dotjson import compile_json
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
    elif guided_decoding_params.regex:
        compilation_key = compilation_manager.submit(
            compile_regex, model_name, guided_decoding_params.regex
        )
    elif guided_decoding_params.grammar:
        compilation_key = compilation_manager.submit(
            compile_grammar, model_name, guided_decoding_params.grammar
        )
    else:
        raise ValueError(f"Unknown guided decoding mode {guided_decoding_params}")

    return LogitsProcessor(compilation_key, compilation_manager)


class LogitsProcessor:
    def __init__(self, compilation_key: str, compilation_manager):
        """Initialize the base logits processor.

        Args:
            guide: Pre-compiled guide instance. If None, compilation will be needed.
        """
        self.compilation_key = compilation_key
        self.compilation_manager = compilation_manager

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
        guide = self.compilation_manager.get_guide(self.compilation_key)
        if guide is None:
            return logits

        # Now use the guide to process logits
        if len(input_ids) == 0:
            allowed_tokens = guide.get_start_tokens()
        else:
            last_token = input_ids[-1]
            allowed_tokens = guide.read_next_token(last_token)

        mask = torch.full((logits.shape[-1],), -torch.inf, device=logits.device)
        allowed_tokens = np.array(allowed_tokens, dtype=np.int64)
        allowed_tokens = torch.tensor(allowed_tokens, device=logits.device)
        mask.index_fill_(0, allowed_tokens, 0)
        return logits.add_(mask)

    def __clone__(self):
        return LogitsProcessor(self.compilation_key, self.compilation_manager)
