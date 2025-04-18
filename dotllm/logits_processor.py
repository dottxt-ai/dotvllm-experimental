"""DotLLM custom logits processors."""

import logging
from typing import List

import torch
from transformers import PreTrainedTokenizerBase

from vllm.sampling_params import GuidedDecodingParams


logger = logging.getLogger("dotllm.logits_processor")


def get_logits_processor(
    guided_decoding_params: GuidedDecodingParams,
    tokenizer: PreTrainedTokenizerBase,
):
    if guided_decoding_params.json:
        json = guided_decoding_params.json
        return GuidedLogitsProcessor()
    elif guided_decoding_params.regex:
        return GuidedLogitsProcessor()
    elif guided_decoding_params.grammar:
        return GuidedLogitsProcessor()
    else:
        raise ValueError(f"Unknown guided decoding mode {mode}")


class GuidedLogitsProcessor:
    """Custom logits processor for guided generation.
    
    This is a simple pass-through implementation that returns the original logits.
    """
    
    def __init__(self) -> None:
        """Initialize the guided logits processor."""
        logger.info("GuidedLogitsProcessor initialized")
    
    def __call__(self, input_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        """Process logits (currently just passes through).
        
        Args:
            logits: The model's output logits tensor.
            tokens: List of previously generated tokens.
            
        Returns:
            The original logits tensor.
        """
        return logits
    
    def clone(self) -> "GuidedLogitsProcessor":
        """Create a clone of this processor for parallel sampling."""
        return GuidedLogitsProcessor()
