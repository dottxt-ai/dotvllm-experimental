"""DotLLM custom logits processors."""

from typing import List
import torch
import logging

logger = logging.getLogger("dotllm.logits_processor")


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
