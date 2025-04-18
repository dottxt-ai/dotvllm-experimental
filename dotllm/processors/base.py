import numpy as np
import torch


class BaseLogitsProcessor:
    def __init__(self):
        pass

    def __call__(self, input_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        """Mask the allowed tokens.

        This implementation is very inefficient and could be greatly improved by:
        1. Modifying the logits in-place
        2. Computing the allowed tokens during the forward pass.

        """
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
