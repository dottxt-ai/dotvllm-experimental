"""DotLLM custom logits processors."""

import logging

from transformers import PreTrainedTokenizerBase

from vllm.sampling_params import GuidedDecodingParams

from dotllm.processors.dotregex import DotRegexLogitsProcessor
from dotllm.processors.dotgrammar import DotGrammarLogitsProcessor
from dotllm.processors.dotjson import DotJsonLogitsProcessor


logger = logging.getLogger("dotllm.logits_processor")


def get_logits_processor(
    guided_decoding_params: GuidedDecodingParams,
    tokenizer: PreTrainedTokenizerBase,
):
    if guided_decoding_params.json:
        return DotJsonLogitsProcessor(tokenizer, guided_decoding_params.json)
    elif guided_decoding_params.regex:
        return DotRegexLogitsProcessor(tokenizer, guided_decoding_params.regex)
    elif guided_decoding_params.grammar:
        return DotGrammarLogitsProcessor(tokenizer, guided_decoding_params.grammar)
    else:
        raise ValueError(f"Unknown guided decoding mode {guided_decoding_params}")
