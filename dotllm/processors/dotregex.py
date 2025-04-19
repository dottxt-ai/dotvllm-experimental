import logging

from dotregex import Vocabulary, Index, Guide

logger = logging.getLogger("dotllm.processors.dotregex")


def compile_regex(model_name: str, regex_str: str):
    logger.info(f"Compiling regex index for pattern: {regex_str[:50]}...")
    vocabulary = Vocabulary.from_pretrained(model_name)
    index = Index(regex_str, vocabulary)
    guide = Guide(index)
    logger.info("Regex index compilation complete")

    return guide
