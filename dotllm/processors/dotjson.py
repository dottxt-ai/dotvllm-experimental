import logging

from dotregex import Vocabulary, Index, Guide

logger = logging.getLogger("dotllm.processors.dotjson")


def compile_json(model_name: str, json_schema: str):
    logger.info(f"Compiling JSON index for schema: {json_schema[:50]}...")
    vocabulary = Vocabulary.from_pretrained(model_name)
    index = Index.from_schema(json_schema, vocabulary)
    guide = Guide(index)
    logger.info("JSON index compilation complete")

    return guide
