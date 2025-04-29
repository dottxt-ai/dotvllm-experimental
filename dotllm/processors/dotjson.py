import logging


logger = logging.getLogger("dotllm.processors.dotjson")


def compile_json(model_name: str, json_schema: str):
    from dotregex import Vocabulary, Index

    logger.info(f"Compiling JSON index for schema: {json_schema[:50]}...")
    vocabulary = Vocabulary.from_pretrained(model_name)
    index = Index.from_schema(json_schema, vocabulary)
    logger.info("JSON index compilation complete")
    return index.serialize()


def build_json_guide(serialized_index):
    from dotregex import Index, Guide

    index = Index.deserialize(serialized_index)
    return Guide(index)
