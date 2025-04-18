
from dotregex import Vocabulary, Index, Guide
from transformers import PreTrainedTokenizerBase

from dotllm.processors.base import BaseLogitsProcessor


class DotJsonLogitsProcessor(BaseLogitsProcessor):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, json_schema: str) -> None:
        """Initialize the Guide."""
        self.json_schema = json_schema
        self.tokenizer = tokenizer
        vocabulary = Vocabulary.from_pretrained(tokenizer.name_or_path)
        index = Index.from_schema(json_schema, vocabulary)
        self.guide = Guide(index)

    def clone(self):
        return DotJsonLogitsProcessor(self.tokenizer, self.json_schema)
