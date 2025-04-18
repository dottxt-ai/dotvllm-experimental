
from dotregex import Vocabulary, Index, Guide
from transformers import PreTrainedTokenizerBase
from dotllm.processors.base import BaseLogitsProcessor


class DotRegexLogitsProcessor(BaseLogitsProcessor):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, regex: str) -> None:
        self.regex = regex
        self.tokenizer = tokenizer
        vocabulary = Vocabulary.from_pretrained(tokenizer.name_or_path)
        index = Index(regex, vocabulary)
        self.guide = Guide(index)

    def clone(self):
        return DotRegexLogitsProcessor(self.tokenizer, self.regex)
