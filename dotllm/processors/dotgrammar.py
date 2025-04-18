from dotcfg import (
    Vocabulary,
    Guide,
    PartialLark,
    CFGVocabularyIndex,
    PartialParser,
)
from transformers import PreTrainedTokenizerBase

from dotllm.processors.base import BaseLogitsProcessor


class DotGrammarLogitsProcessor(BaseLogitsProcessor):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, grammar: str) -> None:
        self.grammar = grammar
        self.tokenizer = tokenizer

        lp = PartialLark(
            grammar,
            parser="lalr",
            deterministic=True,
            start="value",
            lexer="basic",
            lazy_build_scanner_fsm=False,
        )
        parser = PartialParser.from_lark(lp)
        vocabulary = Vocabulary.from_pretrained(tokenizer.name_or_path)
        index = CFGVocabularyIndex.build(parser, vocabulary)
        self.guide = Guide(index)

    def clone(self):
        return DotGrammarLogitsProcessor(self.tokenizer, self.grammar)
