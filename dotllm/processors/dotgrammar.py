import logging


logger = logging.getLogger("dotllm.processors.dotgrammar")


def compile_grammar(model_name: str, grammar: str):
    from dotcfg import (
        Vocabulary,
        PartialLark,
        CFGVocabularyIndex,
        PartialParser,
    )

    logger.info("Compiling grammar index...")
    lp = PartialLark(
        grammar,
        parser="lalr",
        deterministic=True,
        start="value",
        lexer="basic",
        lazy_build_scanner_fsm=False,
    )
    parser = PartialParser.from_lark(lp)
    vocabulary = Vocabulary.from_pretrained(model_name)
    index = CFGVocabularyIndex.build(parser, vocabulary)
    logger.info("Grammar index compilation complete")
    return index.serialize()


def build_grammar_guide(serialized_index):
    from dotcfg import Guide, CFGVocabularyIndex

    index = CFGVocabularyIndex.deserialize(serialized_index)
    return Guide(index)
