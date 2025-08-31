import spacy
from spacy.tokens import Doc
import stanza

# 初始化 Stanza（只需一次）
stanza_nlp = stanza.Pipeline(lang="lv", processors="tokenize,pos,lemma", use_gpu=True)


class StanzaLemmatizerWrapper:
    """把 Stanza lemmatizer 包装成 spaCy pipeline 组件"""

    def __init__(self):
        pass

    def __call__(self, doc: Doc):
        text = doc.text
        stanza_doc = stanza_nlp(text)
        idx = 0
        for sent in stanza_doc.sentences:
            for token in sent.tokens:
                lemma = token.words[0].lemma
                doc[idx].lemma_ = lemma
                idx += 1
        return doc


@spacy.Language.factory("stanza_lemmatizer")
def create_stanza_lemmatizer(nlp, name):
    """注册 factory，方便 nlp.add_pipe("stanza_lemmatizer")"""
    return StanzaLemmatizerWrapper()
