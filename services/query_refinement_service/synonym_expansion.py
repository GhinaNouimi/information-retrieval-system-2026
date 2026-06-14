from nltk.corpus import wordnet
from nltk import pos_tag


def expand_with_synonyms(query: str, max_synonyms: int = 2) -> str:
    words = query.split()
    expanded_words = list(words)

    tagged = pos_tag(words)

    for word, pos in tagged:
        wn_pos = _get_wordnet_pos(pos)
        if wn_pos is None:
            continue
        synonyms = _get_synonyms(word, wn_pos, max_synonyms)
        for synonym in synonyms:
            if synonym not in expanded_words:
                expanded_words.append(synonym)

    return " ".join(expanded_words)


def _get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return None


def _get_synonyms(word: str, pos, max_synonyms: int) -> list:
    synonyms = []

    for synset in wordnet.synsets(word, pos=pos):
        for lemma in synset.lemmas():
            synonym = lemma.name().replace("_", " ")

            if synonym.lower() == word.lower():
                continue
            if not synonym.replace(" ", "").isalpha():
                continue
            if synonym[0].isupper():
                continue
            if len(synonym) < 3:
                continue

            synonyms.append(synonym)

            if len(synonyms) >= max_synonyms:
                return synonyms

    return synonyms