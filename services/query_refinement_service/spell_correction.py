from spellchecker import SpellChecker

spell = SpellChecker()


def correct_spelling(query: str) -> str:
   
    words = query.split()
    corrected_words = []

    for word in words:
        corrected = spell.correction(word)
        if corrected is not None:
            corrected_words.append(corrected)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)