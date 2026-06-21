from services.query_refinement_service.spell_correction import correct_spelling
from services.query_refinement_service.synonym_expansion import expand_with_synonyms


def refine_query(query: str, use_spell_correction: bool = True, use_synonyms: bool = True) -> dict:
   
    original  = query
    corrected = query
    expanded  = query

    if use_spell_correction:
        corrected = correct_spelling(query)

    if use_synonyms:
        expanded = expand_with_synonyms(corrected)

    return {
        "original":  original,
        "corrected": corrected,
        "expanded":  expanded,
        "final":     expanded,
    }