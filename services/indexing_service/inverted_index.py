from collections import defaultdict, Counter
from pathlib import Path
import sys


# حتى نستطيع استيراد preprocess_text من مجلد preprocessing_service
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from services.preprocessing_service.document_preprocessing import preprocess_document

def build_inverted_index(raw_documents: dict[str, str]) -> dict[str, list[tuple[str, int]]]:
    """
    Builds an inverted index from raw documents.

    Input:
        raw_documents = {
            "Doc1": "Apple apple banana!",
            "Doc2": "Apple orange."
        }

    Output:
        {
            "apple": [("Doc1", 2), ("Doc2", 1)],
            "banana": [("Doc1", 1)],
            "orange": [("Doc2", 1)]
        }
    """

    inverted_index = defaultdict(list)

    for doc_id, text in raw_documents.items():
        tokens = preprocess_document(text)
        term_frequencies = Counter(tokens)

        for term, frequency in term_frequencies.items():
            inverted_index[term].append((doc_id, frequency))

    return dict(inverted_index)


if __name__ == "__main__":
    sample_documents = {
        "Doc1": "Apple apple banana!",
        "Doc2": "Apple orange.",
        "Doc3": "Banana banana orange?",
    }

    index = build_inverted_index(sample_documents)

    print("Inverted Index:")
    for term, postings in index.items():
        print(term, "->", postings)