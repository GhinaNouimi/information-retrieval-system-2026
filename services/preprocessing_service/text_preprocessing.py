import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def normalize_text(text: str) -> str:
    """
    Applies text normalization.

    Steps:
    1. Lowercase
    2. Remove punctuation and special symbols
    3. Remove extra spaces
    """

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text: str) -> list[str]:
    """
    Shared preprocessing pipeline used by both documents and queries.

    Steps:
    1. Normalize text
    2. Tokenize
    3. Remove stopwords
    4. Lemmatize tokens

    This function is intentionally shared to guarantee that documents
    and user queries are processed using the same techniques.
    """

    normalized_text = normalize_text(text)
    tokens = nltk.word_tokenize(normalized_text)

    processed_tokens = []

    for token in tokens:
        if token not in stop_words:
            lemma = lemmatizer.lemmatize(token)
            processed_tokens.append(lemma)

    return processed_tokens


if __name__ == "__main__":
    sample_text = "What is the best way to invest in stocks?"

    print("Original text:")
    print(sample_text)

    print("\nProcessed tokens:")
    print(preprocess_text(sample_text))