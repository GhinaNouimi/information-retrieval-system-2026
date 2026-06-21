import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def normalize_text(text: str) -> str:
   

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text: str) -> list[str]:
   

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