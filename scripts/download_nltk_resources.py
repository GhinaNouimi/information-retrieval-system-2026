import nltk


def main():
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger_eng",
    ]

    for resource in resources:
        nltk.download(resource)

    print("NLTK resources installed successfully.")


if __name__ == "__main__":
    main()