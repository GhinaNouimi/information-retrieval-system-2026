from services.preprocessing_service.text_preprocessing import preprocess_text


def preprocess_document(text: str) -> list[str]:
   

    return preprocess_text(text)


if __name__ == "__main__":
    sample_document = "What is the best way to invest in stocks?"

    print("Original document:")
    print(sample_document)

    print("\nProcessed document:")
    print(preprocess_document(sample_document))