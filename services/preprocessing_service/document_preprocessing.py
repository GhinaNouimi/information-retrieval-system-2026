from services.preprocessing_service.text_preprocessing import preprocess_text


def preprocess_document(text: str) -> list[str]:
    return preprocess_text(text)