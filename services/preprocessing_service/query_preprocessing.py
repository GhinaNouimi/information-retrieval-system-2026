from services.preprocessing_service.text_preprocessing import preprocess_text


def preprocess_query(text: str) -> list[str]:
    return preprocess_text(text)