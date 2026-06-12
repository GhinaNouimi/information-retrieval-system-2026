import sqlite3
from shared.config import DATABASE_DIR, DATABASE_PATH


DATABASE_DIR.mkdir(parents=True, exist_ok=True)


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def create_documents_table():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            text TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def insert_document(doc_id: str, text: str):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO documents(doc_id, text)
        VALUES (?, ?)
    """, (doc_id, text))

    connection.commit()
    connection.close()


def get_document_by_id(doc_id: str):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT text
        FROM documents
        WHERE doc_id = ?
    """, (doc_id,))

    result = cursor.fetchone()
    connection.close()

    if result is None:
        return None

    return result[0]


if __name__ == "__main__":
    create_documents_table()

    insert_document("1", "This is a sample document.")

    document_text = get_document_by_id("1")

    print("Document:")
    print(document_text)