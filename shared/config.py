from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "beir/quora/test"
SAMPLE_SIZE = 10_000
LARGE_SIZE  = 50_000      
TOP_K = 10

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TFIDF_SAMPLE_DIR = ARTIFACTS_DIR / "tfidf_sample"
BM25_SAMPLE_DIR = ARTIFACTS_DIR / "bm25_sample"
TFIDF_LARGE_DIR  = ARTIFACTS_DIR / "tfidf_large"  
BM25_LARGE_DIR   = ARTIFACTS_DIR / "bm25_large"    
DATABASE_DIR = ARTIFACTS_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "documents_sample.db"