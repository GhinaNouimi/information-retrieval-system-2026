import ir_datasets

DATASET_ID = "beir/quora/test"

print("Starting dataset validation...")
print(f"Dataset ID: {DATASET_ID}")
print("Loading dataset...")

dataset = ir_datasets.load(DATASET_ID)

print("Dataset loaded successfully.")
print()

print("Basic checks:")
print("Has Documents:", dataset.has_docs())
print("Has Queries:", dataset.has_queries())
print("Has Qrels:", dataset.has_qrels())
print()

docs_count = dataset.docs_count()
queries_count = dataset.queries_count()

print("Counting qrels...")
qrels_count = sum(1 for _ in dataset.qrels_iter())

print()
print("Dataset statistics:")
print("Documents:", docs_count)
print("Queries:", queries_count)
print("Qrels:", qrels_count)
print()

print("Sample document:")
print(next(dataset.docs_iter()))
print()

print("Sample query:")
print(next(dataset.queries_iter()))
print()

print("Sample qrel:")
print(next(dataset.qrels_iter()))
print()

is_valid = (
    dataset.has_docs()
    and dataset.has_queries()
    and dataset.has_qrels()
    and docs_count > 200_000
    and qrels_count > 0
    and "antique" not in DATASET_ID.lower()
)

if is_valid:
    print("Final result: Dataset is valid for the project.")
else:
    print("Final result: Dataset is NOT valid for the project.")