import os
import time

from alligator import PROJECT_ROOT, Alligator
from alligator.log import enable_logging


def main():
    num_workers = 4
    num_ml_workers = 2
    worker_batch_size = 64
    candidate_retrieval_limit = 16
    mongo_uri = "mongodb://localhost:27017/"
    input_csv = os.path.join(PROJECT_ROOT, "tables", "imdb_top_100.csv")

    # Define column types (Wikidata QIDs) for specific columns
    # This is an example mapping columns to entity types
    column_types = {"3": ["Q483394"], "0": ["Q11424"], "7": ["Q5"], "8": ["Q5"]}

    enable_logging()
    tic = time.perf_counter()
    gator = Alligator(
        input_csv=input_csv,
        num_workers=num_workers,
        num_ml_workers=num_ml_workers,
        worker_batch_size=worker_batch_size,
        candidate_retrieval_limit=candidate_retrieval_limit,
        column_types=column_types,  # Pass the column types
        mongo_uri=mongo_uri,
    )
    gator.run()
    toc = time.perf_counter()
    print(f"Entity linking process completed in {toc - tic:0.4f} seconds.")


if __name__ == "__main__":
    main()
