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

    enable_logging()
    tic = time.perf_counter()
    gator = Alligator(
        input_csv=input_csv,
        num_workers=num_workers,
        num_ml_workers=num_ml_workers,
        worker_batch_size=worker_batch_size,
        candidate_retrieval_limit=candidate_retrieval_limit,
        mongo_uri=mongo_uri,
    )
    gator.run()
    toc = time.perf_counter()
    print(f"Entity linking process completed in {toc - tic:0.4f} seconds.")


if __name__ == "__main__":
    main()
