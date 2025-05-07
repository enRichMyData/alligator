import argparse
import asyncio
import os
import re
import sys
import time

import pandas as pd
import tqdm
from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from alligator import PROJECT_ROOT
from alligator.alligator import Alligator
from alligator.database import DatabaseManager
from alligator.mongo import MongoWrapper

load_dotenv(PROJECT_ROOT)


async def main(args: argparse.Namespace):
    perf = {}
    gt = pd.read_csv(
        args.ground_truth,
        delimiter=",",
        names=["tab_id", "row_id", "col_id", "entity"],
        dtype={"tab_id": str, "row_id": str, "col_id": str, "entity": str},
        keep_default_na=False,
    )
    for table_path in tqdm.tqdm(os.listdir(args.tables_dir)):
        if os.path.splitext(table_path)[1] != ".csv":
            continue
        table_path = os.path.join(args.tables_dir, table_path)
        if not os.path.exists(table_path):
            print(f"Error: File {table_path} does not exist.")
            continue
        # Load the table
        try:
            table = pd.read_csv(table_path, header=None)
        except Exception as e:
            print(f"Error loading file {table_path}: {e}")
            continue

        # Identify ne-columns from ground truth
        tab_id = os.path.split(table_path)[-1].split(".")[0]
        unique_cols = gt[gt["tab_id"] == tab_id]["col_id"].unique()
        columns_type = {
            "NE": {str(col_id): "" for col_id in unique_cols},
            "LIT": {
                str(col_id): ""
                for col_id in range(len(table.columns))
                if str(col_id) not in unique_cols
            },
            "IGNORED": [],
        }

        # Correct qids dictionary for every row-col pair
        correct_qids = {}
        min_row_is_one = gt["row_id"].astype(int).min() == 1
        for _, row in gt[gt["tab_id"] == tab_id].iterrows():
            col_id = row["col_id"]
            row_id = row["row_id"]
            entity = row["entity"]
            correct_qids[f"{int(row_id) - int(min_row_is_one)}-{col_id}"] = re.findall(
                r"Q\d+", entity
            )

        tic = time.perf_counter()
        args.gator.input_csv = table_path
        args.gator.columns_type = columns_type
        args.gator.dataset_name = args.dataset_name
        args.gator.table_name = tab_id
        args.gator.correct_qids = correct_qids

        # Check if the table has already been processed
        db = DatabaseManager.get_database(args.gator.mongo_uri, "alligator_db")
        mongo = MongoWrapper(mongo_uri=args.gator.mongo_uri, db_name="alligator_db")
        cursor = mongo.find_one_document(
            collection=db.get_collection("input_data"),
            query={"dataset_name": args.gator.dataset_name, "table_name": tab_id},
            projection={"rank_status": 1, "rerank_status": 1},
        )
        if cursor is not None:
            if cursor.get("rank_status") == "DONE" and cursor.get("rerank_status") == "DONE":
                print(f"Table {tab_id} already processed. Skipping.")
                continue
            else:
                print(f"Table {tab_id} already exists in the database but not fully processed.")
                mongo.delete_documents(
                    collection=db.get_collection("input_data"),
                    query={"dataset_name": args.gator.dataset_name, "table_name": tab_id},
                )
                print(f"Deleted incomplete document for table {tab_id}.")
        else:
            print(f"Table {tab_id} not found in the database. Proceeding with processing.")

        gator = Alligator(**args.gator)
        await gator.run()
        toc = time.perf_counter()
        print(f"Processing completed in {toc - tic:0.4f} seconds.")
        perf[table_path] = {"elapsed_time": toc - tic, "nrows": len(table)}

    perf_df = pd.DataFrame.from_dict(perf, orient="index")
    print(
        "Average elapsed time per row: {} row/s".format(
            perf_df["nrows"].sum() / perf_df["elapsed_time"].sum()
        )
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--tables_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "eval", "tables", "HardTablesR1", "Valid", "tables"),
    )
    parser.add_argument("--dataset_name", type=str, default="htr1-correct-qids")
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "eval", "tables", "HardTablesR1", "Valid", "gt", "cea_gt.csv"
        ),
    )
    parser.add_class_arguments(Alligator, "gator")
    parser.add_argument(
        "--gator.mongo-uri",
        type=str,
        help="MongoDB connection URI",
        default="localhost:27017",
    )
    args = parser.parse_args()

    if not os.path.exists(args.tables_dir):
        print(f"Error: Directory {args.tables_dir} does not exist.")
        sys.exit(1)
    asyncio.run(main(args))
