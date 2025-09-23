import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from evaluators.cea_wd import CEA_Evaluator

from alligator import PROJECT_ROOT
from alligator.database import DatabaseManager
from alligator.mongo import MongoWrapper

load_dotenv(PROJECT_ROOT)


def main(args: argparse.Namespace):
    db = DatabaseManager.get_database(args.mongo_uri, "alligator_db")
    mongo = MongoWrapper(mongo_uri=args.mongo_uri, db_name="alligator_db")
    cursor = mongo.find_documents(
        collection=db.get_collection("input_data"),
        query={"dataset_name": args.dataset_name, "status": "DONE", "rerank_status": "DONE"},
        projection={"cea": 1, "table_name": 1, "dataset_name": 1, "row_id": 1},
    )
    cea_results = []
    for doc in cursor:
        table_name = doc["table_name"]
        row_id = doc["row_id"]
        if "cea" not in doc:
            print(f"Skipping document {doc['_id']} due to missing 'cea'.")
            continue
        for col_id in doc["cea"]:
            winning_entity = doc["cea"][col_id][0]["id"]
            cea_results.append(
                {
                    "tab_id": table_name,
                    "row_id": row_id,
                    "col_id": col_id,
                    "entity": winning_entity,
                }
            )
    gt = pd.read_csv(
        args.ground_truth,
        delimiter=",",
        names=["tab_id", "row_id", "col_id", "entity"],
        dtype={"tab_id": str, "row_id": str, "col_id": str, "entity": str},
        keep_default_na=False,
    )
    min_row_is_one = gt["row_id"].astype(int).min() == 1
    cea_df = pd.DataFrame(cea_results)
    if cea_df["row_id"].astype(int).min() == 0 and min_row_is_one:
        cea_df["row_id"] = (cea_df["row_id"].astype(int) + 1).astype(str)
    os.makedirs("./results", exist_ok=True)
    cea_df.to_csv(f"./results/{args.dataset_name}_cea_results.csv", index=False)

    # CEA evaluate
    evaluator = CEA_Evaluator(args.ground_truth)
    result = evaluator._evaluate(
        {
            "submission_file_path": f"./results/{args.dataset_name}_cea_results.csv",
            "aicrowd_submission_id": "dummy_id",
            "aicrowd_participant_id": "dummy_participant",
        }
    )
    print(f"CEA evaluation result: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and evaluate the Alligator.")
    parser.add_argument("--dataset_name", type=str, default="test-sn-10-correct-qids-lamapi-unimib")
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "eval", "tables", "companies", "SN", "SN_gt_wd.csv"
        ),
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        help="MongoDB connection URI",
        default="mongodb://localhost:27017",
    )
    args = parser.parse_args()
    main(args)
