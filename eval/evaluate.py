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
        projection={"el_results": 1, "table_name": 1, "dataset_name": 1, "row_id": 1},
    )
    results = []
    for doc in cursor:
        table_name = doc["table_name"]
        row_id = doc["row_id"]
        for col_id in doc["el_results"]:
            winning_entity = doc["el_results"][col_id][0]["id"]
            results.append(
                {
                    "tab_id": table_name,
                    "row_id": row_id,
                    "col_id": col_id,
                    "entity": winning_entity,
                }
            )
    df = pd.DataFrame(results)
    if df["row_id"].astype(int).min() == 0:
        df["row_id"] = (df["row_id"].astype(int) + 1).astype(str)
    os.makedirs("./results", exist_ok=True)
    df.to_csv(f"./results/{args.dataset_name}_results.csv", index=False)

    # Evaluate
    evaluator = CEA_Evaluator(args.ground_truth)
    result = evaluator._evaluate(
        {
            "submission_file_path": f"./results/{args.dataset_name}_results.csv",
            "aicrowd_submission_id": "dummy_id",
            "aicrowd_participant_id": "dummy_participant",
        }
    )
    print(f"Evaluation result: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and evaluate the Alligator.")
    parser.add_argument("--dataset_name", type=str, default="htr1-correct-qids")
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "eval", "tables", "HardTablesR1", "Valid", "gt", "cea_gt.csv"
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
