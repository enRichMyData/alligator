import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from evaluators.cpa_wd import CPA_Evaluator

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
        projection={"cpa": 1, "table_name": 1, "dataset_name": 1, "row_id": 1},
    )
    cpa_results = []
    seen_tables = set()
    for doc in cursor:
        table_name = doc["table_name"]
        doc["row_id"]
        if "cpa" not in doc:
            print(f"Skipping document {doc['_id']} due to missing 'cpa'.")
            continue
        for subj_col_id in doc["cpa"]:
            for obj_col_id in doc["cpa"][subj_col_id]:
                if (table_name, subj_col_id, obj_col_id) in seen_tables:
                    continue
                seen_tables.add((table_name, subj_col_id, obj_col_id))
                winning_entity = doc["cpa"][subj_col_id][obj_col_id][0]
                cpa_results.append(
                    {
                        "tab_id": table_name,
                        "sub_col_id": subj_col_id,
                        "obj_col_id": obj_col_id,
                        "property": winning_entity,
                    }
                )
    cpa_df = pd.DataFrame(cpa_results)
    os.makedirs("./results", exist_ok=True)
    cpa_df.to_csv(f"./results/{args.dataset_name}_cpa_results.csv", index=False)

    # CEA evaluate
    evaluator = CPA_Evaluator(args.ground_truth)
    result = evaluator._evaluate(
        {
            "submission_file_path": f"./results/{args.dataset_name}_cpa_results.csv",
            "aicrowd_submission_id": "dummy_id",
            "aicrowd_participant_id": "dummy_participant",
        }
    )
    print(f"CPA evaluation result: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and evaluate the Alligator.")
    parser.add_argument("--dataset_name", type=str, default="htr1-correct-qids")
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "eval", "tables", "HardTablesR1", "Valid", "gt", "cpa_gt.csv"
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
