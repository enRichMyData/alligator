import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from evaluators.cta_wd import CTA_Evaluator

from alligator import PROJECT_ROOT
from alligator.database import DatabaseManager
from alligator.mongo import MongoWrapper

load_dotenv(PROJECT_ROOT)


def main(args: argparse.Namespace):
    types_from_best = args.types_from_best
    db = DatabaseManager.get_database(args.mongo_uri, "alligator_db")
    mongo = MongoWrapper(mongo_uri=args.mongo_uri, db_name="alligator_db")
    projection = {
        "cta": 1,
        "table_name": 1,
        "dataset_name": 1,
    }
    if types_from_best:
        projection["cea"] = 1
    cursor = mongo.find_documents(
        collection=db.get_collection("input_data"),
        query={"dataset_name": args.dataset_name, "status": "DONE", "rerank_status": "DONE"},
        projection=projection,
    )
    cta_results = []
    if types_from_best:
        type_frequencies = {}
        for doc in cursor:
            table_name = doc["table_name"]
            for col_id in doc["cea"]:
                winning_types = doc["cea"][col_id][0]["types"]
                if not winning_types:
                    continue
                key = f"{table_name}_{col_id}"
                if key not in type_frequencies:
                    type_frequencies[key] = {}
                for winning_type in winning_types:
                    winning_type_id = winning_type["id"]
                    if winning_type_id not in type_frequencies[key]:
                        type_frequencies[key][winning_type_id] = 0
                    type_frequencies[key][winning_type_id] += 1
        for tab_col, types_freq in type_frequencies.items():
            table_name, col_id = tab_col.split("_")
            winning_entity = max(types_freq, key=types_freq.get)
            cta_results.append(
                {
                    "tab_id": table_name,
                    "col_id": col_id,
                    "entity": winning_entity,
                }
            )
    else:
        seen_tables = set()
        for doc in cursor:
            table_name = doc["table_name"]
            if "cta" not in doc:
                print(f"Skipping document {doc['_id']} due to missing 'cta'.")
                continue
            for col_id in doc["cta"]:
                if (table_name, col_id) in seen_tables:
                    continue
                seen_tables.add((table_name, col_id))
                winning_entity = doc["cta"][col_id][0]
                cta_results.append(
                    {
                        "tab_id": table_name,
                        "col_id": col_id,
                        "entity": winning_entity,
                    }
                )
    cta_df = pd.DataFrame(cta_results)
    os.makedirs("./results", exist_ok=True)
    cta_df.to_csv(f"./results/{args.dataset_name}_cta_results.csv", index=False)

    # CTA evaluate
    evaluator = CTA_Evaluator(
        args.ground_truth,
        ancestor_path=args.ancestor_path,
        descendent_path=args.descendent_path,
    )
    result = evaluator._evaluate(
        {
            "submission_file_path": f"./results/{args.dataset_name}_cta_results.csv",
            "aicrowd_submission_id": "dummy_id",
            "aicrowd_participant_id": "dummy_participant",
        }
    )
    print(f"CTA evaluation result: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and evaluate the Alligator.")
    parser.add_argument("--dataset_name", type=str, default="htr1-correct-qids")
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "eval", "tables", "HardTablesR1", "Valid", "gt", "cta_gt.csv"
        ),
    )
    parser.add_argument(
        "--ancestor_path",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "eval", "tables", "HardTablesR1", "Valid", "gt", "cta_gt_ancestor.json"
        ),
    )
    parser.add_argument(
        "--descendent_path",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "eval", "tables", "HardTablesR1", "Valid", "gt", "cta_gt_descendent.json"
        ),
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        help="MongoDB connection URI",
        default="mongodb://localhost:27017",
    )
    parser.add_argument(
        "--types_from_best", action="store_true", help="Use types from best CEA results"
    )
    args = parser.parse_args()
    main(args)
