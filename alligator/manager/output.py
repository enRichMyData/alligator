from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from alligator.config import AlligatorConfig
from alligator.database import DatabaseAccessMixin
from alligator.logging import get_logger


class OutputManager(DatabaseAccessMixin):
    """Manages output generation and saving."""

    def __init__(self, config: AlligatorConfig):
        self.config = config
        self.logger = get_logger("output_manager")
        self._mongo_uri = config.database.mongo_uri or "mongodb://gator-mongodb:27017/"
        self._db_name = config.database.db_name or "alligator_db"

    def save_output(self) -> List[Dict[str, Any]]:
        """Save output to CSV and return extracted rows."""
        if not self.config.data.save_output:
            return [{}]

        self.logger.info("Saving output...")
        db = self.get_db()
        input_collection = db[self.config.database.input_collection]

        dataset_name = self.config.data.dataset_name or "default_dataset"
        table_name = self.config.data.table_name or "default_table"

        header = None
        if isinstance(self.config.data.input_csv, pd.DataFrame):
            header = self.config.data.input_csv.columns.tolist()
        elif isinstance(self.config.data.input_csv, (str, Path)):
            header = pd.read_csv(self.config.data.input_csv, nrows=0).columns.tolist()

        # Get first document to determine column count if header is still None
        sample_doc = input_collection.find_one(
            {"dataset_name": dataset_name, "table_name": table_name}
        )
        if not sample_doc:
            self.logger.warning("No documents found for the specified dataset and table.")
            return []

        if header is None:
            self.logger.warning(
                "Could not extract header from input table, using generic column names."
            )
            header = [f"col_{i}" for i in range(len(sample_doc["data"]))]

        # Write directly to CSV without storing in memory
        if self.config.data.save_output_to_csv and isinstance(
            self.config.data.output_csv, (str, Path)
        ):
            first_row = True
            writer = None
            with open(self.config.data.output_csv, "w", newline="", encoding="utf-8") as csvfile:
                for row_data in self.document_generator(header):
                    if first_row:
                        import csv

                        writer = csv.DictWriter(csvfile, fieldnames=row_data.keys())
                        writer.writeheader()
                        first_row = False

                    if writer is not None:
                        writer.writerow(row_data)
            return [{}]
        else:
            return list(self.document_generator(header))

    def document_generator(self, header: List[str]):
        dg = self.get_db()
        input_collection = dg[self.config.database.input_collection]
        cursor = input_collection.find(
            {
                "dataset_name": self.config.data.dataset_name,
                "table_name": self.config.data.table_name,
            },
            projection={"data": 1, "cea": 1, "classified_columns.NE": 1},
        ).batch_size(512)
        for doc in cursor:
            yield self._extract_row_data(doc, header)

    def _extract_row_data(self, doc, header: list[str]) -> Dict[str, Any]:
        """Extract row data from a MongoDB document.

        Encapsulates the common logic for formatting a row from a document.
        """
        # Create base row data with original values
        row_data = dict(zip(header, doc["data"]))
        el_results = doc.get("cea", {})

        # Add entity linking results
        for col_idx, col_type in doc["classified_columns"].get("NE", {}).items():
            col_index = int(col_idx)
            col_header = header[col_index]

            id_field = f"{col_header}_id"
            name_field = f"{col_header}_name"
            desc_field = f"{col_header}_desc"
            score_field = f"{col_header}_score"

            # Get first candidate or empty placeholder
            candidate = el_results.get(col_idx, [{}])[0]

            row_data[id_field] = candidate.get("id", "")
            row_data[name_field] = candidate.get("name", "")
            row_data[desc_field] = candidate.get("description", "")
            row_data[score_field] = candidate.get("score", 0)

        return row_data
