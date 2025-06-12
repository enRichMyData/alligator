import time
from typing import Dict, List, Tuple, cast

import pandas as pd
from column_classifier import ColumnClassifier

from alligator.config import AlligatorConfig
from alligator.database import DatabaseAccessMixin
from alligator.mongo import MongoWrapper


class DataManager(DatabaseAccessMixin):
    """Manages data onboarding, validation, and classification."""

    def __init__(self, config: AlligatorConfig):
        self.config = config
        self._mongo_uri = config.database.mongo_uri or "mongodb://gator-mongodb:27017/"
        self._db_name = config.database.db_name or "alligator_db"
        self.mongo_wrapper = MongoWrapper(
            self._mongo_uri,
            self._db_name,
            config.database.input_collection,
            config.database.error_log_collection,
        )
        self.mongo_wrapper.create_indexes()

    def onboard_data(self) -> None:
        """Efficiently load data into MongoDB using batched inserts."""
        start_time = time.perf_counter()

        # Get database connection
        db = self.get_db()
        input_collection = db[self.config.database.input_collection]

        # Ensure we have valid dataset and table names
        dataset_name = self.config.data.dataset_name or "default_dataset"
        table_name = self.config.data.table_name or "default_table"

        # Step 1: Determine data source and extract sample for classification
        if isinstance(self.config.data.input_csv, pd.DataFrame):
            df = self.config.data.input_csv
            sample = df
            total_rows = len(df)
            is_csv_path = False
        else:
            if self.config.data.input_csv is None:
                raise ValueError("Input CSV path cannot be None")
            sample = pd.read_csv(self.config.data.input_csv, nrows=32)
            total_rows = -1
            is_csv_path = True

        print(f"Onboarding {total_rows} rows for dataset '{dataset_name}', table '{table_name}'")

        # Step 2: Perform column classification
        classified_columns = self._classify_columns(sample)
        ne_cols, lit_cols, ignored_cols, context_cols = self._process_column_types(
            sample, classified_columns
        )

        # Step 3: Process all chunks using the generator
        processed_rows = self._process_data_chunks(
            input_collection,
            ne_cols,
            lit_cols,
            ignored_cols,
            context_cols,
            is_csv_path,
            total_rows,
            start_time,
            dataset_name,
            table_name,
        )

        total_time = time.perf_counter() - start_time
        print(
            f"Data onboarding complete for dataset '{dataset_name}' "
            f"and table '{table_name}' - {processed_rows} rows in {total_time:.1f}s"
        )

    def _classify_columns(self, sample: pd.DataFrame) -> Dict[str, str]:
        """Classify columns using the column classifier."""
        classifier = ColumnClassifier(model_type="fast")
        classification_results = classifier.classify_multiple_tables([sample])
        table_classification = classification_results[0].get("table_1", {})

        classified_columns = {}
        for idx, col in enumerate(sample.columns):
            col_result = table_classification.get(col, {})
            classification = col_result.get("classification", "UNKNOWN")
            classified_columns[str(idx)] = classification

        return classified_columns

    def _process_column_types(
        self, sample: pd.DataFrame, classified_columns: Dict[str, str]
    ) -> Tuple[Dict[str, str], Dict[str, str], List[str], List[str]]:
        """Process column classifications into NE, LIT, and ignored columns."""
        ne_cols: Dict[str, str] = {}
        lit_cols: Dict[str, str] = {}
        ignored_cols: List[str] = []

        ne_types = {"PERSON", "OTHER", "ORGANIZATION", "LOCATION"}
        lit_types = {"NUMBER", "DATE", "STRING"}

        for idx, col in enumerate(sample.columns):
            classification = classified_columns.get(str(idx), "UNKNOWN")
            if classification in ne_types:
                ne_cols[str(idx)] = classification
            elif classification in lit_types:
                if classification == "DATE":
                    classification = "DATETIME"
                lit_cols[str(idx)] = classification

        # Override with target columns if provided
        if self.config.data.target_columns is not None:
            target_ne = self.config.data.target_columns.get("NE", {})
            if target_ne:
                ne_cols = cast(Dict[str, str], target_ne)
                for col in ne_cols:
                    if col not in classified_columns:
                        ne_cols[col] = classified_columns.get(col, "UNKNOWN")

            target_lit = self.config.data.target_columns.get("LIT", {})
            if target_lit:
                lit_cols = cast(Dict[str, str], target_lit)
                for col in lit_cols:
                    if not lit_cols[col]:
                        lit_cols[col] = classified_columns.get(col, "UNKNOWN")

            target_ignored = self.config.data.target_columns.get("IGNORED", [])
            if target_ignored:
                ignored_cols = target_ignored

        # Calculate context columns
        all_recognized_cols = set(ne_cols.keys()) | set(lit_cols.keys())
        all_cols = set([str(i) for i in range(len(sample.columns))])
        if len(all_recognized_cols) != len(all_cols):
            ignored_cols.extend(list(all_cols - all_recognized_cols))
        ignored_cols = list(set(ignored_cols))
        context_cols = list(set([str(i) for i in range(len(sample.columns))]) - set(ignored_cols))
        context_cols = sorted(context_cols, key=lambda x: int(x))

        return ne_cols, lit_cols, ignored_cols, context_cols

    def _get_data_chunks(self, is_csv_path: bool, total_rows: int):
        """Generator that yields chunks of rows, handling both DF and CSV."""
        if self.config.data.dry_run:
            if is_csv_path:
                if self.config.data.input_csv is not None:
                    yield pd.read_csv(self.config.data.input_csv, nrows=1), 0
            else:
                if isinstance(self.config.data.input_csv, pd.DataFrame):
                    yield self.config.data.input_csv.iloc[:1], 0
        else:
            if is_csv_path:
                chunk_size = 2048
                row_count = 0
                if self.config.data.input_csv is not None:
                    for chunk in pd.read_csv(self.config.data.input_csv, chunksize=chunk_size):
                        yield chunk, row_count
                        row_count += len(chunk)
            else:
                if isinstance(self.config.data.input_csv, pd.DataFrame):
                    chunk_size = (
                        1024 if total_rows > 100000 else 2048 if total_rows > 10000 else 4096
                    )
                    total_chunks = (total_rows + chunk_size - 1) // chunk_size
                    for chunk_idx in range(total_chunks):
                        chunk_start = chunk_idx * chunk_size
                        chunk_end = min(chunk_start + chunk_size, total_rows)
                        yield self.config.data.input_csv.iloc[chunk_start:chunk_end], chunk_start

    def _process_data_chunks(
        self,
        input_collection,
        ne_cols: Dict[str, str],
        lit_cols: Dict[str, str],
        ignored_cols: List[str],
        context_cols: List[str],
        is_csv_path: bool,
        total_rows: int,
        start_time: float,
        dataset_name: str,
        table_name: str,
    ) -> int:
        """Process data chunks and insert into MongoDB."""
        processed_rows = 0
        chunk_idx = 0

        for chunk, start_idx in self._get_data_chunks(is_csv_path, total_rows):
            chunk_idx += 1
            documents = []

            for i, (_, row) in enumerate(chunk.iterrows()):
                row_id = start_idx + i
                if (
                    str(row_id) not in self.config.data.target_rows
                    and self.config.data.target_rows
                ):
                    continue

                document = {
                    "dataset_name": dataset_name,
                    "table_name": table_name,
                    "row_id": str(row_id),
                    "data": row.tolist(),
                    "classified_columns": {
                        "NE": ne_cols,
                        "LIT": lit_cols,
                        "IGNORED": ignored_cols,
                    },
                    "context_columns": context_cols,
                    "status": "TODO",
                }

                # Add correct QIDs if available
                correct_qids = {}
                for col_id, _ in ne_cols.items():
                    key = f"{row_id}-{col_id}"
                    if key in self.config.data.correct_qids:
                        correct_qids[key] = self.config.data.correct_qids[key]
                        if isinstance(correct_qids[key], str):
                            correct_qids[key] = [correct_qids[key]]
                        else:
                            correct_qids[key] = list(set(correct_qids[key]))
                document["correct_qids"] = correct_qids
                documents.append(document)

            if documents:
                try:
                    input_collection.insert_many(documents, ordered=False)
                    chunk_size = len(documents)
                    processed_rows += chunk_size
                    elapsed = time.perf_counter() - start_time
                    rows_per_second = processed_rows / elapsed if elapsed > 0 else 0

                    if is_csv_path:
                        print(
                            f"Chunk {chunk_idx}: Processed {chunk_size} rows "
                            f"(total: {processed_rows}) ({rows_per_second:.1f} rows/sec)"
                        )
                    else:
                        chunk_start = start_idx + 1
                        chunk_end = start_idx + chunk_size
                        total_chunks = (total_rows + chunk_size - 1) // chunk_size
                        print(
                            f"Chunk {chunk_idx}/{total_chunks}: "
                            f"Onboarded rows {chunk_start}-{chunk_end} "
                            f"({rows_per_second:.1f} rows/sec)"
                        )
                except Exception as e:
                    print(f"Error inserting batch {chunk_idx}: {str(e)}")
                    if "duplicate key" not in str(e).lower():
                        raise

        return processed_rows
