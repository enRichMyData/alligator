# Alligator

<img src="logo.webp" alt="Alligator Logo" width="400"/>

**Alligator** is a powerful Python library designed for efficient entity linking over tabular data. Whether you're working with large datasets or need to resolve entities across multiple tables, Alligator provides a scalable and easy-to-integrate solution to streamline your data processing pipeline.

## Features

- **Entity Linking:** Seamlessly link entities within tabular data using advanced ML models
- **Scalable:** Designed to handle large datasets efficiently with multiprocessing and async operations
- **Easy Integration:** Can be easily integrated into existing data processing pipelines
- **Automatic Column Classification:** Automatically detects Named Entity (NE) and Literal (LIT) columns
- **Caching System:** Built-in MongoDB caching for improved performance on repeated operations
- **Batch Processing:** Optimized batch processing for handling large volumes of data
- **ML-based Ranking:** Two-stage ML ranking (rank + rerank) for improved accuracy

## Installation

Alligator is not yet available on PyPI. To install it, clone the repository and install it manually:

```bash
git clone https://github.com/your-org/alligator.git
cd alligator
pip install -e .
```

Additionally, you need to download the SpaCy model by running:

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Using the CLI

First, create a `.env` file with the required environment variables:

```env
ENTITY_RETRIEVAL_ENDPOINT=https://lamapi.hel.sintef.cloud/lookup/entity-retrieval
OBJECT_RETRIEVAL_ENDPOINT=https://lamapi.hel.sintef.cloud/entity/objects
LITERAL_RETRIEVAL_ENDPOINT=https://lamapi.hel.sintef.cloud/entity/literals
ENTITY_RETRIEVAL_TOKEN=your_token_here
MONGO_URI=mongodb://gator-mongodb:27017
MONGO_SERVER_PORT=27017
JUPYTER_SERVER_PORT=8888
MONGO_VERSION=7.0
```

Start the MongoDB service:

```bash
docker compose up -d --build
```

Run Alligator from the CLI:

```bash
python3 -m alligator.cli \
  --gator.input_csv tables/imdb_top_1000.csv \
  --gator.entity_retrieval_endpoint "https://lamapi.hel.sintef.cloud/lookup/entity-retrieval" \
  --gator.entity_retrieval_token "your_token_here" \
  --gator.mongo_uri "mongodb://localhost:27017"
```

#### Specifying Column Types via CLI

To specify column types for your input table:

```bash
python3 -m alligator.cli \
  --gator.input_csv tables/imdb_top_1000.csv \
  --gator.entity_retrieval_endpoint "https://lamapi.hel.sintef.cloud/lookup/entity-retrieval" \
  --gator.entity_retrieval_token "your_token_here" \
  --gator.target_columns '{
    "NE": { "0": "OTHER" },
    "LIT": {
      "1": "NUMBER",
      "2": "NUMBER",
      "3": "STRING",
      "4": "NUMBER",
      "5": "STRING"
    },
    "IGNORED": ["6", "9", "10", "7", "8"]
  }' \
  --gator.mongo_uri "mongodb://localhost:27017"
```

### Using Python API

You can run the entity linking process using the `Alligator` class:

```python
import os
import time
from dotenv import load_dotenv
from alligator import Alligator

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    # Create an instance of the Alligator class
    gator = Alligator(
        input_csv="./tables/imdb_top_100.csv",
        dataset_name="cinema",
        table_name="imdb_top_100",
        entity_retrieval_endpoint=os.environ["ENTITY_RETRIEVAL_ENDPOINT"],
        entity_retrieval_token=os.environ["ENTITY_RETRIEVAL_TOKEN"],
        object_retrieval_endpoint=os.environ["OBJECT_RETRIEVAL_ENDPOINT"],
        literal_retrieval_endpoint=os.environ["LITERAL_RETRIEVAL_ENDPOINT"],
        num_workers=2,
        candidate_retrieval_limit=10,
        max_candidates_in_result=3,
        worker_batch_size=64,
        mongo_uri="mongodb://localhost:27017",
    )

    # Run the entity linking process
    tic = time.perf_counter()
    gator.run()
    toc = time.perf_counter()
    print(f"Entity linking completed in {toc - tic:.2f} seconds")
```

#### Specifying Column Types in Python

To specify column types for your input table:

```python
import os
import time
from dotenv import load_dotenv
from alligator import Alligator

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    gator = Alligator(
        input_csv="./tables/imdb_top_100.csv",
        dataset_name="cinema",
        table_name="imdb_top_100",
        entity_retrieval_endpoint=os.environ["ENTITY_RETRIEVAL_ENDPOINT"],
        entity_retrieval_token=os.environ["ENTITY_RETRIEVAL_TOKEN"],
        object_retrieval_endpoint=os.environ["OBJECT_RETRIEVAL_ENDPOINT"],
        literal_retrieval_endpoint=os.environ["LITERAL_RETRIEVAL_ENDPOINT"],
        num_workers=2,
        candidate_retrieval_limit=10,
        max_candidates_in_result=3,
        worker_batch_size=64,
        target_columns={
            "NE": {"0": "OTHER", "7": "OTHER"},
            "LIT": {"1": "NUMBER", "2": "NUMBER", "3": "STRING", "4": "NUMBER", "5": "STRING"},
            "IGNORED": ["6", "9", "10"],
        },
        mongo_uri="mongodb://localhost:27017",
    )

    # Run the entity linking process
    tic = time.perf_counter()
    gator.run()
    toc = time.perf_counter()
    print(f"Entity linking completed in {toc - tic:.2f} seconds")
```

### Configuration Parameters

#### Core Parameters
- `input_csv`: Path to input CSV file or pandas DataFrame
- `output_csv`: Path for output CSV file (optional, auto-generated if not provided)
- `dataset_name`: Name for the dataset (auto-generated if not provided)
- `table_name`: Name for the table (derived from filename if not provided)

#### Processing Parameters
- `num_workers`: Number of parallel workers for entity retrieval (default: CPU count / 2)
- `worker_batch_size`: Batch size for each worker (default: 64)
- `num_ml_workers`: Number of workers for ML ranking stages (default: 2)
- `ml_worker_batch_size`: Batch size for ML workers (default: 256)

#### API Endpoints
- `entity_retrieval_endpoint`: Endpoint for entity candidate retrieval
- `entity_retrieval_token`: Authentication token for API access
- `object_retrieval_endpoint`: Endpoint for object relationships (optional)
- `literal_retrieval_endpoint`: Endpoint for literal relationships (optional)

#### ML and Features
- `candidate_retrieval_limit`: Maximum candidates to fetch per entity (default: 16)
- `max_candidates_in_result`: Maximum candidates in final output (default: 5)
- `ranker_model_path`: Path to ranking model (optional)
- `reranker_model_path`: Path to reranking model (optional)
- `selected_features`: List of features to use (optional)
- `top_n_cta_cpa_freq`: Top N for CTA/CPA frequency features (default: 3)
- `doc_percentage_type_features`: Percentage of documents for type features (default: 1.0)

#### Output Control
- `save_output`: Whether to save results (default: True)
- `save_output_to_csv`: Whether to save to CSV format (default: True)
- `target_rows`: Specific row indices to process (optional)
- `target_columns`: Column type specifications (optional)
- `correct_qids`: Known correct QIDs for evaluation (optional)

#### Performance Tuning
- `http_session_limit`: HTTP connection pool limit (default: 32)
- `http_session_ssl_verify`: SSL verification for HTTP requests (default: False)

### Column Types

In the `target_columns` parameter, specify column types as:

- **NE (Named Entity)**: Columns containing entities to be linked
  - `"PERSON"`: Person names
  - `"ORGANIZATION"`: Organization names
  - `"LOCATION"`: Geographic locations
  - `"OTHER"`: Other named entities

- **LIT (Literal)**: Columns containing literal values
  - `"NUMBER"`: Numeric values
  - `"STRING"`: Text strings
  - `"DATE"`: Date/time values (automatically converted to `"DATETIME"`)

- **IGNORED**: Columns to skip during processing

Columns not explicitly specified are automatically classified using a built-in column classifier.

### Output Format

The output CSV includes:
- Original table columns with their data
- For each NE column, additional columns with suffixes:
  - `_id`: Entity ID (e.g., Wikidata QID)
  - `_name`: Entity name
  - `_desc`: Entity description
  - `_score`: Confidence score

Example output for a table with person names in column 0:
```
original_col_0,person_name_id,person_name_name,person_name_desc,person_name_score,...
"John Smith","Q12345","John Smith","American actor","0.95",...
```

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) first.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, feel free to open an issue on the GitHub repository.
