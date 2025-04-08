# Alligator

<img src="logo.webp" alt="Alligator Logo" width="400"/>

**Alligator** is a powerful Python library designed for efficient entity linking over tabular data. Whether you're working with large datasets or need to resolve entities across multiple tables, Alligator provides a scalable and easy-to-integrate solution to streamline your data processing pipeline.


## Features

- **Entity Linking:** Seamlessly link entities within tabular data.
- **Scalable:** Designed to handle large datasets efficiently.
- **Easy Integration:** Can be easily integrated into existing data processing pipelines.

## Installation

Alligator is not yet available on PyPI. To install it, clone the repository and install it manually:

```bash
git clone https://github.com/your-org/alligator.git
cd alligator
pip install -e .
```

Additionally, one needs to download the SpaCy model by running the following code:

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Using the CLI
You can run the entity linking process via the command line interface (CLI) as follows:

First, create a `.env` file with the required environment variables:

```
ENTITY_RETRIEVAL_ENDPOINT=https://lamapi.hel.sintef.cloud/lookup/entity-retrieval
OBJECT_RETRIEVAL_ENDPOINT=https://lamapi.hel.sintef.cloud/entity/objects
LITERAL_RETRIEVAL_ENDPOINT=https://lamapi.hel.sintef.cloud/entity/literals
LITERAL_RETRIEVAL_TOKEN=lamapi_demo_2023
MONGO_URI=mongodb://gator-mongodb:27017
MONGO_SERVER_PORT=27017
JUPYTER_SERVER_PORT=8888
MONGO_VERSION=7.0
```

Then, start the services (MongoDB service is the one needed) with

```bash
docker compose up -d --build
```

Finally, run Alligator from the CLI with:

```bash
python3 -m alligator.cli \
  --gator.input_csv tables/imdb_top_1000.csv \
  --gator.entity_retrieval_endpoint "https://lamapi.hel.sintef.cloud/lookup/entity-retrieval" \
  --gator.entity_retrieval_token "lamapi_demo_2023" \
  --gator.mongo_uri "localhost:27017"
```

#### Specifying Column Types via CLI
To specify column types for your input table, use the following command:

```bash
python3 -m alligator.cli \
  --gator.input_csv tables/imdb_top_1000.csv \
  --gator.entity_retrieval_endpoint "https://lamapi.hel.sintef.cloud/lookup/entity-retrieval" \
  --gator.entity_retrieval_token "lamapi_demo_2023" \
  --gator.columns_type '{
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
  --gator.mongo_uri "localhost:27017"
```

### Using Python API
You can also run the entity linking process using the `Alligator` class in Python:

```python
import os
import time

from dotenv import load_dotenv

from alligator import Alligator

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    # Load the CSV file into a DataFrame
    file_path = "./tables/imdb_top_100.csv"

    # Create an instance of the Alligator class
    gator = Alligator(
        input_csv=file_path,
        dataset_name="cinema",
        table_name="imdb_top_100",
        entity_retrieval_endpoint=os.environ["ENTITY_RETRIEVAL_ENDPOINT"],
        entity_retrieval_token=os.environ["ENTITY_RETRIEVAL_TOKEN"],
        object_retrieval_endpoint=os.environ["OBJECT_RETRIEVAL_ENDPOINT"],
        literal_retrieval_endpoint=os.environ["LITERAL_RETRIEVAL_ENDPOINT"],
        max_workers=2,
        candidate_retrieval_limit=10,
        max_candidates_in_result=3,
        batch_size=256,
        mongo_uri="localhost:27017",
    )

    # Run the entity linking process
    tic = time.perf_counter()
    gator.run()
    toc = time.perf_counter()
    print("Elapsed time:", toc - tic)
    print("Entity linking process completed.")
```

### Specifying Column Types
If you want to specify column types for your input table, use the following example:

```python
import os
import time

from dotenv import load_dotenv

from alligator import Alligator

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    # Load the CSV file into a DataFrame
    file_path = "./tables/imdb_top_100.csv"

    # Create an instance of the Alligator class
    gator = Alligator(
        input_csv=file_path,
        dataset_name="cinema",
        table_name="imdb_top_100",
        entity_retrieval_endpoint=os.environ["ENTITY_RETRIEVAL_ENDPOINT"],
        entity_retrieval_token=os.environ["ENTITY_RETRIEVAL_TOKEN"],
        object_retrieval_endpoint=os.environ["OBJECT_RETRIEVAL_ENDPOINT"],
        literal_retrieval_endpoint=os.environ["LITERAL_RETRIEVAL_ENDPOINT"],
        max_workers=2,
        candidate_retrieval_limit=10,
        max_candidates_in_result=3,
        batch_size=256,
        columns_type={
            "NE": {"0": "OTHER", "7": "OTHER"},
            "LIT": {"1": "NUMBER", "2": "NUMBER", "3": "STRING", "4": "NUMBER", "5": "STRING"},
            "IGNORED": ["6", "9", "10", "7", "8"],
        },
        mongo_uri="localhost:27017",
    )

    # Run the entity linking process
    tic = time.perf_counter()
    gator.run()
    toc = time.perf_counter()
    print("Elapsed time:", toc - tic)
    print("Entity linking process completed.")

```

In the `columns_type` parameter, one has to specify **for every column index** whether it is a Named-Entity (NE) column or a Literal (LIT) one. All the columns that are not specified neither as NE nor as LIT will be considered as IGNORED columns.

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) first.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, feel free to open an issue on the GitHub repository.
