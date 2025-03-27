import os

from pymongo import ASCENDING, MongoClient  # added ASCENDING import


def get_db():
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db = client["alligator_backend_db"]

    # Ensure indexes are created
    db.datasets.create_index([("dataset_name", ASCENDING)], unique=True)  # updated index field
    db.tables.create_index([("dataset_name", ASCENDING), ("table_name", ASCENDING)], unique=True)

    try:
        yield db
    finally:
        client.close()


def get_alligator_db():
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db = client["alligator_db"]
    try:
        yield db
    finally:
        client.close()
