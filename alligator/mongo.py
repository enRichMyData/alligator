from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar

import pymongo
from pymongo import ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.results import DeleteResult, InsertManyResult, InsertOneResult, UpdateResult

from alligator.database import DatabaseAccessMixin, DatabaseManager

T = TypeVar("T")


class MongoCache:
    """MongoDB-based cache for storing key-value pairs."""

    def __init__(self, db: Database, collection_name: str) -> None:
        self.collection: Collection = db[collection_name]
        self.collection.create_index("key", unique=True)

    def get(self, key: str) -> Optional[Any]:
        result: Optional[Dict[str, Any]] = self.collection.find_one({"key": key})
        if result:
            return result["value"]
        return None

    def put(self, key: str, value: Any) -> None:
        self.collection.update_one({"key": key}, {"$set": {"value": value}}, upsert=True)


class MongoConnectionManager:
    """Legacy connection manager, now delegating to DatabaseManager."""

    @classmethod
    def get_client(cls, uri: str) -> pymongo.MongoClient:
        """Get a MongoDB client with connection pooling."""
        return DatabaseManager.get_connection(uri)

    @classmethod
    def close_connection(cls) -> None:
        """Close all connections."""
        DatabaseManager.close_all_connections()


class MongoWrapper(DatabaseAccessMixin):
    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        error_log_collection_name: str = "error_logs",
    ) -> None:
        self._mongo_uri: str = mongo_uri
        self._db_name: str = db_name
        self.error_log_collection_name: str = error_log_collection_name

    def update_document(
        self,
        collection: Collection,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False,
    ) -> UpdateResult:
        return collection.update_one(query, update, upsert=upsert)

    def update_documents(
        self,
        collection: Collection,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False,
    ) -> UpdateResult:
        return collection.update_many(query, update, upsert=upsert)

    def find_documents(
        self,
        collection: Collection,
        query: Dict[str, Any],
        projection: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        def query_function(
            query: Dict[str, Any], projection: Optional[Dict[str, Any]] = None
        ) -> List[Dict[str, Any]]:
            cursor = collection.find(query, projection)
            if limit is not None:
                cursor = cursor.limit(limit)
            return list(cursor)

        return query_function(query, projection)

    def count_documents(self, collection: Collection, query: Dict[str, Any]) -> int:
        return collection.count_documents(query)

    def find_one_document(
        self,
        collection: Collection,
        query: Dict[str, Any],
        projection: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        return collection.find_one(query, projection=projection)

    def find_one_and_update(
        self,
        collection: Collection,
        query: Dict[str, Any],
        update: Dict[str, Any],
        return_document: bool = False,
    ) -> Optional[Dict[str, Any]]:
        return collection.find_one_and_update(query, update, return_document=return_document)

    def insert_one_document(
        self, collection: Collection, document: Dict[str, Any]
    ) -> InsertOneResult:
        return collection.insert_one(document)

    def insert_many_documents(
        self, collection: Collection, documents: List[Dict[str, Any]]
    ) -> InsertManyResult:
        return collection.insert_many(documents)

    def delete_documents(self, collection: Collection, query: Dict[str, Any]) -> DeleteResult:
        return collection.delete_many(query)

    def log_to_db(
        self, level: str, message: str, trace: Optional[str] = None, attempt: Optional[int] = None
    ) -> None:
        db: Database = self.get_db()
        log_collection: Collection = db[self.error_log_collection_name]
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(),
            "level": level,
            "message": message,
            "traceback": trace,
        }
        if attempt is not None:
            log_entry["attempt"] = attempt
        log_collection.insert_one(log_entry)

    # Ensure indexes for uniqueness and performance
    def create_indexes(self):
        db: Database = self.get_db()
        input_collection: Collection = db["input_data"]

        input_collection.create_index([("dataset_name", ASCENDING), ("table_name", ASCENDING)])
        input_collection.create_index(
            [("dataset_name", ASCENDING), ("table_name", ASCENDING), ("row_id", ASCENDING)],
            unique=True,
        )
        input_collection.create_index([("status", ASCENDING)])
        input_collection.create_index([("rank_status", ASCENDING)])
        input_collection.create_index([("rerank_status", ASCENDING)])
        input_collection.create_index(
            [
                ("dataset_name", ASCENDING),
                ("table_name", ASCENDING),
                ("status", ASCENDING),
                ("rank_status", ASCENDING),
                ("rerank_status", ASCENDING),
            ]
        )
