from typing import Dict

from pymongo import MongoClient
from pymongo.database import Database

from alligator.log import get_logger


class DatabaseAccessMixin:
    """Mixin to provide standardized database access."""

    def get_db(self):
        """Get MongoDB database connection from the centralized pool."""

        return DatabaseManager.get_database(self._mongo_uri, self._db_name)


class DatabaseManager:
    """
    Singleton database manager that maintains connection pools to MongoDB.
    Provides centralized access to database connections with proper connection pooling.
    """

    _instance = None
    _connections: Dict[str, MongoClient] = {}
    _databases: Dict[str, Database] = {}
    _logger = get_logger("database_manager")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_connection(cls, uri: str) -> MongoClient:
        """
        Get or create a MongoDB client connection with proper pooling settings.

        Args:
            uri: MongoDB connection string

        Returns:
            MongoDB client with connection pooling configured
        """
        if uri not in cls._connections:
            # Configure connection pool with sensible defaults
            # These can be tuned based on your application needs
            cls._connections[uri] = MongoClient(
                uri,
                maxPoolSize=50,  # Max connections in the pool
                minPoolSize=10,  # Min connections to maintain
                maxIdleTimeMS=30000,  # Close idle connections after 30s
                socketTimeoutMS=30000,  # Socket timeout
                connectTimeoutMS=5000,  # Connection timeout
                retryWrites=True,  # Retry write operations
                waitQueueTimeoutMS=2000,  # How long to wait for a connection
            )
        return cls._connections[uri]

    @classmethod
    def get_database(cls, uri: str, db_name: str) -> Database:
        """
        Get a database instance with connection pooling.

        Args:
            uri: MongoDB connection string
            db_name: Database name

        Returns:
            MongoDB database instance
        """
        key = f"{uri}:{db_name}"
        if key not in cls._databases:
            client = cls.get_connection(uri)
            cls._databases[key] = client[db_name]
        return cls._databases[key]

    @classmethod
    def close_all_connections(cls) -> None:
        """Close all open connections when shutting down."""
        for uri, client in cls._connections.items():
            try:
                client.close()
                cls._logger.info(f"Closed MongoDB connection to {uri}")
            except Exception as e:
                cls._logger.error(f"Error closing MongoDB connection: {e}")

        cls._connections.clear()
        cls._databases.clear()
