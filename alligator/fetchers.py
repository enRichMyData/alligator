import asyncio
import hashlib
import json
import traceback
from functools import lru_cache
from typing import Dict, List, Optional
from urllib.parse import quote

import aiohttp

from alligator.database import DatabaseAccessMixin
from alligator.mongo import MongoCache, MongoWrapper
from alligator.types import LiteralsData, ObjectsData


@lru_cache(maxsize=int(2**31) - 1)
def get_cache_key(**kwargs) -> str:
    """
    Generate a unique cache key based on arbitrary keyword arguments.

    Args:
        **kwargs: Arbitrary keyword arguments representing request parameters.

    Returns:
        str: A SHA256 hash string uniquely representing the parameter set.
    """
    # Serialize parameters with sorted keys for consistency
    serialized = json.dumps(kwargs, sort_keys=True, default=str)
    key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return key


class CandidateFetcher(DatabaseAccessMixin):
    """
    Extracted logic for fetching candidates.
    Takes a reference to the Alligator instance so we can access
    DB, feature, caching, etc.
    """

    def __init__(
        self,
        endpoint: str,
        token: str,
        num_candidates: int,
        session: aiohttp.ClientSession,
        use_cache: bool = True,
        **kwargs,
    ):
        self.endpoint = endpoint
        self.token = token
        self.num_candidates = num_candidates
        self.session = session
        self._db_name = kwargs.get("db_name", "alligator_db")
        self._mongo_uri = kwargs.get("mongo_uri", "mongodb://gator-mongodb:27017")
        self.input_collection = kwargs.get("input_collection", "input_data")
        self.cache_collection = kwargs.get("cache_collection", "candidate_cache")
        self.mongo_wrapper = MongoWrapper(self._mongo_uri, self._db_name)
        self.use_cache = use_cache
        if self.use_cache:
            self.cache = MongoCache(self._mongo_uri, self._db_name, self.cache_collection)
        else:
            self.cache = None

    def get_candidate_cache(self) -> Optional[MongoCache]:
        return self.cache

    async def fetch_candidates_batch(
        self,
        entities: List[str],
        fuzzies: List[bool],
        qids: List[List[str]],
        types: Optional[List[List[str]]] = None,
    ) -> Dict[str, List[dict]]:
        """
        Fetch candidates for multiple entities in a batch.

        Args:
            entities: List of entity names to find candidates for
            fuzzies: Whether to use fuzzy matching for each entity
            qids: Known correct QIDs for each entity (optional)
            types: List of type QIDs to constrain the search for each entity (optional)

        Returns:
            Dictionary mapping entity names to lists of candidate dictionaries
        """
        if types is None:
            types = [[] for _ in entities]
        return await self.fetch_candidates_batch_async(entities, fuzzies, qids, types)

    async def _fetch_candidates(
        self,
        entity_name,
        fuzzy,
        qid,
        types: Optional[str] = None,
        use_cache: bool = False,
        kind: str = "entity",
    ) -> tuple[str, List[dict]]:
        """
        This used to be Alligator._fetch_candidates. Logic unchanged.
        """
        encoded_entity_name = quote(entity_name)
        url = (
            f"{self.endpoint}?name={encoded_entity_name}"
            f"&limit={self.num_candidates}&fuzzy={fuzzy}"
            f"&token={self.token}"
            f"&kind={kind}"
            f"&cache={str(use_cache)}"
        )
        if qid:
            url += f"&ids={qid}"
        if types:
            url += f"&types={types}"

        backoff = 1
        for attempts in range(5):
            try:
                async with self.session.get(url) as response:
                    response.raise_for_status()
                    fetched_candidates = await response.json()

                    # Ensure all QIDs are included by adding placeholders for missing ones
                    required_qids = qid.split() if qid else []
                    existing_qids = {c["id"] for c in fetched_candidates if c.get("id")}
                    missing_qids = set(required_qids) - existing_qids

                    for missing_qid in missing_qids:
                        fetched_candidates.append(
                            {
                                "id": missing_qid,
                                "name": None,
                                "description": None,
                                "features": None,
                                "is_placeholder": True,
                            }
                        )

                    # Merge with existing cache if present
                    if cache := self.get_candidate_cache():
                        cache_key = get_cache_key(
                            endpoint=self.endpoint,
                            token=self.token,
                            num_candidates=self.num_candidates,
                            entity_name=entity_name,
                            fuzzy=fuzzy,
                            qid=qid,
                            types=types or "",
                            kind=kind,
                        )
                        cache.put(cache_key, fetched_candidates)

                    return entity_name, fetched_candidates

            except Exception:
                if attempts == 4:
                    self.mongo_wrapper.log_to_db(
                        "FETCH_CANDIDATES_ERROR",
                        f"Error fetching candidates for {entity_name}",
                        traceback.format_exc(),
                        attempt=attempts + 1,
                    )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 16)

        # If all attempts fail
        return entity_name, []

    async def fetch_candidates_batch_async(
        self, entities, fuzzies, qids: List[List[str]], types: Optional[List[List[str]]] = None
    ):
        """
        This used to be Alligator.fetch_candidates_batch_async.
        """
        if types is None:
            types = [[] for _ in entities]

        results = {}
        to_fetch = []

        # Decide which entities need to be fetched
        for entity_name, fuzzy, entity_qids, entity_types in zip(entities, fuzzies, qids, types):
            qid_str = " ".join(entity_qids) if entity_qids else ""
            types_str = " ".join(entity_types) if entity_types else ""

            if cache := self.get_candidate_cache():
                cache_key = get_cache_key(
                    endpoint=self.endpoint,
                    token=self.token,
                    num_candidates=self.num_candidates,
                    entity_name=entity_name,
                    fuzzy=fuzzy,
                    qid=qid_str,
                    types=types_str,
                    kind="entity",
                )
                cached_result = cache.get(cache_key)
            else:
                cached_result = None

            if cached_result is not None:
                if len(entity_qids) > 0:
                    cached_qids = {c["id"] for c in cached_result if "id" in c}
                    if all(q in cached_qids for q in entity_qids):
                        results[entity_name] = cached_result
                    else:
                        to_fetch.append((entity_name, fuzzy, qid_str, types_str))
                else:
                    results[entity_name] = cached_result
            else:
                to_fetch.append((entity_name, fuzzy, qid_str, types_str))

        if not to_fetch:
            return self._remove_placeholders(results)

        tasks = []
        for entity_name, fuzzy, qid_str, types_str in to_fetch:
            tasks.append(self._fetch_candidates(entity_name, fuzzy, qid_str, types_str))
        done = await asyncio.gather(*tasks, return_exceptions=False)
        for entity_name, candidates in done:
            results[entity_name] = candidates

        return self._remove_placeholders(results)

    def _remove_placeholders(self, results):
        """This used to be Alligator._remove_placeholders."""
        for entity_name, candidates in results.items():
            results[entity_name] = [c for c in candidates if not c.get("is_placeholder", False)]
        return results


class ObjectFetcher(DatabaseAccessMixin):
    """
    Fetcher for retrieving object information from LAMAPI.
    """

    def __init__(self, endpoint: str, token: str, session: aiohttp.ClientSession, **kwargs):
        self.endpoint = endpoint
        self.token = token
        self.session = session
        self._db_name = kwargs.get("db_name", "alligator_db")
        self._mongo_uri = kwargs.get("mongo_uri", "mongodb://gator-mongodb:27017")
        self.cache_collection = kwargs.get("cache_collection", "object_cache")
        self.mongo_wrapper = MongoWrapper(self._mongo_uri, self._db_name)
        self.cache = MongoCache(self._mongo_uri, self._db_name, self.cache_collection)

    def get_object_cache(self):
        return self.cache

    async def fetch_objects(self, entity_ids: List[str]) -> Dict[str, ObjectsData]:
        """
        Fetch object data for multiple entity IDs.

        Args:
            entity_ids: List of entity IDs to fetch object data for

        Returns:
            Dictionary mapping entity IDs to their object data
        """
        if not entity_ids:
            return {}

        # Filter out already cached IDs
        cache = self.get_object_cache()
        to_fetch = []
        results = {}

        for entity_id in entity_ids:
            cached_result = cache.get(entity_id)
            if cached_result is not None:
                results[entity_id] = cached_result
            else:
                to_fetch.append(entity_id)

        if not to_fetch:
            return results

        # Prepare batch request for non-cached IDs
        url = f"{self.endpoint}?token={self.token}"
        backoff = 1
        for attempts in range(5):
            try:
                # Use the correct JSON structure: {"json": [...]} instead of {"ids": [...]}
                request_data = {"json": to_fetch}
                async with self.session.post(url, json=request_data) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Update cache and results
                    for entity_id, objects_data in data.items():
                        cache.put(entity_id, objects_data)
                        results[entity_id] = objects_data

                    return results

            except Exception:
                if attempts == 4:
                    self.mongo_wrapper.log_to_db(
                        "FETCH_OBJECTS_ERROR",
                        "Error fetching objects for entities",
                        traceback.format_exc(),
                        attempt=attempts + 1,
                    )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 16)

        # If all attempts fail, return what we have
        return results


class LiteralFetcher(DatabaseAccessMixin):
    """
    Fetcher for retrieving literal information from LAMAPI.
    """

    def __init__(self, endpoint: str, token: str, session: aiohttp.ClientSession, **kwargs):
        self.endpoint = endpoint
        self.token = token
        self.session = session
        self._db_name = kwargs.get("db_name", "alligator_db")
        self._mongo_uri = kwargs.get("mongo_uri", "mongodb://gator-mongodb:27017")
        self.cache_collection = kwargs.get("cache_collection", "literal_cache")
        self.mongo_wrapper = MongoWrapper(self._mongo_uri, self._db_name)
        self.cache = MongoCache(self._mongo_uri, self._db_name, self.cache_collection)

    def get_literal_cache(self):
        return self.cache

    async def fetch_literals(self, entity_ids: List[str]) -> Dict[str, LiteralsData]:
        """
        Fetch literal values for multiple entity IDs.

        Args:
            entity_ids: List of entity IDs to fetch literal values for

        Returns:
            Dictionary mapping entity IDs to their literal values
        """
        if not entity_ids:
            return {}

        # Filter out already cached IDs
        cache = self.get_literal_cache()
        to_fetch = []
        results = {}

        for entity_id in entity_ids:
            cached_result = cache.get(entity_id)
            if cached_result is not None:
                results[entity_id] = cached_result
            else:
                to_fetch.append(entity_id)

        if not to_fetch:
            return results

        # Prepare batch request for non-cached IDs
        url = f"{self.endpoint}?token={self.token}"
        backoff = 1
        for attempts in range(5):
            try:
                # Use the correct JSON structure: {"json": [...]} instead of {"ids": [...]}
                request_data = {"json": to_fetch}
                async with self.session.post(url, json=request_data) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Update cache and results
                    for entity_id, literals_data in data.items():
                        cache.put(entity_id, literals_data)
                        results[entity_id] = literals_data

                    return results

            except Exception:
                if attempts == 4:
                    self.mongo_wrapper.log_to_db(
                        "FETCH_LITERALS_ERROR",
                        "Error fetching literals for entities",
                        traceback.format_exc(),
                        attempt=attempts + 1,
                    )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 16)

        # If all attempts fail, return what we have
        return results
