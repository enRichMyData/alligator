import asyncio
import traceback
from urllib.parse import quote

import aiohttp

from alligator import MY_TIMEOUT
from alligator.feature import Feature
from alligator.mongo import MongoCache, MongoWrapper
from alligator.utils import tokenize_text


class CandidateFetcher:
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
        feature: Feature,
        **kwargs,
    ):
        self.endpoint = endpoint
        self.token = token
        self.num_candidates = num_candidates
        self.feature = feature
        self._db_name = kwargs.get("db_name", "crocodile_db")
        self._mongo_uri = kwargs.get("mongo_uri", "mongodb://mongodb:27017")
        self.input_collection = kwargs.get("input_collection", "input_data")
        self.cache_collection = kwargs.get("cache_collection", "candidate_cache")
        self.mongo_wrapper = MongoWrapper(self._mongo_uri, self._db_name)

    def get_db(self):
        """Get MongoDB database connection for current process"""
        from alligator.mongo import MongoConnectionManager

        client = MongoConnectionManager.get_client(self._mongo_uri)
        return client[self._db_name]

    def get_candidate_cache(self):
        return MongoCache(self.get_db(), self.cache_collection)

    def fetch_candidates_batch(self, entities, row_texts, fuzzies, qids):
        return asyncio.run(self.fetch_candidates_batch_async(entities, row_texts, fuzzies, qids))

    async def _fetch_candidates(
        self, entity_name, row_text, fuzzy, qid, session, cache: bool = True
    ):
        """
        This used to be Alligator._fetch_candidates. Logic unchanged.
        """
        encoded_entity_name = quote(entity_name)
        url = (
            f"{self.endpoint}?name={encoded_entity_name}"
            f"&limit={self.num_candidates}&fuzzy={fuzzy}"
            f"&token={self.token}"
        )
        if qid:
            url += f"&ids={qid}"

        backoff = 1
        for attempts in range(5):
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    candidates = await response.json()
                    row_tokens = set(tokenize_text(row_text))
                    fetched_candidates = self.feature.process_candidates(
                        candidates, entity_name, row_tokens
                    )
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
                    cache = self.get_candidate_cache()
                    cache_key = f"{entity_name}_{fuzzy}"
                    cached_result = cache.get(cache_key)

                    if cached_result:
                        all_candidates = {c["id"]: c for c in cached_result if "id" in c}
                        for c in fetched_candidates:
                            if c.get("id"):
                                all_candidates[c["id"]] = c
                        merged_candidates = list(all_candidates.values())
                    else:
                        merged_candidates = fetched_candidates

                    cache.put(cache_key, merged_candidates)
                    return entity_name, merged_candidates

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

    async def fetch_candidates_batch_async(self, entities, row_texts, fuzzies, qids):
        """
        This used to be Alligator.fetch_candidates_batch_async.
        """
        results = {}
        cache = self.get_candidate_cache()
        to_fetch = []

        # Decide which entities need to be fetched
        for entity_name, fuzzy, row_text, qid_str in zip(entities, fuzzies, row_texts, qids):
            cache_key = f"{entity_name}_{fuzzy}"
            cached_result = cache.get(cache_key)
            forced_qids = qid_str.split() if qid_str else []

            if cached_result is not None:
                if forced_qids:
                    cached_qids = {c["id"] for c in cached_result if "id" in c}
                    if all(q in cached_qids for q in forced_qids):
                        results[entity_name] = cached_result
                    else:
                        to_fetch.append((entity_name, fuzzy, row_text, qid_str))
                else:
                    results[entity_name] = cached_result
            else:
                to_fetch.append((entity_name, fuzzy, row_text, qid_str))

        if not to_fetch:
            return self._remove_placeholders(results)

        async with aiohttp.ClientSession(
            timeout=MY_TIMEOUT, connector=aiohttp.TCPConnector(ssl=False, limit=10)
        ) as session:
            tasks = []
            for entity_name, fuzzy, row_text, qid_str in to_fetch:
                tasks.append(
                    self._fetch_candidates(entity_name, row_text, fuzzy, qid_str, session)
                )
            done = await asyncio.gather(*tasks, return_exceptions=False)
            for entity_name, candidates in done:
                results[entity_name] = candidates

        return self._remove_placeholders(results)

    def _remove_placeholders(self, results):
        """This used to be Alligator._remove_placeholders."""
        for entity_name, candidates in results.items():
            results[entity_name] = [c for c in candidates if not c.get("is_placeholder", False)]
        return results