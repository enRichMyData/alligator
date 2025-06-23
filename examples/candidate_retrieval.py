#!/usr/bin/env python3
"""
Candidate Retrieval Test Script

This script tests the candidate retrieval functionality with configurable options for:
- Data source selection (CSV file and columns)
- Batch processing (sequential or concurrent)
- Performance tuning (batch size, delays, caching)
- API configuration (endpoints, tokens, limits)

Key Features:
- Sequential batch processing for endpoint stability
- Optional concurrent processing for improved throughput
- MongoDB caching for dramatic performance improvements
- Fuzzy retry logic for entities that initially return no results
- Comprehensive statistics and progress reporting

Usage:
    python candidate_retrieval.py --help
    python candidate_retrieval.py --sample-size 100 --num-processes 2 --use-cache
"""

import argparse
import asyncio
import os
import time

import aiohttp
import pandas as pd

from alligator import TIMEOUT
from alligator.feature import Feature
from alligator.fetchers import CandidateFetcher
from alligator.utils import clean_str


async def init_connection(http_session_limit: int = 32, http_session_ssl_verify: bool = False):
    """
    Initialize the aiohttp ClientSession with a TCPConnector.
    This function is designed to be run in an asyncio event loop.
    """
    connector = aiohttp.TCPConnector(limit=http_session_limit, ssl=http_session_ssl_verify)
    session = aiohttp.ClientSession(connector=connector, timeout=TIMEOUT)
    return session


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test candidate retrieval with sequential batch processing"
    )

    # Data source options
    parser.add_argument(
        "--csv-file",
        type=str,
        default="/Users/belerico/Projects/alligator-v2/tables/imdb_top_1000.csv",
        help="Path to CSV file to use as data source",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["Director", "Star1", "Series_Title", "Genre"],
        help="CSV columns to extract entities from",
    )
    parser.add_argument(
        "--sample-size", type=int, default=0, help="Number of entities to process (0 for all)"
    )

    # Processing options
    parser.add_argument("--batch-size", type=int, default=32, help="Number of entities per batch")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of concurrent processes to use (1 = sequential)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, help="Delay in seconds between batches"
    )

    # API options
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=50,
        help="Number of candidates to retrieve per entity",
    )
    parser.add_argument(
        "--http-session-limit", type=int, default=32, help="HTTP session connection limit"
    )
    parser.add_argument("--use-cache", action="store_true", help="Enable MongoDB caching")

    # API endpoint options
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.getenv(
            "ENTITY_RETRIEVAL_ENDPOINT", "https://lamapi.hel.sintef.cloud/lookup/entity-retrieval"
        ),
        help="Entity retrieval API endpoint",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("ENTITY_RETRIEVAL_TOKEN", "lamapi_demo_2023"),
        help="API token for authentication",
    )

    return parser.parse_args()


async def process_batch_group(batches_group, retriever, batch_delay, group_id=None):
    """
    Process a group of batches sequentially.
    This allows for parallel processing of batch groups while maintaining
    sequential processing within each group.
    """
    results = {}

    for i, batch in enumerate(batches_group):
        if group_id is not None:
            print(
                f"    Group {group_id}: Processing batch {i+1}/{len(batches_group)} with {len(batch)} entities..."
            )

        batch_result = await retriever.fetch_candidates_batch(
            entities=batch,
            fuzzies=[False for _ in range(len(batch))],
            qids=[[] for _ in range(len(batch))],
            types=[[] for _ in range(len(batch))],
        )

        if batch_result:
            results.update(batch_result)

        # Small delay between batches within the group
        if i < len(batches_group) - 1 and batch_delay > 0:
            await asyncio.sleep(batch_delay)

    return results


async def main():
    """
    Main function to run the candidate retrieval process.
    """
    args = parse_arguments()

    print(f"Configuration:")
    print(f"  CSV file: {args.csv_file}")
    print(f"  Columns: {args.columns}")
    print(f"  Sample size: {args.sample_size} (0 = all)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of processes: {args.num_processes}")
    print(f"  Delay between batches: {args.delay}s")
    print(f"  Candidates per entity: {args.num_candidates}")
    print(f"  Use cache: {args.use_cache}")
    print()

    feature = Feature(dataset_name="default_dataset", table_name="default_table")

    session = await init_connection(
        http_session_limit=args.http_session_limit,
        http_session_ssl_verify=False,
    )
    print(
        f"Initialized aiohttp session with limit {args.http_session_limit} "
        f"and SSL verify False."
    )

    retriever = CandidateFetcher(
        endpoint=args.endpoint,
        token=args.token,
        num_candidates=args.num_candidates,
        session=session,
        use_cache=args.use_cache,
        db_name="alligator_db",
        mongo_uri="mongodb://localhost:27017/",
    )

    # Load data from CSV
    try:
        df = pd.read_csv(args.csv_file)
        print(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"Error: CSV file not found: {args.csv_file}")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Check if specified columns exist
    missing_columns = [col for col in args.columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Columns not found in CSV: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return

    all_entities = df[args.columns].dropna().to_numpy().flatten().tolist()

    # Apply sample size if specified
    if args.sample_size > 0:
        all_entities = all_entities[: args.sample_size]

    data_ne = [clean_str(x) for x in all_entities]
    print(f"Processing {len(data_ne)} entities from columns {args.columns}")

    # Split into batches
    batches = [data_ne[i : i + args.batch_size] for i in range(0, len(data_ne), args.batch_size)]
    print(f"Split into {len(batches)} batches of size {args.batch_size}")

    # CONCURRENT BATCH GROUP PROCESSING:
    # If num_processes > 1, we split batches into groups and process groups concurrently
    # while maintaining sequential processing within each group.
    # This provides a balance between throughput and endpoint stability.

    tic = time.perf_counter()
    try:
        combined_results = {}

        if args.num_processes <= 1:
            # Sequential processing (original approach)
            print(f"Processing {len(batches)} batches sequentially...")
            for i, batch in enumerate(batches):
                print(f"  Processing batch {i+1}/{len(batches)} with {len(batch)} entities...")

                batch_result = await retriever.fetch_candidates_batch(
                    entities=batch,
                    fuzzies=[False for _ in range(len(batch))],
                    qids=[[] for _ in range(len(batch))],
                    types=[[] for _ in range(len(batch))],
                )

                if batch_result:
                    combined_results.update(batch_result)

                # Small delay between batches
                if i < len(batches) - 1 and args.delay > 0:
                    await asyncio.sleep(args.delay)
        else:
            # Concurrent processing with multiple processes
            print(
                f"Processing {len(batches)} batches using {args.num_processes} concurrent processes..."
            )

            # Split batches into groups for parallel processing
            batch_groups = []
            batches_per_process = max(1, len(batches) // args.num_processes)

            for i in range(0, len(batches), batches_per_process):
                group = batches[i : i + batches_per_process]
                if group:  # Only add non-empty groups
                    batch_groups.append(group)

            print(f"Split into {len(batch_groups)} batch groups for parallel processing")

            # Process batch groups concurrently
            tasks = []
            for i, batch_group in enumerate(batch_groups):
                print(
                    f"  Starting process {i+1}/{len(batch_groups)} with {len(batch_group)} batches..."
                )
                task = process_batch_group(batch_group, retriever, args.delay, group_id=i + 1)
                tasks.append(task)

            # Wait for all groups to complete
            group_results = await asyncio.gather(*tasks)

            # Combine results from all groups
            for group_result in group_results:
                if group_result:
                    combined_results.update(group_result)

        # Find entities that need fuzzy retry (no candidates returned)
        entities_needing_retry = []
        for entity_name in data_ne:
            if entity_name in combined_results and len(combined_results[entity_name]) == 0:
                entities_needing_retry.append(entity_name)

        print(f"Found {len(entities_needing_retry)} entities needing fuzzy retry")

        # Perform fuzzy retry for entities with no candidates
        if entities_needing_retry:
            print(f"Performing fuzzy retry for {len(entities_needing_retry)} entities...")

            # Split retry entities into batches
            retry_batches = [
                entities_needing_retry[i : i + args.batch_size]
                for i in range(0, len(entities_needing_retry), args.batch_size)
            ]

            print(f"Processing {len(retry_batches)} retry batches sequentially...")
            for i, batch in enumerate(retry_batches):
                print(
                    f"  Processing retry batch {i+1}/{len(retry_batches)} with {len(batch)} entities..."
                )

                batch_result = await retriever.fetch_candidates_batch(
                    entities=batch,
                    fuzzies=[True for _ in range(len(batch))],  # Use fuzzy=True for retry
                    qids=[[] for _ in range(len(batch))],
                    types=[[] for _ in range(len(batch))],
                )

                # Update combined results with retry results
                if batch_result:
                    for entity_name, candidates in batch_result.items():
                        if candidates:  # Only update if retry found candidates
                            combined_results[entity_name] = candidates

                # Small delay between retry batches
                if (
                    i < len(retry_batches) - 1 and args.delay > 0
                ):  # Don't delay after the last batch
                    await asyncio.sleep(args.delay)  # Configurable delay between batches

        toc = time.perf_counter()

        # Process and verify results
        total_entities = 0
        total_candidates = 0
        entities_with_candidates = 0
        failed_entities = 0

        for entity_name in data_ne:
            candidates = combined_results.get(entity_name, [])
            total_entities += 1
            if candidates is None:
                failed_entities += 1
                print(f"Warning: No candidates returned for '{entity_name}'")
            elif len(candidates) == 0:
                print(f"Info: Empty candidate list for '{entity_name}'")
            else:
                total_candidates += len(candidates)
                entities_with_candidates += 1

        print(f"Fetched candidates in {toc - tic:.4f} seconds")
        print(f"Results summary:")
        print(f"  Total entities processed: {total_entities}")
        print(f"  Entities with candidates: {entities_with_candidates}")
        print(f"  Failed entities: {failed_entities}")
        print(f"  Total candidates retrieved: {total_candidates}")
        if total_entities > 0:
            print(f"  Average candidates per entity: {total_candidates/total_entities:.2f}")
            print(f"  Success rate: {entities_with_candidates/total_entities*100:.1f}%")

        # Show sample results
        if combined_results:
            sample_entities = list(combined_results.keys())[:3]
            print(f"\nSample results for first 3 entities:")
            for entity in sample_entities:
                candidates = combined_results.get(entity, [])
                print(f"  '{entity}': {len(candidates)} candidates")
                if candidates and len(candidates) > 0:
                    top_candidate = candidates[0]
                    candidate_name = top_candidate.get("name", "Unknown")
                    candidate_id = top_candidate.get("id", "No ID")
                    print(f"    Top candidate: {candidate_name} ({candidate_id})")
        else:
            print("Warning: No results found")

    except Exception as e:
        print(f"Error during candidate retrieval: {e}")
        import traceback

        traceback.print_exc()

    await session.close()


if __name__ == "__main__":
    asyncio.run(main())
