#!/usr/bin/env python
import time

from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from alligator import Alligator
from alligator.log import get_logger, setup_logging

load_dotenv()


def main():
    parser = ArgumentParser()
    parser.add_class_arguments(Alligator, "gator")
    parser.add_argument(
        "--gator.mongo-uri",
        type=str,
        help="MongoDB connection URI",
        default="mongodb://gator-mongodb:27017",
    )
    parser.add_argument(
        "--disable-logging",
        action="store_true",
        help="Completely disable all logging for maximum performance",
    )
    args = parser.parse_args()

    # Setup logging based on CLI arguments
    if args.disable_logging:
        setup_logging(disable_logging=True)
        logger = None  # Don't create logger if disabled
    else:
        setup_logging()
        logger = get_logger("cli")

    # Only log if logging is enabled
    if logger:
        logger.info("🚀 Starting the entity linking process...")

    tic = time.perf_counter()
    gator = Alligator(**args.gator)
    gator.run()
    toc = time.perf_counter()

    if logger:
        logger.info(f"✅ Entity linking process completed in {toc - tic:0.4f} seconds.")


if __name__ == "__main__":
    main()
