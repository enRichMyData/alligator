#!/usr/bin/env python
import time

from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from alligator import Alligator
from alligator.logging import get_logger

load_dotenv()
logger = get_logger("cli")


def main():
    parser = ArgumentParser()
    parser.add_class_arguments(Alligator, "gator")
    parser.add_argument(
        "--gator.mongo-uri",
        type=str,
        help="MongoDB connection URI",
        default="mongodb://gator-mongodb:27017",
    )
    args = parser.parse_args()

    logger.info("🚀 Starting the entity linking process...")
    tic = time.perf_counter()
    gator = Alligator(**args.gator)
    gator.run()
    toc = time.perf_counter()
    logger.info(f"✅ Entity linking process completed in {toc - tic:0.4f} seconds.")


if __name__ == "__main__":
    main()
