#!/usr/bin/env python
import time

from dotenv import load_dotenv
from jsonargparse import ArgumentParser

from alligator import Alligator

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
    args = parser.parse_args()

    print("🚀 Starting the entity linking process...")
    tic = time.perf_counter()
    gator = Alligator(**args.gator)
    gator.run()
    toc = time.perf_counter()
    print(f"✅ Entity linking process completed in {toc - tic:0.4f} seconds.")


if __name__ == "__main__":
    main()
