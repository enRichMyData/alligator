#!/usr/bin/env python
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
        default="mongodb://mongodb:27017",
    )
    args = parser.parse_args()

    print("🚀 Starting the entity linking process...")
    gator = Alligator(**args.gator)
    gator.run()
    print("✅ Entity linking process completed.")


if __name__ == "__main__":
    main()
