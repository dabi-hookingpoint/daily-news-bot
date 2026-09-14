import argparse
import os

from graph import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="발행하지 않고 로그만 남기기")
    args = parser.parse_args()

    os.environ["DRY_RUN"] = "1" if args.dry_run else "0"
    run()


if __name__ == "__main__":
    main()
