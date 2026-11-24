import argparse
import sys
from pathlib import Path

from ner_common import load_texts, read_annotations, validate_rows


def main():
    parser = argparse.ArgumentParser(description="Validate NER annotation CSV files.")
    parser.add_argument("--texts", required=True, help="Directory containing one .txt file per document")
    parser.add_argument("--annotations", nargs="+", required=True, help="Annotation CSV files")
    args = parser.parse_args()

    texts = load_texts(args.texts)
    any_errors = False

    for csv_path in args.annotations:
        print(f"\n=== Validating {csv_path} ===")
        rows = read_annotations(csv_path)
        errors, warnings = validate_rows(rows, texts)

        print(f"Rows: {len(rows)}")
        print(f"Errors: {len(errors)}")
        print(f"Warnings: {len(warnings)}")

        for w in warnings:
            print(f"WARNING: {w}")
        for e in errors:
            print(f"ERROR: {e}")

        if errors:
            any_errors = True

    if any_errors:
        print("\nValidation failed with errors.")
        sys.exit(1)
    else:
        print("\nAll annotation files passed validation.")


if __name__ == "__main__":
    main()